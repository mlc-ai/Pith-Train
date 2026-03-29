"""
Testing script for DualPipe with FSDP.
The loss and gradients are compared with a reference implementation.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch
import torch.distributed.fsdp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig

from pithtrain.dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
from pithtrain.layers.factory import ModelImplMode
from pithtrain.layers.group_linear import GroupLinear
from pithtrain.models.deepseek_v2_lite import DeepseekV2LiteModel, DeepseekV2LiteMoEGate
from pithtrain.models.qwen3_30b_a3b import Qwen3MoeGate, Qwen3MoeModel
from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.operators.mla import MLA


def fill_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, GroupLinear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, (DeepseekV2LiteMoEGate, Qwen3MoeGate)):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def calculate_difference(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = output.to(torch.float32)
    target = target.to(torch.float32)
    return F.mse_loss(output, target).clone()


def reference_step(
    x: torch.Tensor,
    l: torch.Tensor,  # noqa: E741
    model: Union[DeepseekV2LiteModel, Qwen3MoeModel],
    chunks: int,
):
    ys, ls = [], []
    for micro_x, micro_l in zip(x.chunk(chunks), l.chunk(chunks)):
        micro_y = model(micro_x)
        loss = criterion(micro_y, micro_l)
        loss.backward()
        ys.append(micro_y)
        ls.append(loss)
    return torch.stack(ls), torch.cat(ys, 0)


def shard_layers(layers: nn.ModuleDict, stage_id: int, num_stages: int, config):
    num_local_layers = [config.num_hidden_layers // num_stages for _ in range(num_stages)]
    layers_per_stage_residual = config.num_hidden_layers % num_stages
    for i in range(layers_per_stage_residual):
        num_local_layers[(1 - (i % 2) * 2) * (i // 2) - (i % 2)] += 1
    layer_id_begin = sum(num_local_layers[:stage_id])
    layer_id_end = layer_id_begin + num_local_layers[stage_id]
    return nn.ModuleDict({str(i): layers[str(i)] for i in range(layer_id_begin, layer_id_end)})


def shard_experts(model, ep_rank, ep_size):
    for name, child in model.named_children():
        if isinstance(child, GroupLinear):
            experts_per_ep_rank = child.num_groups // ep_size
            new_mod = GroupLinear(experts_per_ep_rank, child.in_features, child.out_features)
            expert_begin = ep_rank * experts_per_ep_rank
            expert_end = (ep_rank + 1) * experts_per_ep_rank
            new_mod.weight.data = child.weight.data[expert_begin:expert_end]
            new_mod.weight.grad = child.weight.grad[expert_begin:expert_end]
            setattr(model, name, new_mod)
        else:
            shard_experts(child, ep_rank, ep_size)


def apply_fsdp(model, mesh: torch.distributed.DeviceMesh, dtype):
    # MoE params are sharded by EP, we only additionally shard on the DP dimension
    moe_fsdp_mesh = mesh["dp"]
    # For other params, we shard on the both DP and EP dimensions
    other_fsdp_mesh = mesh["dp", "ep"]._flatten()
    mp = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    # FSDP recommends shard models from the bottom to the top.
    for i in range(2):
        assert isinstance(model[i], (DeepseekV2LiteModel, Qwen3MoeModel))
        if model[i].embed_tokens is not None:
            fully_shard(
                model[i].embed_tokens,
                mesh=other_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp,
            )
        if model[i].norm is not None:
            assert model[i].lm_head is not None
            fully_shard(
                model[i].norm, mesh=other_fsdp_mesh, reshard_after_forward=True, mp_policy=mp
            )
            fully_shard(
                model[i].lm_head, mesh=other_fsdp_mesh, reshard_after_forward=True, mp_policy=mp
            )
        for layer in model[i].layers.values():
            if hasattr(layer.mlp, "experts"):
                fully_shard(
                    layer.mlp.experts, mesh=moe_fsdp_mesh, reshard_after_forward=False, mp_policy=mp
                )
            fully_shard(layer, mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_attn")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_mlp")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_aggregate")
        fully_shard(model[i], mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
    return model


def main(ctx: DistributedCtx, model_name: str):
    """
    Main testing function.

    Parameters
    ----------
    ctx : DistributedCtx
        Distributed context.
    model_name : str
        Model name or local config path.
    """

    pp_group = ctx.device_mesh.get_group("pp")
    ep_group = ctx.device_mesh.get_group("ep")
    dp_size, pp_size, ep_size = ctx.dp_size, ctx.pp_size, ctx.ep_size
    pp_rank, ep_rank = ctx.pp_rank, ctx.ep_rank

    if ctx.rank == 0:
        print("[INFO] Testing FSDP x DualPipeV x EP with model: %s" % model_name, flush=True)
        print(
            "[INFO] DP size: %d, PP size: %d, EP size: %d." % (dp_size, pp_size, ep_size),
            flush=True,
        )
    torch.distributed.barrier()

    torch.manual_seed(1234)
    torch.set_default_device(torch.cuda.current_device())
    dtype = torch.bfloat16

    num_chunks, micro_batch_size, sequence_length = 20, 3, 128

    config_path = Path(__file__).resolve().parent.parent / model_name
    config = AutoConfig.from_pretrained(config_path)
    B, S = micro_batch_size, sequence_length

    if config.model_type == "deepseek_v2":
        ModelClass = DeepseekV2LiteModel
        if ctx.rank == 0:
            H = config.num_attention_heads
            DQ = config.qk_nope_head_dim + config.qk_rope_head_dim
            DV = config.v_head_dim
            MLA.autotune(B, S, H, DQ, DV, DQ**-0.5)
        config.num_hidden_layers = min(config.num_hidden_layers, 8)
    elif config.model_type == "qwen3_moe":
        ModelClass = Qwen3MoeModel
        config.num_hidden_layers = min(config.num_hidden_layers, 8)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    torch.distributed.barrier()
    torch.manual_seed(1234)

    hidden_size, vocab_size = config.hidden_size, config.vocab_size

    # Create the dummy inputs.
    full_x = torch.randint(
        0, vocab_size, (ep_size * num_chunks * micro_batch_size, sequence_length)
    )
    full_l = torch.randn(
        ep_size * num_chunks * micro_batch_size, sequence_length, vocab_size, dtype=dtype
    )
    local_x = full_x.reshape(ep_size, num_chunks * micro_batch_size, sequence_length)[ep_rank]
    local_l = full_l.reshape(ep_size, num_chunks * micro_batch_size, sequence_length, vocab_size)[
        ep_rank
    ]

    # Create the reference full model.
    config.ep_size = 1

    full_modules = ModelClass(config, num_stages=1, stage_id=0)

    full_modules.to(dtype=dtype)
    config.ep_size = ep_size
    full_modules.apply(fill_weights)

    # Run the reference step.
    if ctx.rank == 0:
        print("[INFO] Running the reference step.", flush=True)
    torch.distributed.barrier()

    ModelImplMode.use_reference_fwd = True
    loss_ref, output_ref = reference_step(full_x, full_l, full_modules, num_chunks * ep_size)

    if ctx.rank == 0:
        print("[INFO] Completed the reference step.", flush=True)
    torch.distributed.barrier()

    # Setup DualPipeV.
    ModelImplMode.use_reference_fwd = False
    set_p2p_tensor_shapes([(micro_batch_size, sequence_length, hidden_size)])
    set_p2p_tensor_dtype(dtype)

    # Shard the full modules whose weights and gradients will be used for checking.
    num_stages = pp_size * 2
    local_full_modules = []

    local_full_modules.append(ModelClass(config, num_stages=num_stages, stage_id=pp_rank))
    local_full_modules.append(
        ModelClass(config, num_stages=num_stages, stage_id=num_stages - 1 - pp_rank)
    )

    local_full_modules = nn.Sequential(*local_full_modules)
    if pp_rank == 0:
        local_full_modules[0].embed_tokens = full_modules.embed_tokens
        local_full_modules[1].norm = full_modules.norm
        local_full_modules[1].lm_head = full_modules.lm_head
    local_full_modules[0].layers = shard_layers(full_modules.layers, pp_rank, num_stages, config)
    local_full_modules[1].layers = shard_layers(
        full_modules.layers, num_stages - 1 - pp_rank, num_stages, config
    )
    if ep_size > 1:
        shard_experts(local_full_modules[0], ep_rank=ep_rank, ep_size=ep_size)
        shard_experts(local_full_modules[1], ep_rank=ep_rank, ep_size=ep_size)

    # Create the local modules with the same weights but zero gradients.
    local_modules = []

    local_modules.append(
        ModelClass(config, num_stages=num_stages, stage_id=pp_rank, ep_group=ep_group)
    )
    local_modules.append(
        ModelClass(
            config, num_stages=num_stages, stage_id=num_stages - 1 - pp_rank, ep_group=ep_group
        )
    )

    local_modules = nn.Sequential(*local_modules)
    local_modules.to(dtype=dtype)
    local_modules[0].load_state_dict(local_full_modules[0].state_dict())
    local_modules[1].load_state_dict(local_full_modules[1].state_dict())
    local_modules.zero_grad()
    apply_fsdp(local_modules, ctx.device_mesh, dtype)

    # Wrap the modules with DualPipeV.
    kwargs = dict()
    kwargs["pp_group"] = pp_group
    kwargs["ep_group"] = ep_group
    kwargs["const_inputs"] = ()
    dualpipev_model = DualPipeV(local_modules, **kwargs)

    # Run the DualPipeV step.
    kwargs = dict()
    kwargs["num_chunks"] = num_chunks
    kwargs["criterion"] = criterion
    kwargs["return_outputs"] = False
    local_x = None if pp_rank != 0 else local_x
    local_l = None if pp_rank != 0 else local_l
    kwargs["labels"] = (local_l,)

    if ctx.rank == 0:
        print("[INFO] Running the DualPipeV step.", flush=True)
    torch.distributed.barrier()

    loss, outputs = dualpipev_model.step(local_x, **kwargs)

    if ctx.rank == 0:
        print("[INFO] Completed the DualPipeV step.", flush=True)
    torch.distributed.barrier()

    # Validate the loss.
    if pp_rank == 0:
        loss_ref = loss_ref.reshape(ep_size, -1)
        loss_ref = loss_ref[ep_rank]
        print("[INFO] rank-%d, loss: %s, loss_ref: %s" % (ctx.rank, loss, loss_ref), flush=True)
        assert torch.allclose(loss, loss_ref, rtol=1e-3, atol=1e-3)
    else:
        assert loss is None

    if ctx.rank == 0:
        print("[INFO] Loss matches the reference.", flush=True)
    torch.distributed.barrier()

    # Validate the gradients.
    eps = 1e-2
    largest_diff = 0
    largest_diff_param = None

    for (n, p), p_ref in zip(local_modules.named_parameters(), local_full_modules.parameters()):
        if p.grad is None:
            print("[warn] rank-%d, Parameter %s doesn't have a gradient, skipping." % (ctx.rank, n))
            continue
        p_grad = p.grad
        if isinstance(p_grad, torch.distributed.tensor.DTensor):
            p_grad = p_grad.full_tensor()
        if ".experts." not in n and ep_size > 1:
            p_grad = p_grad.clone()
            torch.distributed.all_reduce(p_grad, group=ep_group)
        if torch.all(p_grad == 0) and torch.all(p_ref.grad == 0):
            print("[warn] rank-%d, Parameter %s has all-zero gradient, skipping." % (ctx.rank, n))
            continue
        diff = calculate_difference(p_grad, p_ref.grad)
        if diff > largest_diff:
            largest_diff = diff
            largest_diff_param = n
        if diff > eps:
            print(
                "[ERROR] rank-%d, Parameter %s grad mismatch: diff=%.6f, eps=%.6f, p_grad:%s..., p_ref.grad:%s..."
                % (ctx.rank, n, diff, eps, p_grad.flatten()[:5], p_ref.grad.flatten()[:5])
            )
    assert largest_diff < eps

    for rank in range(ctx.world_size):
        if rank == ctx.rank:
            print(
                "[INFO] rank-%d, Gradient check completed. Largest diff = %.6f for param %s."
                % (ctx.rank, largest_diff, largest_diff_param)
            )
        torch.distributed.barrier()

    if ctx.rank == 0:
        print("[INFO] All gradients match the reference.", flush=True)
    torch.distributed.barrier()


if __name__ == "__main__":
    models = []
    models.append("examples/pretrain_language_model/deepseek-v2-lite/config.json")
    models.append("examples/pretrain_language_model/qwen3-30b-a3b/config.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--model", type=str, choices=models, required=True)
    parsed = parser.parse_args()

    cfg, ctx = SimpleNamespace(), SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.expert_parallel_size = parsed.ep_size
    ctx.distributed = DistributedCtx()

    with distributed_context(cfg, ctx):
        main(ctx.distributed, parsed.model)
