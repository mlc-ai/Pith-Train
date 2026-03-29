"""
Backward compatibility test for DeepSeek V2 Lite after context parallelism changes.

Verifies that:
1. The 4D device mesh (pp, dp, cp, ep) works correctly with cp_size=1
2. The updated apply_fsdp with CP-aware mesh dimensions shards correctly
3. DeepSeek V2 Lite model construction does not receive cp_group
4. A full forward+backward step through DualPipeV produces valid loss and gradients

Launch with:
    torchrun --nproc-per-node=2 tests/test_deepseek_backward_compat.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from pithtrain.dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
from pithtrain.layers.group_linear import GroupLinear
from pithtrain.models.deepseek_v2_lite import DeepseekV2LiteModel, DeepseekV2LiteMoEGate
from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.modules.training import apply_fsdp
from pithtrain.operators.mla import MLA


def fill_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, GroupLinear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, DeepseekV2LiteMoEGate):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def criterion(output, target):
    return F.mse_loss(output.float(), target.float()).clone()


def main():
    cfg_ns, ctx_ns = SimpleNamespace(), SimpleNamespace()
    cfg_ns.distributed = DistributedCfg()
    cfg_ns.distributed.pipeline_parallel_size = 1
    cfg_ns.distributed.expert_parallel_size = 1
    cfg_ns.distributed.context_parallel_size = 1
    ctx_ns.distributed = DistributedCtx()

    with distributed_context(cfg_ns, ctx_ns):
        ctx = ctx_ns.distributed
        rank = ctx.rank

        if rank == 0:
            print("[INFO] Testing DeepSeek V2 Lite backward compatibility")
            print(
                f"[INFO] Mesh shape: (pp={ctx.pp_size}, dp={ctx.dp_size}, cp={ctx.cp_size}, ep={ctx.ep_size})"
            )

        torch.manual_seed(1234)
        torch.set_default_device(torch.cuda.current_device())
        dtype = torch.bfloat16
        B, S = 2, 128
        num_chunks = 4

        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "examples",
            "pretrain_language_model",
            "deepseek_v2_lite",
            "deepseek_v2_lite",
        )
        config = AutoConfig.from_pretrained(config_path)
        config.num_hidden_layers = 4
        config.ep_size = 1

        if rank == 0:
            H = config.num_attention_heads
            DQ = config.qk_nope_head_dim + config.qk_rope_head_dim
            DV = config.v_head_dim
            MLA.autotune(B, S, H, DQ, DV, DQ**-0.5)
        torch.distributed.barrier()

        pp_size = ctx.pp_size
        pp_rank = ctx.pp_rank
        ep_group = ctx.device_mesh.get_group("ep")
        pp_group = ctx.device_mesh.get_group("pp")
        hidden_size = config.hidden_size

        modules = [
            DeepseekV2LiteModel(config, pp_size * 2, pp_rank, ep_group),
            DeepseekV2LiteModel(config, pp_size * 2, pp_size * 2 - 1 - pp_rank, ep_group),
        ]
        modules = nn.Sequential(*modules)
        modules.to(dtype=dtype)
        modules.apply(fill_weights)

        apply_fsdp(modules, ctx.device_mesh)

        model = DualPipeV(modules, const_inputs=(), pp_group=pp_group, ep_group=ep_group)
        set_p2p_tensor_shapes([(B, S, hidden_size)])
        set_p2p_tensor_dtype(dtype)

        vocab_size = config.vocab_size
        x = torch.randint(0, vocab_size, (num_chunks * B, S)) if pp_rank == 0 else None
        labels = torch.randn(num_chunks * B, S, vocab_size, dtype=dtype) if pp_rank == 0 else None

        if rank == 0:
            print("[INFO] Running DualPipeV step...", flush=True)
        torch.distributed.barrier()

        loss, _ = model.step(
            x, num_chunks=num_chunks, criterion=criterion, labels=(labels,), return_outputs=False
        )

        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())

        passed = True
        if pp_rank == 0:
            loss_val = loss.mean().item()
            if not torch.isfinite(loss).all():
                print(f"[FAIL] rank-{rank}: loss is not finite: {loss}", flush=True)
                passed = False
            else:
                print(f"[PASS] rank-{rank}: loss={loss_val:.4f}", flush=True)
        else:
            if loss is not None:
                print(f"[FAIL] rank-{rank}: loss should be None on non-first PP rank", flush=True)
                passed = False

        if has_grads == 0:
            print(f"[FAIL] rank-{rank}: no gradients computed ({total_params} params)", flush=True)
            passed = False
        else:
            print(
                f"[PASS] rank-{rank}: {has_grads}/{total_params} params have gradients", flush=True
            )

        all_passed = torch.tensor(1 if passed else 0, device="cuda")
        torch.distributed.all_reduce(all_passed, op=torch.distributed.ReduceOp.MIN)

        if rank == 0:
            if all_passed.item():
                print("\nAll backward compatibility checks passed.")
            else:
                print("\nSome checks FAILED.")

    sys.exit(0 if all_passed.item() else 1)


if __name__ == "__main__":
    main()
