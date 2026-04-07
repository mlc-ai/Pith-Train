"""Checkpoint conversion from HuggingFace to DCP and vice versa."""

import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

from pithtrain.config import SlottedDefault
from pithtrain.modules.logging import LoggingCfg, LoggingCtx, logging_context


@dataclass(init=False, slots=True)
class ConvertCheckpointCfg(SlottedDefault):
    """
    Configuration for checkpoint conversion.
    """

    operation: Literal["hf2dcp", "dcp2hf"]
    """
    Conversion operation: "hf2dcp" or "dcp2hf".
    """

    load_path: Path
    """
    Source checkpoint directory to load from.
    """

    save_path: Path
    """
    Destination checkpoint directory to save to.
    """

    max_shard_size: int = 8 * 1024**3
    """
    Maximum shard size in bytes for dcp2hf (default 8GB).
    """

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """
    Logging configuration.
    """


@dataclass(init=False, slots=True)
class ConvertCheckpointCtx(SlottedDefault):
    """
    Context for checkpoint conversion.
    """

    logging: LoggingCtx = field(default_factory=LoggingCtx)
    """
    Active logging context.
    """


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize MXFP4-packed blocks with per-row shared 8-bit exponent.

    Low nibble first; scales are biased by 127.  Adapted from
    Megatron-Bridge ``gpt_oss_bridge._dequantize_mxfp4``.
    """
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    FP4_VALUES = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)
        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


def _is_gpt_oss(load_path: Path) -> bool:
    config_path = Path(load_path, "config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type") == "gpt_oss"
    return False


def _hf2dcp_gpt_oss(load_path: Path, save_path: Path, stdout: Logger) -> None:
    with open(Path(load_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    stdout.info(
        "Converting GPT-OSS HF checkpoint from %s (%d shards)" % (load_path, len(shard_files))
    )

    raw: Dict[str, torch.Tensor] = dict()
    for i, shard_file in enumerate(sorted(shard_files), start=1):
        stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
        with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    # MXFP4 on-disk is [E, out, in] with the quantization axis along in
    # (32 FP4 values per block).  Dequant drops the last two dims:
    #   gate_up_proj_blocks [E, 2*inter, G, 16] → [E, 2*inter, hidden]
    #   down_proj_blocks    [E, hidden,  G, 16] → [E, hidden, inter]
    dequantized: Dict[str, torch.Tensor] = dict()
    seen_blocks = set()
    for key in sorted(raw.keys()):
        if key.endswith("_blocks"):
            base = key.removesuffix("_blocks")
            scales_key = base + "_scales"
            if scales_key in raw:
                stdout.info("Dequantizing MXFP4: %s" % base)
                flat = _dequantize_mxfp4(raw[key], raw[scales_key])
                dequantized[base] = flat.contiguous()
                seen_blocks.add(key)
                seen_blocks.add(scales_key)

    for key, tensor in raw.items():
        if key not in seen_blocks and key not in dequantized:
            dequantized[key] = tensor

    model_state_dict: Dict[str, torch.Tensor] = dict()
    for key, tensor in dequantized.items():
        canon = key.removeprefix("model.")

        if canon.endswith(
            (
                ".mlp.experts.gate_up_proj",
                ".mlp.experts.gate_up_proj_bias",
                ".mlp.experts.down_proj",
                ".mlp.experts.down_proj_bias",
            )
        ):
            for idx in range(tensor.shape[0]):
                expert_key = canon.replace(".experts.", ".experts.%d." % idx)
                model_state_dict[expert_key] = tensor[idx].contiguous()
        else:
            model_state_dict[canon] = tensor

    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))


def hf2dcp(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """
    Convert HuggingFace checkpoint to DCP format.
    """
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)

    if _is_gpt_oss(load_path):
        _hf2dcp_gpt_oss(load_path, save_path, stdout)
        return

    with open(Path(load_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    stdout.info("Converting HF checkpoint from %s (%d shards)" % (load_path, len(shard_files)))

    model_state_dict: Dict[str, torch.Tensor] = dict()
    for i, shard_file in enumerate(sorted(shard_files), start=1):
        stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
        with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
            for key in f.keys():
                model_state_dict[key.removeprefix("model.")] = f.get_tensor(key)

    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))


def _is_gpt_oss_dcp(metadata) -> bool:
    return any("gate_up_proj" in k for k in metadata.state_dict_metadata.keys())


def _stack_experts(state_dict: Dict[str, torch.Tensor], stdout: Logger) -> Dict[str, torch.Tensor]:
    """Stack per-expert canonical keys along dim 0 and transpose 3-D weights
    to HF's live ``[E, in, out]`` layout.  Our DCP stores ``[E, out, in]``
    (TorchTitan / TE convention); the transpose lives at this HF boundary
    so the model and ``hf2dcp`` never need to see it.  1-D biases pass
    through unchanged.
    """
    _WEIGHT_KEYS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")

    indexed = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")
    to_stack: Dict[str, Dict[int, torch.Tensor]] = {}
    plain: Dict[str, torch.Tensor] = {}

    for canon, tensor in state_dict.items():
        m = indexed.match(canon)
        if m:
            prefix, idx_str, suffix = m.group(1), m.group(2), m.group(3)
            stacked_canon = "%s.%s" % (prefix, suffix)
            to_stack.setdefault(stacked_canon, {})[int(idx_str)] = tensor
        else:
            plain[canon] = tensor

    result = dict(plain)
    for stacked_canon, by_idx in to_stack.items():
        items = sorted(by_idx.items())
        stacked = torch.stack([t for _, t in items])
        if stacked_canon.endswith(_WEIGHT_KEYS):
            stacked = stacked.transpose(-2, -1).contiguous()
        result[stacked_canon] = stacked
    stdout.info(
        "Stacked %d expert tensors → %d grouped keys"
        % (sum(len(v) for v in to_stack.values()), len(to_stack))
    )
    return result


def dcp2hf(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """Convert DCP checkpoint to HuggingFace format."""
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    max_shard_size = cfg.max_shard_size
    stdout.info("Converting DCP checkpoint from %s" % load_path)

    model_prefix = "app.model."
    state_dict, metadata = dict(), FileSystemReader(load_path).read_metadata()
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if key.startswith(model_prefix):
            state_dict[key] = torch.empty(tensor_meta.size, dtype=tensor_meta.properties.dtype)
    dcp.load(state_dict, checkpoint_id=load_path, no_dist=True)
    stdout.info("Loaded %d model weights from DCP" % len(state_dict))

    canonical = {k.removeprefix(model_prefix): v for k, v in state_dict.items()}

    if _is_gpt_oss_dcp(metadata):
        canonical = _stack_experts(canonical, stdout)

    hf_state_dict = dict()
    for canon, tensor in canonical.items():
        hf_state_dict[canon if canon.startswith("lm_head.") else "model." + canon] = tensor

    shards: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    current_shard: Dict[str, torch.Tensor] = dict()
    current_size, shard_idx = 0, 0

    for key, tensor in hf_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(("model-%05d.safetensors" % shard_idx, current_shard))
            current_shard, current_size, shard_idx = dict(), 0, shard_idx + 1
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(("model-%05d.safetensors" % shard_idx, current_shard))

    weight_map, total_size = dict(), 0
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (_, shard_tensors) in enumerate(shards):
        shard_name = "model-%05d-of-%05d.safetensors" % (i, len(shards))
        stdout.info("Writing shard %d/%d: %s" % (i + 1, len(shards), shard_name))
        save_file(shard_tensors, str(Path(save_path, shard_name)))
        for key in shard_tensors:
            weight_map[key] = shard_name
        total_size += sum(t.numel() * t.element_size() for t in shard_tensors.values())

    with open(Path(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    stdout.info(
        "Saved HF checkpoint to %s (%d weights, %d shards)"
        % (save_path, len(weight_map), len(shards))
    )


def launch(cfg: ConvertCheckpointCfg) -> None:
    """
    Launch checkpoint conversion.
    """
    with ExitStack() as stack:
        ctx = ConvertCheckpointCtx()
        stack.enter_context(logging_context(cfg, ctx))
        ctx.logging.stdout.info("launch(cfg=%s)" % cfg)
        match cfg.operation:
            case "hf2dcp":
                hf2dcp(cfg, ctx.logging.stdout)
            case "dcp2hf":
                dcp2hf(cfg, ctx.logging.stdout)
