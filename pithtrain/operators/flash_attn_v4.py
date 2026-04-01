"""
Flash Attention 4 (CuTeDSL).

Wraps FA4's internal _flash_attn_fwd/_flash_attn_bwd with torch.library.custom_op
so that torch.compile can trace through them. Supports both symmetric (GQA/MHA)
and asymmetric (MLA) head dimensions under BSHD layout.
"""

from typing import Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

# fmt: off
# mypy: ignore-errors

# ---------------------------------------------------------------------------
# MHA / GQA
# ---------------------------------------------------------------------------

@torch.library.custom_op("pithtrain::flash_attn4_mha_fwd", mutates_args=())
def _mha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    o, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=True)
    return o, lse

@_mha_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool):
    (b, s, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn4_mha_bwd", mutates_args=())
def _mha_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=causal)
    return dq, dk, dv

@_mha_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

def _mha_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, softmax_scale, causal = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal

def _mha_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = _mha_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale, ctx.causal)
    return dq, dk, dv, None, None

_mha_fwd.register_autograd(_mha_backward, setup_context=_mha_setup_context)

def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = False) -> torch.Tensor:
    o, _ = _mha_fwd(q, k, v, softmax_scale, causal)
    return o

# ---------------------------------------------------------------------------
# MLA
#
# Wraps Q/K concat and FA4 into opaque custom_ops to work around an Inductor
# SM100 codegen bug that produces NaN on FA4's asymmetric-dim backward.
# ---------------------------------------------------------------------------

@torch.library.custom_op("pithtrain::flash_attn4_mla_fwd", mutates_args=())
def _mla_fwd(q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor, v: torch.Tensor, softmax_scale: float, qk_nope_head_dim: int, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, q_nope.shape[2], -1)], dim=-1)
    o, lse = _flash_attn_fwd(q, k, v.contiguous(), softmax_scale=softmax_scale, causal=causal, return_lse=True)
    return o, lse

@_mla_fwd.register_fake
def _(q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor, v: torch.Tensor, softmax_scale: float, qk_nope_head_dim: int, causal: bool):
    b, s, h, dv = q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v.shape[-1]
    o = torch.empty((b, s, h, dv), dtype=q_nope.dtype, device=q_nope.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q_nope.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn4_mla_bwd", mutates_args=())
def _mla_bwd(q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, grad_o: torch.Tensor, softmax_scale: float, qk_nope_head_dim: int, causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = q_nope.shape[2]
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, h, -1)], dim=-1)
    dq, dk, dv = _flash_attn_bwd(q, k, v.contiguous(), o, grad_o, lse, softmax_scale=softmax_scale, causal=causal)
    dq_nope, dq_pe = dq[:, :, :, :qk_nope_head_dim], dq[:, :, :, qk_nope_head_dim:]
    dq_nope, dq_pe = dq_nope.contiguous(), dq_pe.contiguous()
    dk_nope, dk_pe = dk[:, :, :, :qk_nope_head_dim], dk[:, :, :, qk_nope_head_dim:]
    dk_nope, dk_pe = dk_nope.contiguous(), dk_pe.sum(dim=2, keepdim=True)
    return dq_nope, dq_pe, dk_nope, dk_pe, dv

@_mla_bwd.register_fake
def _(q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, grad_o: torch.Tensor, softmax_scale: float, qk_nope_head_dim: int, causal: bool):
    return torch.empty_like(q_nope), torch.empty_like(q_pe), torch.empty_like(k_nope), torch.empty_like(k_pe), torch.empty_like(v)

def _mla_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q_nope, q_pe, k_nope, k_pe, v, softmax_scale, qk_nope_head_dim, causal = inputs
    o, lse = output
    ctx.save_for_backward(q_nope, q_pe, k_nope, k_pe, v, o, lse)
    ctx.softmax_scale = softmax_scale
    ctx.qk_nope_head_dim = qk_nope_head_dim
    ctx.causal = causal

def _mla_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:
    q_nope, q_pe, k_nope, k_pe, v, o, lse = ctx.saved_tensors
    dq_nope, dq_pe, dk_nope, dk_pe, dv = _mla_bwd(q_nope, q_pe, k_nope, k_pe, v, o, lse, grad_o, ctx.softmax_scale, ctx.qk_nope_head_dim, ctx.causal)
    return dq_nope, dq_pe, dk_nope, dk_pe, dv, None, None, None

_mla_fwd.register_autograd(_mla_backward, setup_context=_mla_setup_context)

def mla_flash_attn_func(q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor, v: torch.Tensor, softmax_scale: float, qk_nope_head_dim: int, causal: bool = False) -> torch.Tensor:
    o, _ = _mla_fwd(q_nope, q_pe, k_nope, k_pe, v, softmax_scale, qk_nope_head_dim, causal)
    return o
