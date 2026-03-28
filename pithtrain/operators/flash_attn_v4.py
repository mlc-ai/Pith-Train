"""
Flash Attention 4 (CuTeDSL) custom_op wrapper for GQA.

Wraps FA4's internal _flash_attn_fwd/_flash_attn_bwd with torch.library.custom_op
so that torch.compile can trace through them.

Layout: BSHD (batch, sequence, heads, head_dim).
"""

from typing import Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

# fmt: off
# mypy: ignore-errors

@torch.library.custom_op("pithtrain::flash_attn_fa4_fwd", mutates_args=())
def _flash_attn_fa4_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    o, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=True)
    return o, lse

@_flash_attn_fa4_fwd.register_fake
def _(q, k, v, softmax_scale, causal):
    (b, s, h, d) = q.shape
    o = torch.empty((b, s, h, d), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn_fa4_bwd", mutates_args=())
def _flash_attn_fa4_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=causal)
    return dq, dk, dv

@_flash_attn_fa4_bwd.register_fake
def _(q, k, v, o, lse, do, softmax_scale, causal):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv

def _setup_context(ctx, inputs, output):
    q, k, v, softmax_scale, causal = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal

def _backward(ctx, grad_o, grad_lse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = _flash_attn_fa4_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale, ctx.causal)
    return dq, dk, dv, None, None

_flash_attn_fa4_fwd.register_autograd(_backward, setup_context=_setup_context)

def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = False) -> torch.Tensor:
    """Drop-in replacement for flash_attn.cute.flash_attn_func, compatible with torch.compile."""
    o, _ = _flash_attn_fa4_fwd(q, k, v, softmax_scale, causal)
    return o
