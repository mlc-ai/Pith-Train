"""
Fused SwiGLU-style ``silu(gate) * up`` autograd function.

Eager PyTorch implements this as two element-wise kernels in the forward pass
(``silu`` then ``mul``) and several kernels in the backward pass, all of which
are memory-bound. In addition, autograd saves ``sigmoid(gate)``, ``silu(gate)``,
and ``up`` for backward, which wastes activation memory on a recomputable
intermediate.

The kernels here:

- fuse the forward ``silu(gate) * up`` into one Triton kernel (two loads,
  one store), and
- fuse the backward into one Triton kernel that recomputes ``silu`` /
  ``sigmoid`` on the fly (three loads, two stores).

Only ``gate`` and ``up`` are saved for backward - ``silu(gate)`` is no longer
stored as an activation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``out = silu(gate) * up`` element-wise in one pass."""
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    silu = gate_f * tl.sigmoid(gate_f)
    out = silu * up_f

    tl.store(out_ptr + offs, out.to(up.dtype), mask=mask)


@triton.jit
def _silu_mul_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``grad_gate`` and ``grad_up`` element-wise in one pass.

    With ``s = silu(gate)`` and ``sigma = sigmoid(gate)``:
        grad_up   = grad_out * s
        grad_gate = grad_out * up * sigma * (1 + gate - s)
    """
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    grad_out = tl.load(grad_out_ptr + offs, mask=mask)
    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)

    grad_out_f = grad_out.to(tl.float32)
    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)

    sig = tl.sigmoid(gate_f)
    silu = gate_f * sig

    grad_up = grad_out_f * silu
    grad_gate = grad_out_f * up_f * sig * (1.0 + gate_f - silu)

    tl.store(grad_gate_ptr + offs, grad_gate.to(gate.dtype), mask=mask)
    tl.store(grad_up_ptr + offs, grad_up.to(up.dtype), mask=mask)


_BLOCK_SIZE = 1024


class _SiLUMul(torch.autograd.Function):
    """Fused SwiGLU activation that saves only ``gate`` and ``up``."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
        assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
        assert gate.is_contiguous(), "gate must be contiguous"
        assert up.is_contiguous(), "up must be contiguous"
        ctx.save_for_backward(gate, up)
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, _BLOCK_SIZE),)
        _silu_mul_fwd_kernel[grid](gate, up, out, n, BLOCK_SIZE=_BLOCK_SIZE)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        assert grad_output.shape == gate.shape, (
            f"grad_output shape {grad_output.shape} != gate shape {gate.shape}"
        )
        assert grad_output.dtype == gate.dtype, (
            f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}"
        )
        assert grad_output.is_contiguous(), "grad_output must be contiguous"
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        n = gate.numel()
        grid = (triton.cdiv(n, _BLOCK_SIZE),)
        _silu_mul_bwd_kernel[grid](
            grad_output, gate, up, grad_gate, grad_up, n, BLOCK_SIZE=_BLOCK_SIZE
        )
        return grad_gate, grad_up


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` with a single-kernel backward.

    Parameters
    ----------
    gate, up : torch.Tensor
        Same-shape, same-dtype tensors produced by the gate and up projections
        of a SwiGLU-style MLP.

    Returns
    -------
    torch.Tensor
        Element-wise ``silu(gate) * up`` in the same dtype as the inputs.
    """
    return _SiLUMul.apply(gate, up)
