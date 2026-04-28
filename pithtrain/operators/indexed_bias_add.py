"""
x + bias[group_ids] with a fast segmented-reduction backward for the
common MoE pattern where tokens have already been sorted by destination
expert (so group_ids is non-decreasing and each group is a contiguous
range of rows [grouped_mm_offs[g-1], grouped_mm_offs[g])).

Autograd's default bias[group_ids] backward lowers to index_put_ with
bf16 atomic adds on bias rows — for GPT-OSS expert biases that's the
single dominant kernel by a wide margin (~48% of GPU time, vs <2% for
the actual expert GEMMs).  We replace it with a Triton-fused per-group
reduction (one block per (group, h-tile)) that is fully parallel and
atomic-free.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_sum_kernel(
    x_ptr,
    offs_padded_ptr,
    out_ptr,
    H: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    One block reduces rows [start, end) of group g for a HEAD_BLOCK-wide
    slab of the hidden dim.  Accumulation is in fp32 for numerical
    stability; the cast back to the source dtype happens at the call site.
    """
    g = tl.program_id(0)
    h_blk = tl.program_id(1)

    start = tl.load(offs_padded_ptr + g)
    end = tl.load(offs_padded_ptr + g + 1)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H

    acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for m_start in tl.range(start, end, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < end
        ptrs = x_ptr + m_offs[:, None] * H + h_offs[None, :]
        x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    tl.store(out_ptr + g * H + h_offs, acc, mask=h_mask)


def _segment_sum_along_rows(
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Per-group sum over rows.  Tokens are assumed already sorted by group.

    Parameters:
        x: (M, H) contiguous tensor.
        grouped_mm_offs: (E,) cumulative end-offsets per group (group g
            spans rows [offs[g-1], offs[g]) with offs[-1] := 0).
        out_dtype: dtype of the returned bias gradient (matches the bias
            parameter's dtype).

    Returns:
        (E, H) tensor where row g holds the sum of x rows in group g.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    E = grouped_mm_offs.shape[0]

    # Prepend zero so each block can read both [start, end) without a branch.
    offs_padded = torch.empty(E + 1, dtype=torch.int32, device=x.device)
    offs_padded[0] = 0
    # .clamp keeps padding-rows out of the reduction if grouped_mm_offs
    # exceeds M; copy into int32 once.
    offs_padded[1:].copy_(grouped_mm_offs.to(torch.int32).clamp(max=M))

    out = torch.empty((E, H), dtype=torch.float32, device=x.device)

    HEAD_BLOCK = 128
    BLOCK_M = 128
    grid = (E, triton.cdiv(H, HEAD_BLOCK))
    _segment_sum_kernel[grid](
        x,
        offs_padded,
        out,
        H=H,
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_M=BLOCK_M,
    )
    return out.to(out_dtype) if out_dtype is not torch.float32 else out


class _IndexedBiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, group_ids, grouped_mm_offs):
        ctx.bias_dtype = bias.dtype
        ctx.save_for_backward(grouped_mm_offs)
        return x + bias[group_ids]

    @staticmethod
    def backward(ctx, grad_out):
        (offs,) = ctx.saved_tensors
        grad_out_contig = grad_out.contiguous()
        bias_grad = _segment_sum_along_rows(grad_out_contig, offs, ctx.bias_dtype)
        # x is identity-add → grad_x = grad_out; group_ids/offs are integer.
        return grad_out, bias_grad, None, None


def indexed_bias_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    group_ids: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x + bias[group_ids] with fast segmented-sum backward.

    group_ids must be non-decreasing and grouped_mm_offs the matching
    per-group cumulative end offsets (i.e. offs[-1] == M).
    """
    return _IndexedBiasAdd.apply(x, bias, group_ids, grouped_mm_offs)
