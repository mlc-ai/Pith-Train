"""PyTorch implementation of all-to-all communication operator."""

from typing import List

import torch
import torch.autograd
import torch.distributed

# Cache the compiler-disabled wrapper once at import time.
_all_to_all_single = torch.compiler.disable(torch.distributed.all_to_all_single)


# Pad all-to-all output allocations to multiples of this many rows.
# Reduces CUDA memory fragmentation by collapsing varying allocation sizes
# into the same caching-allocator bucket for reuse across micro-batches.
_ALLTOALL_PAD_ALIGNMENT = 1024


@torch.no_grad()
def direct_all_to_all(
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """
    Async all-to-all without torch.autograd.Function overhead.

    The returned tensor has a ``.comm_work`` attribute
    (``torch.distributed.Work`` handle).  The caller is responsible for
    waiting on the handle and implementing backward manually.

    Args match ``torch.distributed.all_to_all_single``.
    """
    valid_size = sum(output_split_sizes)
    padded_size = (
        (valid_size + _ALLTOALL_PAD_ALIGNMENT - 1)
        // _ALLTOALL_PAD_ALIGNMENT
        * _ALLTOALL_PAD_ALIGNMENT
    )
    buf = input.new_empty(padded_size, *input.shape[1:])
    output = buf[:valid_size]
    comm_work = _all_to_all_single(
        output, input, output_split_sizes, input_split_sizes, group, async_op=True
    )
    output.comm_work = comm_work
    return output
