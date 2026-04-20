"""Validates the dedup EP dispatch pipeline (single-process simulation)."""

import pytest
import torch


def simulate_sender(hidden_states, topk_ids, ep_size, experts_per_rank):
    m, k = topk_ids.shape
    num_experts = ep_size * experts_per_rank
    expert_idxs = topk_ids.view(-1)

    cnts = topk_ids.new_zeros((m, num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_idxs.argsort()

    tokens_per_ep_rank = tokens_per_expert.view(ep_size, -1).sum(dim=1)
    input_splits = tokens_per_ep_rank.tolist()

    gpu_ids = topk_ids // experts_per_rank
    cnts_dedup = topk_ids.new_zeros((m, ep_size))
    cnts_dedup.scatter_(1, gpu_ids, 1)

    nz = cnts_dedup.T.nonzero()
    dispatch_token_idxs = nz[:, 1]
    dedup_sorted_tokens = hidden_states[dispatch_token_idxs]

    dedup_tokens_per_gpu = cnts_dedup.sum(dim=0)
    dedup_input_splits = dedup_tokens_per_gpu.tolist()

    token_ids = idxs // k
    gpu_ids_sorted = gpu_ids.view(-1)[idxs]
    nz_keys = nz[:, 0] * m + nz[:, 1]
    query_keys = gpu_ids_sorted * m + token_ids
    global_pos = torch.searchsorted(nz_keys, query_keys)
    gpu_starts = dedup_tokens_per_gpu.cumsum(0) - dedup_tokens_per_gpu
    expand_idx = global_pos - gpu_starts[gpu_ids_sorted]

    ref_tokens = hidden_states[idxs // k]

    def _split(tensor, splits):
        return dict(zip(range(ep_size), tensor.split(splits)))

    return {
        "input_splits": input_splits,
        "dedup_input_splits": dedup_input_splits,
        "tokens_per_expert": tokens_per_expert,
        "dedup_chunks": _split(dedup_sorted_tokens, dedup_input_splits),
        "expand_idx_chunks": _split(expand_idx, input_splits),
        "ref_chunks": _split(ref_tokens, input_splits),
    }


def simulate_receiver(sender_data_list, receiver_gpu, ep_size, experts_per_rank):
    g = receiver_gpu
    dedup_output_splits = []
    output_splits = []

    for sender in sender_data_list:
        tpe_local = sender["tokens_per_expert"][g * experts_per_rank : (g + 1) * experts_per_rank]
        output_splits.append(tpe_local.sum().item())
        dedup_output_splits.append(sender["dedup_input_splits"][g])

    dedup_gathered = torch.cat([s["dedup_chunks"][g] for s in sender_data_list])
    received_expand_idx = torch.cat([s["expand_idx_chunks"][g] for s in sender_data_list])

    dedup_counts = torch.tensor(dedup_output_splits, dtype=received_expand_idx.dtype)
    dedup_starts = dedup_counts.cumsum(0) - dedup_counts
    offset_adj = dedup_starts.repeat_interleave(torch.tensor(output_splits, dtype=torch.long))
    adjusted = received_expand_idx + offset_adj

    expanded = dedup_gathered[adjusted]
    reference = torch.cat([s["ref_chunks"][g] for s in sender_data_list])
    return expanded, reference, adjusted, sum(dedup_output_splits)


CONFIGS = [
    (32, 8, 32, 8),
    (64, 4, 16, 4),
    (128, 8, 64, 8),
    (16, 2, 8, 4),
    (100, 8, 32, 4),
    (256, 8, 128, 16),
    (10, 4, 8, 2),
    (32, 1, 8, 4),  # k=1: no dedup
    (32, 4, 8, 2),  # many experts per rank
    (2, 2, 4, 2),  # tiny
    (1, 2, 8, 4),  # single token
    (2048, 8, 128, 2),  # Qwen3-30B-A3B: many experts, few EP ranks
]


def _reference_dispatch_token_idxs(topk_ids_cpu, ep_size, experts_per_rank):
    """Derive per-GPU dispatch token index sets from the reference nonzero path."""
    m = topk_ids_cpu.shape[0]
    gpu_ids = topk_ids_cpu // experts_per_rank
    cnts_dedup = topk_ids_cpu.new_zeros((m, ep_size))
    cnts_dedup.scatter_(1, gpu_ids, 1)
    nz = cnts_dedup.T.nonzero()
    dispatch_idxs = nz[:, 1]
    dedup_per_gpu = cnts_dedup.sum(dim=0).long()
    gpu_starts = dedup_per_gpu.cumsum(0) - dedup_per_gpu
    return dispatch_idxs, dedup_per_gpu, gpu_starts


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_fused_dedup_dispatch(ms, k, num_experts, ep_size, seed):
    """Compare fused Triton kernel outputs against the PyTorch reference."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    H = 64
    device = "cuda"

    hidden_states = torch.randn(ms, H, device=device)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:k] for _ in range(ms)])

    # Reference (CPU)
    ref = simulate_sender(hidden_states.cpu(), topk_ids.cpu(), ep_size, experts_per_rank)

    # Fused kernel
    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank)

    # -- Deterministic checks (exact match) --
    ref_tokens_per_ep_rank = ref["tokens_per_expert"].view(ep_size, -1).sum(dim=1)
    assert torch.equal(tokens_per_ep_rank.cpu(), ref_tokens_per_ep_rank)
    ref_dedup_tokens_per_gpu = torch.tensor(ref["dedup_input_splits"], dtype=torch.int64)
    assert torch.equal(dedup_tokens_per_gpu.cpu(), ref_dedup_tokens_per_gpu)

    # -- send_meta interleaved layout (embeds tokens_per_expert + dedup counts) --
    ref_send_meta = torch.cat(
        [
            ref["tokens_per_expert"].view(ep_size, experts_per_rank),
            ref_dedup_tokens_per_gpu.unsqueeze(1),
        ],
        dim=1,
    ).view(-1)
    assert torch.equal(send_meta.cpu(), ref_send_meta), "send_meta layout mismatch"

    # -- Set equality per GPU chunk for dispatch_token_idxs --
    gpu_starts = dedup_tokens_per_gpu.cumsum(0) - dedup_tokens_per_gpu
    ref_dispatch, _, ref_gpu_starts = _reference_dispatch_token_idxs(
        topk_ids.cpu(), ep_size, experts_per_rank
    )
    for g in range(ep_size):
        count = ref["dedup_input_splits"][g]
        if count == 0:
            continue
        our_start = gpu_starts[g].item()
        our_set = set(dispatch_token_idxs[our_start : our_start + count].cpu().tolist())
        ref_start = ref_gpu_starts[g].item()
        ref_set = set(ref_dispatch[ref_start : ref_start + count].tolist())
        assert our_set == ref_set, f"GPU {g}: dispatch_token_idxs mismatch"

    # -- Semantic consistency: expand_idx correctness --
    if ms > 0 and k > 0:
        token_ids = idxs // k
        expert_ids_sorted = topk_ids.view(-1)[idxs]
        gpu_ids_sorted = expert_ids_sorted // experts_per_rank
        gpu_starts_dev = gpu_starts.to(device)
        gathered = dispatch_token_idxs[gpu_starts_dev[gpu_ids_sorted] + expand_idx]
        assert torch.equal(gathered, token_ids), "expand_idx semantic invariant violated"


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_dedup_end_to_end(ms, k, num_experts, ep_size, seed):
    """Full sender->receiver pipeline using fused kernel, verifying expanded == reference."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    H = 64
    device = "cuda"

    sender_data_list = []
    for _ in range(ep_size):
        h = torch.randn(ms, H, device=device)
        ids = torch.stack([torch.randperm(num_experts, device=device)[:k] for _ in range(ms)])
        (
            tokens_per_ep_rank,
            dedup_tokens_per_gpu,
            dispatch_token_idxs,
            idxs,
            expand_idx,
            send_meta,
        ) = fused_dedup_prepare_dispatch(ids, num_experts, ep_size, experts_per_rank)

        # Extract tokens_per_expert from send_meta interleaved layout
        meta_2d = send_meta.view(ep_size, experts_per_rank + 1)
        tokens_per_expert = meta_2d[:, :experts_per_rank].reshape(-1)

        total_dedup = dedup_tokens_per_gpu.sum().item()
        dispatch_token_idxs = dispatch_token_idxs[:total_dedup]
        dedup_sorted_tokens = h[dispatch_token_idxs]

        dedup_input_splits = dedup_tokens_per_gpu.cpu().tolist()
        input_splits = tokens_per_ep_rank.cpu().tolist()
        ref_tokens = h[idxs // k]

        def _split(tensor, splits, ep=ep_size):
            return dict(zip(range(ep), tensor.cpu().split(splits)))

        sender_data_list.append(
            {
                "input_splits": input_splits,
                "dedup_input_splits": dedup_input_splits,
                "tokens_per_expert": tokens_per_expert.cpu(),
                "dedup_chunks": _split(dedup_sorted_tokens, dedup_input_splits),
                "expand_idx_chunks": _split(expand_idx, input_splits),
                "ref_chunks": _split(ref_tokens, input_splits),
            }
        )

    for g in range(ep_size):
        expanded, reference, adjusted, dedup_total = simulate_receiver(
            sender_data_list, g, ep_size, experts_per_rank
        )
        assert torch.equal(expanded, reference), f"GPU {g}: expanded != reference"
        if adjusted.numel() > 0:
            assert adjusted.min() >= 0
            assert adjusted.max() < dedup_total


def test_fused_dedup_dispatch_m_zero():
    """Edge case: m=0 should return empty tensors."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    topk_ids = torch.empty((0, 4), dtype=torch.int64, device="cuda")
    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts=16, ep_size=4, experts_per_rank=4)

    assert tokens_per_ep_rank.shape == (4,)
    assert dedup_tokens_per_gpu.shape == (4,)
    assert idxs.numel() == 0
    assert expand_idx.numel() == 0
    assert send_meta.shape == (4 * (4 + 1),)
    assert send_meta.sum() == 0


# -- Unit tests for post-all-to-all fused kernels --


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_adjust_expand_idx(ms, k, num_experts, ep_size, seed):
    """Compare fused adjust_expand_idx against PyTorch reference."""
    from pithtrain.operators.ep_dispatch import adjust_expand_idx

    torch.manual_seed(seed)
    device = "cuda"

    # Generate realistic inputs: simulate received data from ep_size senders
    dedup_tokens_from_each_gpu = torch.randint(1, max(ms, 2), (ep_size,), device=device)
    output_splits_tensor = torch.randint(1, max(ms * k // ep_size, 2), (ep_size,), device=device)
    total = output_splits_tensor.sum().item()
    total_dedup = dedup_tokens_from_each_gpu.sum().item()

    received_expand_idx = torch.randint(0, max(total_dedup, 1), (total,), device=device)

    # Reference (PyTorch)
    dedup_starts = dedup_tokens_from_each_gpu.cumsum(0) - dedup_tokens_from_each_gpu
    offset_adj = dedup_starts.repeat_interleave(output_splits_tensor)
    ref = received_expand_idx + offset_adj

    # Fused kernel
    result = adjust_expand_idx(
        received_expand_idx, dedup_tokens_from_each_gpu, output_splits_tensor
    )

    assert torch.equal(result, ref), "adjust_expand_idx mismatch"


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_build_expert_idxs(ms, k, num_experts, ep_size, seed):
    """Compare fused build_expert_idxs against PyTorch reference."""
    from pithtrain.operators.ep_dispatch import build_expert_idxs

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    device = "cuda"

    # Generate realistic tokens_per_expert_group
    tokens_per_expert_group = torch.randint(0, max(ms, 2), (num_experts,), device=device)

    # Reference (PyTorch)
    ref_output_splits = tokens_per_expert_group.view(ep_size, experts_per_rank).sum(1)
    ref_expert_idxs = (
        torch.arange(num_experts, device=device) % experts_per_rank
    ).repeat_interleave(tokens_per_expert_group)

    # Fused kernel (over-allocated, then sliced like the caller does)
    total = tokens_per_expert_group.sum().item()
    max_total = max(total, ms * k * ep_size)  # ensure >= actual total
    expert_idxs, output_splits_tensor = build_expert_idxs(
        tokens_per_expert_group, ep_size, experts_per_rank, max_total=max_total
    )
    expert_idxs = expert_idxs[:total]

    assert torch.equal(output_splits_tensor, ref_output_splits), "output_splits_tensor mismatch"
    assert torch.equal(expert_idxs, ref_expert_idxs), "expert_idxs mismatch"
