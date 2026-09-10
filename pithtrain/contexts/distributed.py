"""
Distributed runtime state.

Context-parallelism and expert-parallelism are decoupled by MoE parallel folding.
"""

import torch

rank: int
"""
Global rank of this process, in [0, world_size).
"""

world_size: int
"""
Total number of processes across all nodes.
"""

local_rank: int
"""
Rank within this node, and the CUDA device index for this process.
"""

local_world_size: int
"""
Number of processes on this node, i.e. GPUs per node.
"""

device: torch.device
"""
The CUDA device for this process. Prefer it over plain cuda when allocating.
"""

attn_mesh: torch.distributed.DeviceMesh
"""
Attention view of the rank space: (pp, dp, cp), CP innermost so each CP group is contiguous.

The flattened dp x cp submesh is the FSDP mesh for the attention parameters, meaning everything
outside the routed experts, and the group the load-balance statistics reduce over.
"""

expt_mesh: torch.distributed.DeviceMesh
"""
Expert view of the same rank space: (pp, dp, ep), EP innermost so each EP group is contiguous.

The dp axis of this view groups the ranks holding the same experts, and is the FSDP mesh for
the routed expert weights.
"""

pp_group: torch.distributed.ProcessGroup
"""
Pipeline-parallel group. DualPipeV sends activations and gradients over it point to point.
"""

cp_group: torch.distributed.ProcessGroup
"""
Context-parallel group. Ring attention exchanges K/V over it, and the logged loss reduces over it.
"""

ep_group: torch.distributed.ProcessGroup
"""
Expert-parallel group. The MoE dispatch and combine all-to-alls run over it.
"""

pp_rank: int
"""
Index within the pipeline. Under DualPipeV, rank r holds chunks r and 2 * pp_size - 1 - r.
"""

pp_size: int
"""
Pipeline-parallel degree, set by DistributedCfg.pipeline_parallel_size.
"""

dp_rank: int
"""
Index along the dp axis of attn_mesh, and the only thing deciding which data this rank loads.
"""

dp_size: int
"""
Data-parallel degree of the attention view, world_size // (pp_size * cp_size).
"""

cp_rank: int
"""
Index within the CP group. The sequence is cut into 2 * cp_size blocks and this rank holds
blocks cp_rank and 2 * cp_size - cp_rank - 1, the zigzag layout that balances causal attention.
"""

cp_size: int
"""
Context-parallel degree, set by DistributedCfg.context_parallel_size.
"""

ep_rank: int
"""
Index within the EP group, naming the contiguous block of experts this rank hosts.
"""

ep_size: int
"""
Expert-parallel degree, set by DistributedCfg.expert_parallel_size.
"""
