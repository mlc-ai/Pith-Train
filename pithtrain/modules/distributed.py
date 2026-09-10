"""PithTrain distributed module."""

import atexit
import os
import sys
import threading
from dataclasses import dataclass
from datetime import timedelta

import torch

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed


@dataclass(init=False, slots=True)
class DistributedCfg(SlottedDefault):
    """
    Configuration for distributed runtime.

    Parallelism degrees (PP, CP, EP), FSDP replica count, and operation timeout. Both DP
    degrees are derived from the world size.
    """

    pipeline_parallel_size: int = 1
    """
    Degree of pipeline parallelism (PP).

    Partition the model layers across ranks; each rank holds a consecutive slice. Forward and
    backward execution is scheduled by DualPipeV.
    """

    context_parallel_size: int = 1
    """
    Degree of context parallelism (CP).

    Shard the sequence dimension across CP ranks. K/V exchange uses ring attention with a zigzag
    token layout.
    """

    expert_parallel_size: int = 1
    """
    Degree of expert parallelism (EP).

    Distribute the MoE experts across ranks; non-expert layers are unaffected. Token routing uses
    EP dispatch and combine kernels with token deduplication.
    """

    timeout: timedelta = timedelta(minutes=15)
    """
    Timeout for distributed operations.

    Applied to NCCL collectives and the watchdog heartbeat. Scale up for multi-node runs; keep
    small to fail fast.
    """

    hsdp_replica: int = 1
    """
    Number of replicas each FSDP shard group is split into.

    At 1, FSDP shards every parameter across the whole replica group for its class: the dp x cp
    stage for the attention parameters, the dp axis of the expert view for the expert parameters.
    Above 1, both of those groups split into this many replicas, so both must divide by it, and
    FSDP shards within one replica and all-reduces across them. Raise it when one replica already
    holds the model, trading memory for a cheaper gradient reduction.
    """


def setup_torch_runtime() -> None:
    """Apply torch runtime tuning: enable TF32 matmul and raise the dynamo recompile cap."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 64


def setup_default_process_group(cfg: DistributedCfg) -> None:
    """
    Initialize the default process group from torchrun environment variables.

    Read global/local rank info into the distributed context, apply NCCL env tuning, register
    cleanup at exit, and set the current CUDA device from the local rank.
    """
    assert torch.cuda.is_available(), "CUDA is not available."
    assert "TORCHELASTIC_RUN_ID" in os.environ, "Not launched with torchrun."

    distributed.rank = int(os.environ["RANK"])
    distributed.world_size = int(os.environ["WORLD_SIZE"])
    distributed.local_rank = int(os.environ["LOCAL_RANK"])
    distributed.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(int(cfg.timeout.total_seconds()))

    kwargs = dict(backend="nccl", device_id=distributed.local_rank, timeout=cfg.timeout)
    torch.distributed.init_process_group(**kwargs)
    atexit.register(torch.distributed.destroy_process_group)
    torch.cuda.set_device(distributed.local_rank)
    distributed.device = torch.device("cuda", distributed.local_rank)


def setup_failfast_excepthook() -> None:
    """
    Install a fail-fast excepthook that bypasses the NCCL drain on uncaught exceptions.

    Default torch.distributed shutdown can hang indefinitely while draining in-flight NCCL work
    that peers will never satisfy. Hard-exiting bypasses the drain so NCCL wor on other ranks
    fail fast instead of hanging.
    """
    original = sys.excepthook

    def excepthook(exc_type, exc_value, exc_tb, *_):
        try:
            original(exc_type, exc_value, exc_tb)
        except Exception:
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(1)

    sys.excepthook = excepthook
    threading.excepthook = lambda args: excepthook(*args)


def setup_device_mesh(cfg: DistributedCfg) -> None:
    """
    Build the attention and expert views of the rank space and publish per-axis groups.

    MoE parallel folding (https://arxiv.org/abs/2504.14960 section 3.2): attention and the
    experts factor the same block of world_size // pp_size ranks two different ways, dp x cp for
    attention and dp x ep for the experts. PP is the one shared axis and stays outermost, so a
    rank agrees with itself about which pipeline stage it holds. Each view puts its high-traffic
    axis innermost, so the ring K/V exchange and the MoE all-to-all each run over a contiguous
    rank block and stay inside the NVLink domain independently of one another.

    So cp_size and ep_size each need only divide the stage size, not each other, and ep_rank says
    only which experts a rank hosts: which data a rank loads is dp_rank alone.
    """
    pp_size = cfg.pipeline_parallel_size
    cp_size = cfg.context_parallel_size
    ep_size = cfg.expert_parallel_size

    world_size = distributed.world_size
    if world_size % pp_size != 0:
        raise RuntimeError(f"{world_size=} not divisible by {pp_size=}")
    stage_size = world_size // pp_size
    if stage_size % cp_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {cp_size=}")
    if stage_size % ep_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {ep_size=}")
    attn_dp_size = stage_size // cp_size
    expt_dp_size = stage_size // ep_size

    # Both views carry pp, so the pp communicator is built twice, at the cost of one extra
    # ncclCommSplit. Only the pp group on attn_mesh is ever read.
    init = torch.distributed.init_device_mesh
    attn_mesh = init("cuda", (pp_size, attn_dp_size, cp_size), mesh_dim_names=("pp", "dp", "cp"))
    expt_mesh = init("cuda", (pp_size, expt_dp_size, ep_size), mesh_dim_names=("pp", "dp", "ep"))
    distributed.attn_mesh, distributed.expt_mesh = attn_mesh, expt_mesh

    distributed.pp_size, distributed.pp_rank = pp_size, attn_mesh.get_local_rank("pp")
    distributed.pp_group = attn_mesh.get_group("pp")

    distributed.cp_size, distributed.cp_rank = cp_size, attn_mesh.get_local_rank("cp")
    distributed.cp_group = attn_mesh.get_group("cp")

    distributed.ep_size, distributed.ep_rank = ep_size, expt_mesh.get_local_rank("ep")
    distributed.ep_group = expt_mesh.get_group("ep")

    # Neither dp axis gets a process group: no collective runs over them directly, and FSDP
    # reduces there off a DeviceMesh, which the two views already provide. Only the attention dp
    # is published, since that is what decides which data a rank loads.
    distributed.dp_size, distributed.dp_rank = attn_dp_size, attn_mesh.get_local_rank("dp")


def setup_distributed(cfg: object) -> None:
    """Initialize the distributed runtime: process group and device mesh."""
    assert hasattr(cfg, "distributed") and isinstance(cfg.distributed, DistributedCfg)
    setup_torch_runtime()
    setup_default_process_group(cfg.distributed)
    setup_failfast_excepthook()
    setup_device_mesh(cfg.distributed)
