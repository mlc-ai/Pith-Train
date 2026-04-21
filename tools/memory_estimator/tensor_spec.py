"""
Symbolic tensor descriptors and memory tracking containers.

No real tensors are created - all sizes are computed analytically from shapes and dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# dtype -> element size in bytes
DTYPE_SIZES: dict[str, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
}


@dataclass(slots=True)
class TensorSpec:
    """A symbolic tensor descriptor - shape + dtype, no allocation."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # one of DTYPE_SIZES keys

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bytes(self) -> int:
        return self.numel * DTYPE_SIZES[self.dtype]

    def __repr__(self) -> str:
        shape_str = ",".join(str(d) for d in self.shape)
        return f"{self.name} [{shape_str}] {self.dtype} = {format_bytes(self.bytes)}"


@dataclass(slots=True)
class MemoryBucket:
    """A named group of TensorSpecs representing one logical memory component."""

    name: str
    specs: list[TensorSpec] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes for s in self.specs)

    def add(self, name: str, shape: tuple[int, ...], dtype: str) -> TensorSpec:
        spec = TensorSpec(name, shape, dtype)
        self.specs.append(spec)
        return spec

    def __repr__(self) -> str:
        return f"MemoryBucket({self.name!r}, {len(self.specs)} tensors, {format_bytes(self.total_bytes)})"


@dataclass(slots=True)
class MemorySnapshot:
    """Full memory state at one point in the DualPipeV schedule."""

    event_name: str

    # Static (constant during step)
    params_unsharded: list[MemoryBucket] = field(default_factory=list)  # [module0, module1]
    fsdp_sharded_params: MemoryBucket = field(default_factory=lambda: MemoryBucket("fsdp_sharded"))
    optimizer_states: MemoryBucket = field(default_factory=lambda: MemoryBucket("optimizer"))

    # Dynamic
    activations: dict[tuple[int, int], MemoryBucket] = field(default_factory=dict)
    autograd_overhead: dict[tuple[int, int], MemoryBucket] = field(default_factory=dict)
    gradients: MemoryBucket = field(default_factory=lambda: MemoryBucket("gradients"))
    comm_buffers: MemoryBucket = field(default_factory=lambda: MemoryBucket("comm_buffers"))
    weight_grad_store: MemoryBucket = field(default_factory=lambda: MemoryBucket("wgrad_store"))
    pp_transfer_buffers: MemoryBucket = field(default_factory=lambda: MemoryBucket("pp_transfer"))

    # Overhead
    non_pytorch_bytes: int = 0  # CUDA context + NCCL communicators + torch.compile codegen
    fragmentation_factor: float = 0.10

    @property
    def static_bytes(self) -> int:
        total = sum(b.total_bytes for b in self.params_unsharded)
        total += self.fsdp_sharded_params.total_bytes
        total += self.optimizer_states.total_bytes
        return total

    @property
    def activation_bytes(self) -> int:
        return sum(b.total_bytes for b in self.activations.values())

    @property
    def autograd_bytes(self) -> int:
        return sum(b.total_bytes for b in self.autograd_overhead.values())

    @property
    def dynamic_bytes(self) -> int:
        total = self.activation_bytes
        total += self.autograd_bytes
        total += self.gradients.total_bytes
        total += self.comm_buffers.total_bytes
        total += self.weight_grad_store.total_bytes
        total += self.pp_transfer_buffers.total_bytes
        return total

    @property
    def subtotal_bytes(self) -> int:
        return self.static_bytes + self.dynamic_bytes + self.non_pytorch_bytes

    @property
    def fragmentation_bytes(self) -> int:
        return int(self.subtotal_bytes * self.fragmentation_factor)

    @property
    def total_bytes(self) -> int:
        return self.subtotal_bytes + self.fragmentation_bytes

    @property
    def live_chunk_counts(self) -> tuple[int, int]:
        """Return (phase0_live, phase1_live) chunk counts."""
        p0 = sum(1 for (p, _) in self.activations if p == 0)
        p1 = sum(1 for (p, _) in self.activations if p == 1)
        return p0, p1


def format_bytes(b: int) -> str:
    """Format bytes as a human-readable string."""
    if b == 0:
        return "0 B"
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024**3:.3f} GB"


def format_gb(b: int) -> str:
    """Format bytes as GB with 2 decimal places."""
    return f"{b / 1024**3:.2f}"
