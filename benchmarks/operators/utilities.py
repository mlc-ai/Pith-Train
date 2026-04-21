from dataclasses import dataclass


@dataclass
class Metrics:
    """Benchmark results for a single kernel pass (forward or backward)."""

    ref_ms: float
    """Wall time in milliseconds for the ref kernel."""
    our_ms: float
    """Wall time in milliseconds for the our kernel."""
    flops: int
    """Total floating-point operations."""
    io_bytes: int
    """Total global memory I/O in bytes."""
    te_ms: float = float("nan")
    """Wall time in milliseconds for the TransformerEngine kernel."""
