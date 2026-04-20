"""
DualPipeV 8-step schedule simulator.

Replays the exact scheduling algorithm from DualPipeV.step() (dualpipev.py:482-547),
tracking live activations, gradients, communication buffers, and deferred weight
gradient storage at each event boundary. Produces a timeline of MemorySnapshots
with the peak high-water mark identified.
"""

from __future__ import annotations

from dataclasses import dataclass

from .activation_profile import ActivationEstimator, TokenCounts
from .model_profile import ModelConfig, ModelMemoryProfile, ParallelismConfig
from .tensor_spec import MemoryBucket, MemorySnapshot


@dataclass(slots=True)
class ScheduleEvent:
    """One event in the simulated schedule timeline."""

    name: str
    snapshot: MemorySnapshot
    delta_bytes: int  # change from previous event


@dataclass(slots=True)
class ScheduleResult:
    """Output of the schedule simulation."""

    timeline: list[ScheduleEvent]
    peak_event_idx: int
    peak_bytes: int
    static_bytes: int


class ScheduleSimulator:
    """Simulate the DualPipeV 8-step schedule and track memory at each point."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        parallel_cfg: ParallelismConfig,
        profile: ModelMemoryProfile,
        activation_est: ActivationEstimator,
        token_counts: TokenCounts,
        pp_rank: int,
        fragmentation_factor: float = 0.10,
        non_pytorch_bytes: int = 0,
    ):
        self.model_cfg = model_cfg
        self.parallel_cfg = parallel_cfg
        self.profile = profile
        self.act_est = activation_est
        self.tc = token_counts
        self.pp_rank = pp_rank
        self.pp_size = parallel_cfg.pp_size
        self.frag_factor = fragmentation_factor

        self.is_first_pp_rank = pp_rank == 0
        self.is_last_pp_rank = pp_rank == self.pp_size - 1
        self.non_pytorch_bytes = non_pytorch_bytes

        # Precompute per-module activation costs
        self._chunk_records: dict[int, MemoryBucket] = {}
        self._chunk_autograd: dict[int, MemoryBucket] = {}
        for mod_idx in range(2):
            records, autograd = activation_est.compute_chunk_activations(mod_idx, 0, 0)
            self._chunk_records[mod_idx] = records
            self._chunk_autograd[mod_idx] = autograd

        # Precompute static memory buckets
        self._params = [profile.compute_module_params(i) for i in range(2)]
        self._fsdp_sharded = profile.compute_fsdp_sharded_params()
        self._optimizer = profile.compute_optimizer_states()
        self._grad_buckets = [profile.compute_gradient_bucket(i) for i in range(2)]

        # Precompute per-chunk costs
        self._wgrad_bytes = [
            activation_est.compute_wgrad_store_bytes_per_chunk(i) for i in range(2)
        ]
        self._comm_bytes = activation_est.compute_comm_buffer_bytes()
        self._pp_transfer_bytes = activation_est.compute_pp_transfer_bytes()

        # Num chunks
        dp_size = parallel_cfg.dp_size
        ep_size = parallel_cfg.ep_size
        micro_bs = parallel_cfg.micro_batch_size
        global_bs = parallel_cfg.global_batch_size
        self.num_chunks = global_bs // (micro_bs * dp_size * ep_size)

    def simulate(self) -> ScheduleResult:
        pp_rank = self.pp_rank
        pp_size = self.pp_size
        num_chunks = self.num_chunks

        # State tracking
        # Which (phase, chunk_id) have live activations
        live_acts: dict[tuple[int, int], int] = {}  # -> module_idx
        live_autograd: dict[tuple[int, int], int] = {}  # -> module_idx
        # Gradients: once backward completes a chunk on a module, all its layers have .grad
        grads_allocated: set[int] = set()  # module indices that have gradients
        # WeightGradStore: accumulated bytes from deferred wgrads
        wgrad_store_bytes = 0
        # PP transfer buffers: detached tensors at last PP rank
        pp_transfers: set[int] = set()  # chunk_ids with live transfers

        # Forward chunk counters
        fwd_chunk_id = [0, 0]
        bwd_chunk_id = [0, 0]

        timeline: list[ScheduleEvent] = []
        prev_total = 0

        def _module_for_phase(phase: int) -> int:
            """Phase 0 -> module 0, phase 1 -> module 1."""
            return phase

        def _snapshot(event_name: str) -> None:
            nonlocal prev_total
            snap = MemorySnapshot(event_name=event_name)
            snap.fragmentation_factor = self.frag_factor
            snap.non_pytorch_bytes = self.non_pytorch_bytes

            # Static
            snap.params_unsharded = list(self._params)
            snap.fsdp_sharded_params = self._fsdp_sharded
            snap.optimizer_states = self._optimizer

            # Activations
            for key, mod_idx in live_acts.items():
                phase, chunk_id = key
                records, _ = self.act_est.compute_chunk_activations(mod_idx, phase, chunk_id)
                snap.activations[key] = records
            for key, mod_idx in live_autograd.items():
                phase, chunk_id = key
                _, autograd = self.act_est.compute_chunk_activations(mod_idx, phase, chunk_id)
                snap.autograd_overhead[key] = autograd

            # Gradients
            grad_bucket = MemoryBucket("gradients")
            for mod_idx in sorted(grads_allocated):
                for spec in self._grad_buckets[mod_idx].specs:
                    grad_bucket.specs.append(spec)
            snap.gradients = grad_bucket

            # Comm buffers
            comm_bucket = MemoryBucket("comm_buffers")
            comm_bucket.add("p2p_recv_buffers", (self._comm_bytes,), "int8")
            snap.comm_buffers = comm_bucket

            # WeightGradStore
            wgrad_bucket = MemoryBucket("wgrad_store")
            if wgrad_store_bytes > 0:
                wgrad_bucket.add("deferred_wgrad_tensors", (wgrad_store_bytes,), "int8")
            snap.weight_grad_store = wgrad_bucket

            # PP transfer buffers
            pp_bucket = MemoryBucket("pp_transfer")
            for cid in sorted(pp_transfers):
                pp_bucket.add(
                    f"pp_transfer_chunk_{cid}",
                    (self._pp_transfer_bytes,),
                    "int8",
                )
            snap.pp_transfer_buffers = pp_bucket

            total = snap.total_bytes
            delta = total - prev_total
            prev_total = total
            timeline.append(ScheduleEvent(name=event_name, snapshot=snap, delta_bytes=delta))

        def forward_chunk(phase: int, event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            mod_idx = _module_for_phase(phase)
            chunk_id = fwd_chunk_id[phase]
            fwd_chunk_id[phase] += 1
            key = (phase, chunk_id)
            live_acts[key] = mod_idx
            live_autograd[key] = mod_idx

            # At last PP rank, phase 0 forward creates a transfer to phase 1
            if self.is_last_pp_rank and phase == 0:
                pp_transfers.add(chunk_id)

            _snapshot(f"{event_prefix} fwd(p{phase}) c{chunk_id}")

        def backward_chunk(phase: int, enable_zb: bool, event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            mod_idx = _module_for_phase(phase)
            chunk_id = bwd_chunk_id[phase]
            bwd_chunk_id[phase] += 1
            key = (phase, chunk_id)

            # Free activations
            live_acts.pop(key, None)
            live_autograd.pop(key, None)

            # Allocate gradients (first backward on this module allocates all grads)
            grads_allocated.add(mod_idx)

            # At last PP rank, phase 1 backward produces grads that go to phase 0
            # but the output_grad_chunks[0] is just a reference, not new memory

            if enable_zb:
                wgrad_store_bytes += self._wgrad_bytes[mod_idx]

            # Consume PP transfer if this is phase 1 at last rank
            if self.is_last_pp_rank and phase == 1:
                pp_transfers.discard(chunk_id)

            _snapshot(f"{event_prefix} bwd(p{phase},zb={enable_zb}) c{chunk_id}")

        def forward_backward_chunk(phase0: int, phase1: int, event_prefix: str) -> None:
            """
            Overlapped forward (phase0) + backward (phase1).

            Memory peaks when both activation sets coexist.
            We model this by first adding the forward chunk, then taking snapshot
            (which includes both old backward activations and new forward activations),
            then removing the backward chunk.
            """
            nonlocal wgrad_store_bytes
            mod_idx0 = _module_for_phase(phase0)
            mod_idx1 = _module_for_phase(phase1)

            fwd_cid = fwd_chunk_id[phase0]
            fwd_chunk_id[phase0] += 1
            bwd_cid = bwd_chunk_id[phase1]
            bwd_chunk_id[phase1] += 1

            fwd_key = (phase0, fwd_cid)
            bwd_key = (phase1, bwd_cid)

            # Add forward activations (both sets coexist at peak)
            live_acts[fwd_key] = mod_idx0
            live_autograd[fwd_key] = mod_idx0

            if self.is_last_pp_rank and phase0 == 0:
                pp_transfers.add(fwd_cid)

            # Gradients from backward
            grads_allocated.add(mod_idx1)

            # PP transfer consumed
            if self.is_last_pp_rank and phase1 == 1:
                pp_transfers.discard(bwd_cid)

            # Snapshot with both sets alive (peak of overlap)
            _snapshot(f"{event_prefix} fwd_bwd(p{phase0},p{phase1}) fc{fwd_cid}/bc{bwd_cid}")

            # Now free backward activations (post-overlap)
            live_acts.pop(bwd_key, None)
            live_autograd.pop(bwd_key, None)

        def weight_chunk(event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            # Pops one batch of deferred wgrads - free the oldest module's worth
            # In practice, weight_chunk pops from a FIFO queue
            if wgrad_store_bytes > 0:
                # Each pop frees one chunk's worth of wgrad from whichever module
                # We use a simple approximation: free the average of both modules
                wgrad_store_bytes = max(0, wgrad_store_bytes - max(self._wgrad_bytes))
            _snapshot(f"{event_prefix} weight_chunk")

        # === Static baseline ===
        _snapshot("static_baseline")

        # === Step 1: nF0 = (pp_size - pp_rank - 1) * 2 ===
        step_1 = (pp_size - pp_rank - 1) * 2
        for i in range(step_1):
            forward_chunk(0, f"S1.i{i}")

        # === Step 2: nF0F1 = pp_rank + 1 ===
        step_2 = pp_rank + 1
        for i in range(step_2):
            forward_chunk(0, f"S2.i{i}")
            forward_chunk(1, f"S2.i{i}")

        # === Step 3: nB1W1F1 = pp_size - pp_rank - 1 ===
        step_3 = pp_size - pp_rank - 1
        for i in range(step_3):
            backward_chunk(1, enable_zb=True, event_prefix=f"S3.i{i}")
            weight_chunk(f"S3.i{i}")
            forward_chunk(1, f"S3.i{i}")

        # === Step 4: nF0B1F1B0 = num_chunks - pp_size * 2 + pp_rank + 1 ===
        step_4 = num_chunks - pp_size * 2 + pp_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_last_pp_rank:
                    # Special case: no overlap for first iteration on last rank
                    forward_chunk(0, "S4.i0.last_rank")
                    backward_chunk(1, enable_zb=False, event_prefix="S4.i0.last_rank")
                else:
                    forward_backward_chunk(0, 1, f"S4.i{i}")
            else:
                forward_backward_chunk(0, 1, f"S4.i{i}")
            forward_backward_chunk(1, 0, f"S4.i{i}")

        # === Step 5: nB1F1B0 = pp_size - pp_rank - 1 ===
        step_5 = pp_size - pp_rank - 1
        for i in range(step_5):
            backward_chunk(1, enable_zb=False, event_prefix=f"S5.i{i}")
            forward_backward_chunk(1, 0, f"S5.i{i}")

        # === Step 6: nB1B0 = pp_rank + 1 (second half uses zero bubble) ===
        step_6 = pp_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and pp_rank % 2 == 1:
                enable_zb = True
            backward_chunk(1, enable_zb=enable_zb, event_prefix=f"S6.i{i}")
            if i == step_6 // 2 and pp_rank % 2 == 0:
                enable_zb = True
            backward_chunk(0, enable_zb=enable_zb, event_prefix=f"S6.i{i}")

        # === Step 7: nWB0 = pp_size - pp_rank - 1 ===
        step_7 = pp_size - pp_rank - 1
        for i in range(step_7):
            weight_chunk(f"S7.i{i}")
            backward_chunk(0, enable_zb=True, event_prefix=f"S7.i{i}")

        # === Step 8: nW = pp_rank + 1 ===
        step_8 = pp_rank + 1
        for i in range(step_8):
            weight_chunk(f"S8.i{i}")

        # Find peak
        peak_idx = 0
        peak_bytes = 0
        for idx, event in enumerate(timeline):
            total = event.snapshot.total_bytes
            if total > peak_bytes:
                peak_bytes = total
                peak_idx = idx

        static_bytes = timeline[0].snapshot.static_bytes if timeline else 0

        return ScheduleResult(
            timeline=timeline,
            peak_event_idx=peak_idx,
            peak_bytes=peak_bytes,
            static_bytes=static_bytes,
        )
