"""Layer partitioning for DualPipeV pipeline stages.

In the V-shaped pipeline, stage 0 holds ``embed_tokens`` and stage N-1 holds
``norm`` + ``lm_head``.  These extra components add parameter, optimizer-state,
gradient, and activation memory that the middle stages don't carry.  To
compensate, we give the edge stages fewer decoder layers, shifting the surplus
to their immediate neighbors.
"""

from typing import List

from pithtrain.dualpipe.utils import print_msg


def layer_partition(num_layers: int, num_stages: int, verbose: bool = True) -> List[int]:
    """Distribute *num_layers* decoder layers across *num_stages* pipeline stages.

    The algorithm:

    1. Start with an even ``num_layers // num_stages`` per stage.
    2. Distribute the ``num_layers % num_stages`` remainder layers one at a
       time to the globally smallest stage, breaking ties in favour of
       inner stages so that edges stay light when possible.

    Parameters
    ----------
    num_layers : int
        Total number of decoder layers (must be >= num_stages).
    num_stages : int
        Number of pipeline stages (= 2 x pp_size).

    Returns
    -------
    list[int]
        Number of layers per stage.  ``sum(result) == num_layers``.
    """
    assert num_layers >= num_stages >= 1, (
        f"Need num_layers ({num_layers}) >= num_stages ({num_stages}) >= 1"
    )

    base, remainder = divmod(num_layers, num_stages)
    layers = [base] * num_stages

    # -- Distribute remainder --
    for _ in range(remainder):
        min_val = min(layers)
        best = None
        # Prefer an inner stage at the minimum value
        for i in range(1, num_stages - 1):
            if layers[i] == min_val:
                best = i
                break
        # Fall back to an edge stage
        if best is None:
            for i in range(num_stages):
                if layers[i] == min_val:
                    best = i
                    break
        layers[best] += 1

    assert sum(layers) == num_layers

    # DualPipeV constraint: each pp_rank holds two stages (phase 0 and phase 1)
    # whose layer counts must differ by at most 1 for the overlapped
    # forward-backward schedule to work correctly.
    if num_stages % 2 == 0:
        pp_size = num_stages // 2
        for k in range(pp_size):
            diff = abs(layers[k] - layers[num_stages - 1 - k])
            assert diff <= 1, (
                f"DualPipeV requires abs(num_layers_phase0 - num_layers_phase1) <= 1 "
                f"for each pp_rank, but pp_rank {k} has {layers[k]} vs "
                f"{layers[num_stages - 1 - k]} layers (diff={diff}). "
                f"Partition: {layers}"
            )

    if verbose:
        print_msg(
            f"layer_partition: {num_layers} layers / {num_stages} stages -> {layers}",
            rank0_only=True,
        )
    return layers
