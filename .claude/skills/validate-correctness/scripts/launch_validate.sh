#!/bin/bash
# Run a short PithTrain training run for correctness validation.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

SCRIPT=.claude/skills/validate-correctness/scripts/validate.py

SLURM_NNODES=${SLURM_NNODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-$(hostname)}
RDZV_HOST=$(command -v scontrol &>/dev/null && scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1 || echo localhost)

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=$SLURM_NNODES)
TORCHRUN_ARGS+=(--node-rank=$SLURM_NODEID)
TORCHRUN_ARGS+=(--nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d)
TORCHRUN_ARGS+=(--rdzv-endpoint=$RDZV_HOST:15213)

torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $@
