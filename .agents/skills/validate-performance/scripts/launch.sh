#!/bin/bash
# Launch one arm of a throughput measurement under torchrun.
#
#   Usage:  bash launch.sh <run-file.py>
#   Output: the training log on stdout and stderr
#
# Twin of .agents/skills/validate-correctness/scripts/launch.sh; keep the two in sync.

set -euo pipefail

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

RDZV_HOST=localhost
[ -n "${SLURM_STEP_NODELIST:-}" ] && RDZV_HOST=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$RDZV_HOST:15213)

torchrun ${TORCHRUN_ARGS[@]} "$1"
