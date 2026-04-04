#!/bin/bash
# Run 15-step correctness validation training.
#
# Single-node usage:
#   bash .claude/skills/correctness-validation/scripts/launch_validate.sh deepseek-v2-lite
#   bash .claude/skills/correctness-validation/scripts/launch_validate.sh qwen3-30b-a3b
#
# Multi-node usage (SLURM):
#   srun -W 0 .claude/skills/correctness-validation/scripts/launch_validate.sh deepseek-v2-lite

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

if [ $# -ne 1 ]; then
    echo "Usage: launch_validate.sh <model>" >&2
    echo "  Models: deepseek-v2-lite, qwen3-30b-a3b" >&2
    exit 1
fi

MODEL=$1

case $MODEL in
    deepseek-v2-lite)
        SCRIPT=.claude/skills/correctness-validation/scripts/validate_deepseek_v2_lite.py
        ;;
    qwen3-30b-a3b)
        SCRIPT=.claude/skills/correctness-validation/scripts/validate_qwen3_30b_a3b.py
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo "  Models: deepseek-v2-lite, qwen3-30b-a3b" >&2
        exit 1
        ;;
esac

# Setup distributed — auto-detect SLURM or fall back to single-node.
SLURM_NNODES=${SLURM_NNODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}}
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-$(hostname)}

LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID)
LAUNCH_ARGS+=(--nproc-per-node=$(echo "$SLURM_STEP_GPUS" | tr ',' '\n' | wc -l))
LAUNCH_ARGS+=(--rdzv-backend=c10d)
LAUNCH_ARGS+=(--rdzv-endpoint=$(command -v scontrol &>/dev/null && scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1 || echo localhost):15213)

OUTPUT=logging/correctness-validation/validate_${MODEL}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
torchrun ${LAUNCH_ARGS[@]} $SCRIPT
