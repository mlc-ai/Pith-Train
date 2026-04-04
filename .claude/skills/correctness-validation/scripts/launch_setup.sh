#!/bin/bash
# Download data, tokenize corpus, and convert checkpoint for correctness validation.
#
# Single-node usage:
#   bash .claude/skills/correctness-validation/scripts/launch_setup.sh deepseek-v2-lite
#   bash .claude/skills/correctness-validation/scripts/launch_setup.sh qwen3-30b-a3b
#
# Multi-node usage (SLURM) — run on every node since workspace is node-local:
#   srun -W 0 .claude/skills/correctness-validation/scripts/launch_setup.sh deepseek-v2-lite

set -euo pipefail
export PYTHONUNBUFFERED=1

if [ $# -ne 1 ]; then
    echo "Usage: launch_setup.sh <model>" >&2
    echo "  Models: deepseek-v2-lite, qwen3-30b-a3b" >&2
    exit 1
fi

MODEL=$1

case $MODEL in
    deepseek-v2-lite)
        SCRIPT=.claude/skills/correctness-validation/scripts/setup_deepseek_v2_lite.py
        ;;
    qwen3-30b-a3b)
        SCRIPT=.claude/skills/correctness-validation/scripts/setup_qwen3_30b_a3b.py
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo "  Models: deepseek-v2-lite, qwen3-30b-a3b" >&2
        exit 1
        ;;
esac

OUTPUT=logging/correctness-validation/setup_${MODEL}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
python3 $SCRIPT
