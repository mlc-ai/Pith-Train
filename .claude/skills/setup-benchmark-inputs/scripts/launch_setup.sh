#!/bin/bash
# Setup the minimal corpus + checkpoint artifacts needed to benchmark a model.

set -euo pipefail
export PYTHONUNBUFFERED=1

python3 .claude/skills/setup-benchmark-inputs/scripts/setup.py $@
