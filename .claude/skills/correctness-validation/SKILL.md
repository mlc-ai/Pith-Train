---
name: correctness-validation
description: Validates that code changes do not break training correctness by comparing loss curves between a base branch and the current feature branch. Use when user asks to "validate correctness", "check if changes break training", "compare loss curves", "run a regression test", or "verify my changes are correct". Also use when a feature branch modifies model code, operators, pipeline logic, or distributed training modules.
---

# Correctness Validation

Validates training correctness by running a short 15-step training run on both a base branch and the current feature branch, then comparing three metrics step-by-step: cross-entropy loss, load-balance loss, and gradient norm.

## Overview

The validation has two phases:

1. **Shared setup** (run once, reused across branches): download a minimal DCLM corpus shard, tokenize it, download and convert the HuggingFace checkpoint to DCP format.
2. **Branch comparison**: run 15 training steps on the base branch (via git worktree) and the feature branch, then compare the stdout logs.

Shared setup artifacts live in `workspace/` and are deterministic given the same seed and released checkpoint, so they are safe to share between branches.

## Prerequisites

- **Python environment**: Use the `.venv` in the original repo root (not the worktree). Activate it before running any scripts: `source $REPO_ROOT/.venv/bin/activate`. If `.venv` does not exist, create it following the README instructions (`uv venv && uv sync`).
- **Hardware**: Minimum **4x B200 GPUs** (PP=2, EP=2 with DeepSeek-V2-Lite).

Note: both `.venv` and `workspace/` live in the original repo root. The worktree gets both via symlink (see Step 4).

## Supported Models

Each model has a validation script and a setup script under `scripts/`:

| Model | Setup Script | Validation Script | GPUs |
|---|---|---|---|
| DeepSeek-V2-Lite | `setup_deepseek_v2_lite.py` | `validate_deepseek_v2_lite.py` | 4 (PP=2, EP=2) |
| Qwen3-30B-A3B | `setup_qwen3_30b_a3b.py` | `validate_qwen3_30b_a3b.py` | 16 (PP=2, EP=8) |

## Step-by-Step Workflow

### Step 1: Determine Impact and Select Models

Analyze the code change to decide which models need validation. The goal is to run validation on **every model whose behavior could be affected**.

**How to analyze impact:**

1. Get the list of changed files:
   ```bash
   git diff --name-only <base_branch>
   ```

2. **If changes are under a model-specific directory** (e.g., `pithtrain/models/deepseek_v2_lite/` or `pithtrain/models/qwen3_moe/`), only that model is affected.

3. **If changes are in shared code** (e.g., `pithtrain/operators/`, `pithtrain/layers/`, `pithtrain/dualpipe/`, `pithtrain/modules/`, `pithtrain/tasks/`), read the changed code and determine whether it touches a feature that is model-specific or universal:
   - Read each model's `config.json` at `examples/pretrain_language_model/<model>/config.json` to understand what features that model uses (attention type, shared experts, expert count, RoPE variant, etc.)
   - Read the changed code to understand what architectural features it touches
   - A model is affected if it uses any feature touched by the change

4. **If unsure whether a model is affected, include it.** Over-validating is better than missing a regression.

### Step 2: Detect Environment

Check if running under SLURM by testing for `SLURM_JOB_ID`:

```bash
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM detected (job $SLURM_JOB_ID) — will use srun for multi-node launch"
else
    echo "No SLURM — single-node launch"
fi
```

This determines whether to prefix commands with `srun -W 0`. The workspace directory is **node-local storage**, so setup (data download, tokenization, checkpoint conversion) must run on **every node**.

### Step 3: Shared Setup

Run the setup launch script for each affected model. The setup scripts are idempotent — they skip steps whose output already exists.

```bash
# Single-node (replace <model> with deepseek-v2-lite or qwen3-30b-a3b)
bash .claude/skills/correctness-validation/scripts/launch_setup.sh <model>

# Multi-node (SLURM) — must run on every node since workspace is node-local
srun -W 0 .claude/skills/correctness-validation/scripts/launch_setup.sh <model>
```

This downloads a single minimal DCLM shard (`global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst`), tokenizes it with the model's tokenizer, downloads the HuggingFace checkpoint, and converts it to DCP format.

### Step 4: Create Git Worktree for Base Branch

Create a worktree for the base branch. Symlink `workspace/` and `.venv` from the repo root so both branches share the same data and environment.

```bash
BASE_BRANCH=main  # or the branch this feature was based on
WORKTREE=$(mktemp -d)
REPO_ROOT=$(git rev-parse --show-toplevel)

git worktree add $WORKTREE $BASE_BRANCH
ln -sfn $REPO_ROOT/workspace $WORKTREE/workspace
ln -sfn $REPO_ROOT/.venv $WORKTREE/.venv
```

### Step 5: Run Validation on Base Branch

Run 15 training steps in the base worktree. Only run the model(s) selected in Step 1.

```bash
cd $WORKTREE

# Single-node (replace <model> with deepseek-v2-lite or qwen3-30b-a3b)
bash .claude/skills/correctness-validation/scripts/launch_validate.sh <model>

# Multi-node (SLURM)
srun -W 0 .claude/skills/correctness-validation/scripts/launch_validate.sh <model>
```

The launch script auto-detects SLURM environment variables (`SLURM_NNODES`, `SLURM_NODEID`, `SLURM_STEP_GPUS`, `SLURM_STEP_NODELIST`) to configure `torchrun` arguments. On single-node, it falls back to localhost defaults.

Logs are written to `logging/correctness-validation/validate_<model>_node<N>.log`.

Return to the original repo directory after the run completes.

### Step 6: Run Validation on Feature Branch

Run the same 15 steps in the current (feature) working directory, for the same model(s).

```bash
cd $REPO_ROOT

# Single-node
bash .claude/skills/correctness-validation/scripts/launch_validate.sh <model>

# Multi-node (SLURM)
srun -W 0 .claude/skills/correctness-validation/scripts/launch_validate.sh <model>
```

### Step 7: Compare Results

Run the compare script for each model that was validated. Use the node-0 logs (rank 0 emits the metrics). Run `python3 .claude/skills/correctness-validation/scripts/compare.py --help` for full options.

```bash
python3 .claude/skills/correctness-validation/scripts/compare.py \
    $WORKTREE/logging/correctness-validation/validate_<model>_node0.log \
    logging/correctness-validation/validate_<model>_node0.log
```

The compare script parses both logs, extracts per-step metrics, and reports pass/fail. It checks:

- **cross-entropy-loss**: relative tolerance per step
- **load-balance-loss**: relative tolerance per step
- **gradient-norm**: relative tolerance per step

Default tolerance is 1e-3 relative difference. Use `--tolerance` to adjust.

Expected output on success:

```
PASS: All metrics within tolerance across all steps.
```

Expected output on failure:

```
FAIL: Metrics diverged beyond tolerance:
  cross-entropy-loss:
    step 003: cross-entropy-loss diverged — base=2.663700, feature=2.680100, rel_diff=6.16e-03 > tolerance=1e-03
```

### Step 8: Clean Up

```bash
git worktree remove $WORKTREE
```

## Log Format

The training scripts emit lines like:

```
2026-04-02 12:32:40 | INFO | step 00000001/00000015 | step-time 110.990 sec | cross-entropy-loss 2.6637 | load-balance-loss 0.001234 | learning-rate 1.000000e-06 | gradient-norm 20.3210 | tokens-per-second 18,895 | peak-gpu-memory 47.20 GB
```

The compare script parses pipe-separated key-value pairs from lines containing `| INFO | step `.

## Common Issues

### Setup fails on HuggingFace download

Ensure `HF_TOKEN` is set if the model is gated. DeepSeek-V2-Lite and Qwen3-30B-A3B are public models.

### OOM during validation

DeepSeek-V2-Lite requires 4x B200 GPUs. Qwen3-30B-A3B requires 16x B200 GPUs. If OOM occurs, check that no other processes are using GPU memory.

### Logs show no load-balance-loss

The validation scripts set `moe_load_balance_coef > 0` to ensure this metric is logged. If it is missing, check that the validation script (not an example script) was used.

### Tolerance too strict

FP8 operations and flash attention can introduce small non-determinism. If validation fails with very small differences, try increasing tolerance:

```bash
python3 .claude/skills/correctness-validation/scripts/compare.py \
    base.log feature.log --tolerance 4e-3
```

### Worktree conflicts

If the worktree was not cleaned up from a previous run, use `git worktree list` to find it and `git worktree remove <path> --force` to remove it.
