"""
Compare training logs from two branches for correctness validation.

Parses stdout logs from PithTrain training runs, extracts per-step metrics
(cross-entropy-loss, load-balance-loss, gradient-norm), and reports whether
the two runs produce consistent results within a configurable tolerance.

Usage:
    python3 compare.py <base_log> <feature_log> [--tolerance 1e-3]

Exit code 0 = PASS, exit code 1 = FAIL.
"""

import argparse
import re
import sys

METRICS = ["cross-entropy-loss", "load-balance-loss", "gradient-norm"]
STEP_PATTERN = re.compile(r"step\s+(\d+)/(\d+)")


def parse_log(path):
    """
    Parse a PithTrain training log and extract per-step metrics.

    Parameters
    ----
    path : str
        Path to the log file.

    Returns
    ----
    steps : list[dict[str, float]]
        List of dicts, one per step, with metric names as keys.
    """
    steps = []

    with open(path) as f:
        for line in f:
            if "| INFO | step " not in line:
                continue

            parts = line.split("|")
            metrics = dict()

            for part in parts:
                part = part.strip()

                m = STEP_PATTERN.match(part)
                if m:
                    metrics["step"] = int(m.group(1))
                    continue

                tokens = part.rsplit(None, 1)
                if len(tokens) == 2:
                    key = tokens[0].strip()
                    val = tokens[1].strip().replace(",", "")
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass

            if "step" in metrics:
                steps.append(metrics)

    return steps


def compare_metric(base_steps, feature_steps, metric, tolerance):
    """
    Compare a single metric across steps.

    Parameters
    ----
    base_steps : list[dict]
        Parsed steps from the base branch log.
    feature_steps : list[dict]
        Parsed steps from the feature branch log.
    metric : str
        Name of the metric to compare.
    tolerance : float
        Maximum allowed relative difference.

    Returns
    ----
    failures : list[str]
        List of failure messages. Empty if all steps pass.
    """
    failures = []

    for base, feature in zip(base_steps, feature_steps):
        step = base["step"]
        base_val = base.get(metric)
        feature_val = feature.get(metric)

        if base_val is None and feature_val is None:
            continue

        if base_val is None or feature_val is None:
            failures.append(
                f"  step {step:03d}: {metric} present in one log but not the other "
                f"(base={base_val}, feature={feature_val})"
            )
            continue

        if base_val == 0 and feature_val == 0:
            continue

        denom = abs(base_val) if base_val != 0 else abs(feature_val)
        rel_diff = abs(base_val - feature_val) / denom

        if rel_diff > tolerance:
            failures.append(
                f"  step {step:03d}: {metric} diverged — "
                f"base={base_val:.6f}, feature={feature_val:.6f}, "
                f"rel_diff={rel_diff:.2e} > tolerance={tolerance:.0e}"
            )

    return failures


def print_comparison_table(base_steps, feature_steps):
    """
    Print a step-by-step comparison table to stdout.

    Parameters
    ----
    base_steps : list[dict]
        Parsed steps from the base branch log.
    feature_steps : list[dict]
        Parsed steps from the feature branch log.
    """
    print("Step-by-step comparison:")
    print("-" * 100)

    header = f"{'step':>5}"
    for metric in METRICS:
        header += f" | {'base ' + metric:>28} {'feature':>12} {'rel_diff':>10}"
    print(header)
    print("-" * 100)

    for base, feature in zip(base_steps, feature_steps):
        step = base["step"]
        row = f"{step:5d}"
        for metric in METRICS:
            bv = base.get(metric)
            fv = feature.get(metric)
            if bv is not None and fv is not None:
                denom = abs(bv) if bv != 0 else (abs(fv) if fv != 0 else 1.0)
                rd = abs(bv - fv) / denom
                row += f" | {bv:28.6f} {fv:12.6f} {rd:10.2e}"
            elif bv is not None:
                row += f" | {bv:28.6f} {'N/A':>12} {'N/A':>10}"
            elif fv is not None:
                row += f" | {'N/A':>28} {fv:12.6f} {'N/A':>10}"
            else:
                row += f" | {'N/A':>28} {'N/A':>12} {'N/A':>10}"
        print(row)

    print("-" * 100)
    print()


def main():
    """
    Entry point. Parse arguments, compare logs, and report pass/fail.
    """
    parser = argparse.ArgumentParser(description="Compare PithTrain training logs.")
    parser.add_argument("base_log", help="Path to base branch log file")
    parser.add_argument("feature_log", help="Path to feature branch log file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Relative tolerance for metric comparison (default: 1e-3)",
    )
    args = parser.parse_args()

    base_steps = parse_log(args.base_log)
    feature_steps = parse_log(args.feature_log)

    print(f"Base log:    {args.base_log} ({len(base_steps)} steps)")
    print(f"Feature log: {args.feature_log} ({len(feature_steps)} steps)")
    print(f"Tolerance:   {args.tolerance:.0e}")
    print()

    if len(base_steps) == 0:
        print("FAIL: No training steps found in base log.")
        sys.exit(1)

    if len(feature_steps) == 0:
        print("FAIL: No training steps found in feature log.")
        sys.exit(1)

    if len(base_steps) != len(feature_steps):
        print(
            f"WARNING: Step count mismatch — base has {len(base_steps)}, "
            f"feature has {len(feature_steps)}. Comparing first "
            f"{min(len(base_steps), len(feature_steps))} steps."
        )
        print()

    all_failures = dict()
    for metric in METRICS:
        failures = compare_metric(base_steps, feature_steps, metric, args.tolerance)
        if failures:
            all_failures[metric] = failures

    print_comparison_table(base_steps, feature_steps)

    if not all_failures:
        print("PASS: All metrics within tolerance across all steps.")
        sys.exit(0)
    else:
        print("FAIL: Metrics diverged beyond tolerance:")
        for metric, failures in all_failures.items():
            print(f"\n  {metric}:")
            for f in failures:
                print(f)
        sys.exit(1)


if __name__ == "__main__":
    main()
