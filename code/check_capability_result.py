#!/usr/bin/env python3
"""Validate and summarize capability_eval.py JSON output."""
import argparse
import json
import math
import re
import sys


TASK_TO_OUTPUT = {
    "mmlu": ("mmlu", "accuracy"),
    "arc": ("arc_challenge", "accuracy"),
    "gsm8k": ("gsm8k", "accuracy"),
    "refusal": ("refusal", "rate"),
}
DISPLAY_TASKS = [
    ("mmlu", "MMLU", "accuracy"),
    ("arc_challenge", "ARC-C", "accuracy"),
    ("gsm8k", "GSM8K", "accuracy"),
    ("refusal", "refusal", "rate"),
]


def condition_order(keys, topk=None):
    def rank(k):
        if k == "baseline":
            return (0, 0, k)
        m = re.match(r"ablate_rand(\d+)$", k)
        if m:
            return (1, int(m.group(1)), k)
        m = re.match(r"ablate_top(\d+)$", k)
        if m:
            return (2, int(m.group(1)), k)
        return (3, topk if topk is not None else 0, k)

    return sorted(keys, key=rank)


def condition_label(cond):
    if cond == "baseline":
        return "baseline"
    m = re.match(r"ablate_rand(\d+)$", cond)
    if m:
        return f"random-{m.group(1)}"
    m = re.match(r"ablate_top(\d+)$", cond)
    if m:
        return f"top-{m.group(1)}"
    return cond


def add_error(errors, context, message):
    errors.append(f"{context}: {message}")


def validate_interval(metric, key, context, errors):
    vals = metric.get(key)
    if not isinstance(vals, list) or len(vals) != 3:
        add_error(errors, context, f"missing {key}=[point, lo, hi]")
        return None
    try:
        p, lo, hi = [float(v) for v in vals]
    except (TypeError, ValueError):
        add_error(errors, context, f"{key} contains non-numeric values")
        return None
    if not all(math.isfinite(v) for v in (p, lo, hi)):
        add_error(errors, context, f"{key} contains non-finite values")
        return None
    if not (0.0 <= lo <= p <= hi <= 1.0):
        add_error(errors, context, f"{key} must satisfy 0 <= lo <= point <= hi <= 1")
        return None
    return p, lo, hi


def validate_counts(metric, interval, count_key, context, errors):
    n = metric.get("n")
    count = metric.get(count_key)
    if not isinstance(n, int) or n <= 0:
        add_error(errors, context, "n must be a positive integer")
        return
    if not isinstance(count, int) or not (0 <= count <= n):
        add_error(errors, context, f"{count_key} must be an integer in [0, n]")
        return
    if interval is None:
        return
    p = interval[0]
    expected = count / n
    if abs(p - expected) > 1e-9:
        add_error(errors, context, f"point estimate {p:.12g} != {count_key}/n {expected:.12g}")


def selected_outputs(data):
    tasks = data.get("tasks")
    if not tasks:
        return []
    outputs = []
    for task in tasks:
        if task not in TASK_TO_OUTPUT:
            continue
        outputs.append(TASK_TO_OUTPUT[task])
    return outputs


def validate(data, require_full=False):
    errors = []
    warnings = []
    conditions = data.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        errors.append("root: conditions must be a non-empty object")
        return errors, warnings

    topk = data.get("topk")
    if require_full:
        if not isinstance(topk, int):
            errors.append("root: topk must be an integer when --require-full is set")
            required = ["baseline"]
        else:
            required = ["baseline", f"ablate_rand{topk}", f"ablate_top{topk}"]
        for cond in required:
            if cond not in conditions:
                errors.append(f"root: missing required condition {cond}")

    required_tasks = selected_outputs(data)
    if not required_tasks:
        warnings.append("root: tasks metadata missing; validating observed task metrics only")

    for cond, metrics in conditions.items():
        if not isinstance(metrics, dict):
            add_error(errors, cond, "condition value must be an object")
            continue
        expected = required_tasks or [
            (task, key) for task, _, key in DISPLAY_TASKS if task in metrics
        ]
        for task, key in expected:
            context = f"{cond}.{task}"
            if task not in metrics:
                add_error(errors, context, "missing selected task result")
                continue
            if not isinstance(metrics[task], dict):
                add_error(errors, context, "task result must be an object")
                continue
            interval = validate_interval(metrics[task], key, context, errors)
            if key == "accuracy":
                validate_counts(metrics[task], interval, "correct", context, errors)
            else:
                validate_counts(metrics[task], interval, "refusals", context, errors)

    return errors, warnings


def format_interval(metric, key):
    vals = metric.get(key)
    if not isinstance(vals, list) or len(vals) != 3:
        return "missing"
    p, lo, hi = [float(v) for v in vals]
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def summarize(data):
    conditions = data.get("conditions", {})
    conds = condition_order(conditions.keys(), data.get("topk"))
    print("summary:")
    for cond in conds:
        parts = []
        metrics = conditions[cond]
        for task, label, key in DISPLAY_TASKS:
            if task in metrics:
                parts.append(f"{label}={format_interval(metrics[task], key)}")
        print(f"  {condition_label(cond)}: " + "; ".join(parts))

    base = conditions.get("baseline")
    if not base:
        return
    print("delta vs baseline:")
    for cond in conds:
        if cond == "baseline":
            continue
        metrics = conditions[cond]
        parts = []
        for task, label, key in DISPLAY_TASKS:
            if task not in base or task not in metrics:
                continue
            b = base[task].get(key)
            v = metrics[task].get(key)
            if isinstance(b, list) and isinstance(v, list) and len(b) == 3 and len(v) == 3:
                parts.append(f"{label}={float(v[0]) - float(b[0]):+.3f}")
        print(f"  {condition_label(cond)}: " + "; ".join(parts))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/capability.json")
    ap.add_argument(
        "--require-full", action="store_true",
        help="require baseline, random-topk, top-topk, and all selected tasks",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)
    errors, warnings = validate(data, args.require_full)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summarize(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
