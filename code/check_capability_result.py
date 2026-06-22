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
PAPER_TASKS = ["mmlu", "gsm8k", "arc", "refusal"]
PAPER_MIN_SAMPLE_SIZES = {
    "mmlu": 500,
    "gsm8k": 150,
    "arc": 300,
    "refusal": 256,
}
PAPER_SPLITS = {
    "mmlu": "test",
    "gsm8k": "test",
    "arc": "test",
}
PAPER_MODEL_MARKERS = {
    "model": ["Meta-Llama-3-8B-Instruct"],
    "base": ["Meta-Llama-3-8B"],
    "instruct": ["Meta-Llama-3-8B-Instruct"],
}
PAPER_FORBIDDEN_MODEL_MARKERS = {
    "base": ["Instruct"],
}
PAPER_MAX_GSM8K_INVALID_RATE = 0.10


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


def validate(data, require_full=False, require_paper=False):
    errors = []
    warnings = []
    conditions = data.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        errors.append("root: conditions must be a non-empty object")
        return errors, warnings

    topk = data.get("topk")
    if require_full or require_paper:
        if not isinstance(topk, int):
            errors.append("root: topk must be an integer when full validation is set")
            required = ["baseline"]
        else:
            required = ["baseline", f"ablate_rand{topk}", f"ablate_top{topk}"]
        for cond in required:
            if cond not in conditions:
                errors.append(f"root: missing required condition {cond}")

    if require_paper:
        if data.get("layer") != 14:
            errors.append("root: paper capability study must use layer=14")
        if topk != 128:
            errors.append("root: paper capability study must use topk=128")
        for key, markers in PAPER_MODEL_MARKERS.items():
            val = data.get(key)
            if not isinstance(val, str) or not all(m in val for m in markers):
                errors.append(
                    f"root: {key} must identify {'/'.join(markers)}; got {val!r}"
                )
            for marker in PAPER_FORBIDDEN_MODEL_MARKERS.get(key, []):
                if isinstance(val, str) and marker in val:
                    errors.append(f"root: {key} must not identify {marker!r}; got {val!r}")
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            errors.append("root: paper capability study must record tasks as a list")
        else:
            missing = [task for task in PAPER_TASKS if task not in tasks]
            if missing:
                errors.append(f"root: paper capability study missing tasks {missing}")

        sample_sizes = data.get("sample_sizes")
        if not isinstance(sample_sizes, dict):
            errors.append("root: paper capability study must record sample_sizes")
        else:
            for task, minimum in PAPER_MIN_SAMPLE_SIZES.items():
                got = sample_sizes.get(task)
                if not isinstance(got, int) or got < minimum:
                    errors.append(
                        f"root: sample_sizes.{task} must be at least {minimum}; got {got!r}"
                    )

        sample_indices = data.get("sample_indices")
        if not isinstance(sample_indices, dict):
            errors.append("root: paper capability study must record sample_indices")
        elif isinstance(sample_sizes, dict):
            for task in PAPER_TASKS:
                idx = sample_indices.get(task)
                n = sample_sizes.get(task)
                if not isinstance(idx, list):
                    errors.append(f"root: sample_indices.{task} must be a list")
                elif isinstance(n, int) and len(idx) != n:
                    errors.append(
                        f"root: sample_indices.{task} length {len(idx)} != sample_sizes.{task} {n}"
                    )

        splits = data.get("splits")
        if not isinstance(splits, dict):
            errors.append("root: paper capability study must record dataset splits")
        else:
            for task, expected in PAPER_SPLITS.items():
                got = splits.get(task)
                if got != expected:
                    errors.append(
                        f"root: splits.{task} must be {expected!r}; got {got!r}"
                    )

        intervention = data.get("intervention")
        if not isinstance(intervention, dict):
            errors.append("root: paper capability study must record intervention metadata")
        else:
            expected_matrix = "model.layers.14.self_attn.o_proj.weight"
            if intervention.get("matrix") != expected_matrix:
                errors.append(
                    f"root: intervention.matrix must be {expected_matrix!r}"
                )
            if intervention.get("applied_to") != "every decoder layer residual stream":
                errors.append("root: intervention.applied_to must confirm all decoder layers")
            projection = str(intervention.get("projection", ""))
            if "h <- h - (h @ Q) @ Q.T" not in projection:
                errors.append("root: intervention.projection must record the residual projection")
            basis_meta = intervention.get("basis_metadata")
            if not isinstance(basis_meta, dict):
                errors.append("root: intervention.basis_metadata must be recorded")
            else:
                if basis_meta.get("matrix") != expected_matrix:
                    errors.append("root: intervention.basis_metadata.matrix mismatch")
                shape = basis_meta.get("shape")
                if not (
                    isinstance(shape, list)
                    and len(shape) == 2
                    and all(isinstance(x, int) and x > 0 for x in shape)
                ):
                    errors.append("root: intervention.basis_metadata.shape must be [rows, cols]")
                fro = basis_meta.get("delta_fro_norm")
                if not isinstance(fro, (int, float)) or not math.isfinite(fro) or fro <= 0:
                    errors.append("root: intervention.basis_metadata.delta_fro_norm must be positive")
                sv = basis_meta.get("top_singular_values")
                if not isinstance(sv, list) or not sv:
                    errors.append("root: intervention.basis_metadata.top_singular_values missing")
                else:
                    try:
                        svf = [float(x) for x in sv]
                    except (TypeError, ValueError):
                        errors.append("root: intervention.basis_metadata.top_singular_values must be numeric")
                    else:
                        if not all(math.isfinite(x) and x > 0 for x in svf):
                            errors.append("root: intervention.basis_metadata.top_singular_values must be positive")
                        if any(a < b for a, b in zip(svf, svf[1:])):
                            errors.append("root: intervention.basis_metadata.top_singular_values must be descending")

    required_tasks = selected_outputs(data)
    if not required_tasks:
        warnings.append("root: tasks metadata missing; validating observed task metrics only")
    if require_paper:
        required_tasks = [TASK_TO_OUTPUT[task] for task in PAPER_TASKS]

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
                if require_paper and task == "gsm8k":
                    invalid = metrics[task].get("invalid_predictions")
                    n = metrics[task].get("n")
                    if not isinstance(invalid, int) or invalid < 0:
                        add_error(errors, context, "invalid_predictions must be a non-negative integer")
                    elif isinstance(n, int) and n > 0:
                        rate = invalid / n
                        if rate > PAPER_MAX_GSM8K_INVALID_RATE:
                            add_error(
                                errors,
                                context,
                                "invalid_predictions exceeds "
                                f"{PAPER_MAX_GSM8K_INVALID_RATE:.0%}: {invalid}/{n}",
                            )
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
    ap.add_argument(
        "--require-paper", action="store_true",
        help=(
            "require the canonical paper capability study: layer 14, top-128, "
            "baseline/random/top conditions, MMLU/GSM8K/ARC/refusal, minimum "
            "sample sizes, dataset splits, and intervention provenance"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)
    errors, warnings = validate(data, args.require_full, args.require_paper)
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
