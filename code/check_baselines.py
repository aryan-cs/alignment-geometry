#!/usr/bin/env python3
"""Validate a baseline bake-off result for the misalignment direction study.

Expected input: results/data/baselines.json with a top-level ``methods`` object.
Each method should contain detection metrics, and methods with interventions
should contain control metrics. The validator enforces that the proposed
weight-spectral method is compared against simpler baselines rather than reported
in isolation.
"""
import argparse
import json
import math
import re
import sys


REQUIRED_METHODS = [
    "weight_svd",
    "activation_pca",
    "random_projection",
    "diff_of_means",
]


def add(errors, context, msg):
    errors.append(f"{context}: {msg}")


def finite(x, context, errors, lo=None, hi=None):
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        add(errors, context, f"expected finite number, got {x!r}")
        return None
    v = float(x)
    if lo is not None and v < lo:
        add(errors, context, f"{v:.6g} < {lo:.6g}")
    if hi is not None and v > hi:
        add(errors, context, f"{v:.6g} > {hi:.6g}")
    return v


def parse_ratio(text):
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def validate_detection(method, row, errors, min_folds):
    det = row.get("detection")
    ctx = f"methods.{method}.detection"
    if not isinstance(det, dict):
        add(errors, ctx, "missing detection object")
        return None
    margin = finite(det.get("mean_margin"), f"{ctx}.mean_margin", errors)
    auc = det.get("auc")
    if auc is not None:
        finite(auc, f"{ctx}.auc", errors, 0.0, 1.0)
    ratio = parse_ratio(det.get("mis_above_ben"))
    if ratio is not None:
        wins, folds = ratio
        if folds < min_folds:
            add(errors, f"{ctx}.mis_above_ben", f"fold count {folds} < {min_folds}")
        if method == "weight_svd" and wins != folds:
            add(errors, f"{ctx}.mis_above_ben", f"weight_svd must win every fold, got {wins}/{folds}")
    elif method == "weight_svd":
        add(errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
    return margin


def validate_control(method, row, errors, min_drop):
    ctrl = row.get("control")
    if ctrl is None:
        return
    ctx = f"methods.{method}.control"
    if not isinstance(ctrl, dict):
        add(errors, ctx, "control must be an object")
        return
    base = finite(ctrl.get("baseline_rate"), f"{ctx}.baseline_rate", errors, 0.0, 1.0)
    intervention = finite(ctrl.get("intervention_rate"), f"{ctx}.intervention_rate", errors, 0.0, 1.0)
    random = ctrl.get("random_rate")
    if random is not None:
        random = finite(random, f"{ctx}.random_rate", errors, 0.0, 1.0)
    if method == "weight_svd" and base is not None and intervention is not None:
        if base - intervention < min_drop:
            add(errors, ctx, f"baseline-intervention drop {base - intervention:.3f} < {min_drop:.3f}")
    if method == "weight_svd" and random is not None and intervention is not None:
        if random - intervention < min_drop:
            add(errors, ctx, f"random-intervention gap {random - intervention:.3f} < {min_drop:.3f}")


def validate(data, args):
    errors = []
    methods = data.get("methods")
    if not isinstance(methods, dict):
        return ["root: methods must be an object"]
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        errors.append("root: missing required methods " + ", ".join(missing))
    margins = {}
    for method, row in methods.items():
        if not isinstance(row, dict):
            add(errors, f"methods.{method}", "method row must be an object")
            continue
        margin = validate_detection(method, row, errors, args.min_folds)
        if margin is not None:
            margins[method] = margin
        validate_control(method, row, errors, args.min_control_drop)
    weight = margins.get("weight_svd")
    random = margins.get("random_projection")
    if weight is not None:
        if weight < args.min_weight_margin:
            errors.append(
                f"claim gate: weight_svd mean_margin {weight:.3f} < {args.min_weight_margin:.3f}"
            )
        if random is not None and weight - random < args.min_weight_over_random:
            errors.append(
                "claim gate: weight_svd mean_margin must exceed random_projection "
                f"by {args.min_weight_over_random:.3f}; got {weight - random:.3f}"
            )
    return errors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/baselines.json")
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--min-weight-margin", type=float, default=0.05)
    ap.add_argument("--min-weight-over-random", type=float, default=0.05)
    ap.add_argument("--min-control-drop", type=float, default=0.015)
    return ap.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input))
    errors = validate(data, args)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"validated baseline bake-off {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
