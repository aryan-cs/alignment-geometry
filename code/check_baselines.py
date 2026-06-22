#!/usr/bin/env python3
"""Validate a baseline bake-off result for the misalignment direction study.

Expected input: results/data/baselines.json with a top-level ``methods`` object.
Each method should contain detection metrics, and methods with interventions
should contain control metrics. The validator enforces that the proposed
weight-spectral method is compared against simpler baselines rather than reported
in isolation.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_activation_pca_artifact import validate as validate_activation_pca  # noqa: E402


REQUIRED_METHODS = [
    "weight_svd",
    "activation_pca",
    "random_projection",
    "diff_of_means",
]

SCORE_DEFINITION = "||v^T dW||_2 / ||dW||_F"
ROOT = Path(__file__).resolve().parents[1]


def add(errors, context, msg):
    errors.append(f"{context}: {msg}")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return set(line.strip() for line in proc.stdout.splitlines() if line.strip())


def resolve_artifact(path_text):
    path = Path(path_text)
    full = path if path.is_absolute() else ROOT / path
    try:
        rel = str(full.resolve().relative_to(ROOT))
    except ValueError:
        rel = None
    return full, rel


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


def auc_from_scores(mis_scores, ben_scores):
    total = 0
    wins = 0.0
    for mis in mis_scores:
        for ben in ben_scores:
            total += 1
            if mis > ben:
                wins += 1.0
            elif mis == ben:
                wins += 0.5
    return wins / total if total else None


def validate_detection(method, row, errors, min_folds):
    det = row.get("detection")
    ctx = f"methods.{method}.detection"
    if not isinstance(det, dict):
        add(errors, ctx, "missing detection object")
        return None, None
    margin = finite(det.get("mean_margin"), f"{ctx}.mean_margin", errors)
    auc = det.get("auc")
    if auc is not None:
        auc = finite(auc, f"{ctx}.auc", errors, 0.0, 1.0)
    ratio = parse_ratio(det.get("mis_above_ben"))
    folds = det.get("folds")
    fold_signature = None
    if not isinstance(folds, list) or len(folds) < min_folds:
        add(errors, f"{ctx}.folds", f"must contain at least {min_folds} folds")
    else:
        wins = 0
        margins = []
        mis_scores = []
        ben_scores = []
        held = []
        for i, fold in enumerate(folds):
            fctx = f"{ctx}.folds[{i}]"
            if not isinstance(fold, dict):
                add(errors, fctx, "fold must be an object")
                continue
            held_idx = fold.get("held")
            if not isinstance(held_idx, int) or held_idx < 0:
                add(errors, f"{fctx}.held", "must be a non-negative integer")
            else:
                held.append(held_idx)
            mis = finite(fold.get("mis_score"), f"{fctx}.mis_score", errors, 0.0, 1.0)
            ben = finite(fold.get("ben_score"), f"{fctx}.ben_score", errors, 0.0, 1.0)
            if mis is None or ben is None:
                continue
            wins += int(mis > ben)
            margins.append(mis - ben)
            mis_scores.append(mis)
            ben_scores.append(ben)
        fold_signature = tuple(held) if len(held) == len(folds) else None
        if ratio is None:
            add(errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
        elif ratio != (wins, len(folds)):
            add(errors, f"{ctx}.mis_above_ben", f"{ratio} does not match fold scores {(wins, len(folds))}")
        empirical_margin = sum(margins) / len(margins) if margins else None
        if margin is not None and empirical_margin is not None and abs(margin - empirical_margin) > 1e-9:
            add(errors, f"{ctx}.mean_margin", f"{margin:.12g} != fold mean {empirical_margin:.12g}")
        empirical_auc = auc_from_scores(mis_scores, ben_scores)
        if auc is not None and empirical_auc is not None and abs(auc - empirical_auc) > 1e-9:
            add(errors, f"{ctx}.auc", f"{auc:.12g} != fold AUC {empirical_auc:.12g}")
    if ratio is not None:
        wins, folds = ratio
        if folds < min_folds:
            add(errors, f"{ctx}.mis_above_ben", f"fold count {folds} < {min_folds}")
        if method == "weight_svd" and wins != folds:
            add(errors, f"{ctx}.mis_above_ben", f"weight_svd must win every fold, got {wins}/{folds}")
    elif method == "weight_svd":
        add(errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
    return margin, fold_signature


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
    if data.get("schema") != "baseline_bakeoff_v1":
        errors.append("root: schema must be baseline_bakeoff_v1")
    if data.get("schema_version") != 1:
        errors.append("root: schema_version must be 1")
    if data.get("score") != SCORE_DEFINITION:
        errors.append(f"root: score must be {SCORE_DEFINITION!r}")
    methods = data.get("methods")
    if not isinstance(methods, dict):
        return ["root: methods must be an object"]
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        errors.append("root: missing required methods " + ", ".join(missing))
    margins = {}
    fold_signatures = {}
    tracked = tracked_files()
    if tracked is None:
        errors.append("root: git ls-files failed")
        tracked = set()
    for method, row in methods.items():
        if not isinstance(row, dict):
            add(errors, f"methods.{method}", "method row must be an object")
            continue
        margin, signature = validate_detection(method, row, errors, args.min_folds)
        if margin is not None:
            margins[method] = margin
        if signature is not None:
            fold_signatures[method] = signature
        if method == "activation_pca":
            if row.get("source") != "external_activation_artifact":
                add(errors, "methods.activation_pca.source", "must be external_activation_artifact")
            artifact_path = row.get("artifact_path")
            if not isinstance(artifact_path, str) or not artifact_path:
                add(errors, "methods.activation_pca.artifact_path", "must be a nonempty string")
                artifact_full = None
                artifact_rel = None
            else:
                artifact_full, artifact_rel = resolve_artifact(artifact_path)
                if artifact_rel is None:
                    add(errors, "methods.activation_pca.artifact_path", "must point inside the repository")
                elif not os.path.exists(artifact_full):
                    add(errors, "methods.activation_pca.artifact_path", f"missing artifact {artifact_rel}")
                else:
                    if os.path.getsize(artifact_full) <= 0:
                        add(errors, "methods.activation_pca.artifact_path", f"empty artifact {artifact_rel}")
                    if getattr(args, "require_tracked_artifacts", False) and artifact_rel not in tracked:
                        add(errors, "methods.activation_pca.artifact_path", f"untracked artifact {artifact_rel}")
            digest = row.get("artifact_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                add(errors, "methods.activation_pca.artifact_sha256", "must be a sha256 hex digest")
            elif artifact_path and artifact_rel is not None and artifact_full is not None and os.path.exists(artifact_full):
                observed = file_sha256(artifact_full)
                if observed != digest:
                    add(errors, "methods.activation_pca.artifact_sha256", "hash mismatch")
                try:
                    artifact_data = json.load(open(artifact_full))
                    artifact_errors = validate_activation_pca(artifact_data, args.min_folds)
                except Exception as exc:
                    artifact_errors = [f"failed to validate artifact: {exc}"]
                for artifact_error in artifact_errors:
                    add(errors, "methods.activation_pca.artifact", artifact_error)
        validate_control(method, row, errors, args.min_control_drop)
    if fold_signatures:
        first_method, first_signature = next(iter(fold_signatures.items()))
        for method, signature in fold_signatures.items():
            if signature != first_signature:
                add(
                    errors,
                    f"methods.{method}.detection.folds",
                    f"held indices {signature} differ from {first_method} {first_signature}",
                )
    weight = margins.get("weight_svd")
    random = margins.get("random_projection")
    diff = margins.get("diff_of_means")
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
        if diff is not None and weight - diff < args.min_weight_over_diff:
            errors.append(
                "claim gate: weight_svd mean_margin must be at least diff_of_means "
                f"by {args.min_weight_over_diff:.3f}; got {weight - diff:.3f}"
            )
    return errors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/baselines.json")
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--min-weight-margin", type=float, default=0.05)
    ap.add_argument("--min-weight-over-random", type=float, default=0.05)
    ap.add_argument("--min-weight-over-diff", type=float, default=0.0)
    ap.add_argument("--min-control-drop", type=float, default=0.015)
    ap.add_argument(
        "--require-tracked-artifacts",
        action="store_true",
        help="require external artifacts referenced by the bake-off to be tracked by git",
    )
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
