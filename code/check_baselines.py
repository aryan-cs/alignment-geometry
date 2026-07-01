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
POSITIVE_OUTCOME_MODE = "positive"
AUDIT_OUTCOME_MODE = "negative_or_inconclusive_audit"
OUTCOME_MODES = (POSITIVE_OUTCOME_MODE, AUDIT_OUTCOME_MODE)
FINAL_HANDOFF_MIN_FOLDS = 16


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


def wilson(k, n, z=1.96):
    if not isinstance(k, int) or not isinstance(n, int) or n <= 0 or k < 0 or k > n:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


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


def validate_detection(method, row, evidence_errors, claim_failures, args, n_pairs):
    det = row.get("detection")
    ctx = f"methods.{method}.detection"
    if not isinstance(det, dict):
        add(evidence_errors, ctx, "missing detection object")
        return None, None
    margin = finite(det.get("mean_margin"), f"{ctx}.mean_margin", evidence_errors)
    auc = det.get("auc")
    if auc is not None:
        auc = finite(auc, f"{ctx}.auc", evidence_errors, 0.0, 1.0)
    ratio = parse_ratio(det.get("mis_above_ben"))
    folds = det.get("folds")
    fold_signature = None
    if not isinstance(folds, list):
        add(evidence_errors, f"{ctx}.folds", "must be a list")
    else:
        if len(folds) < args.min_folds:
            add(
                evidence_errors,
                f"{ctx}.folds",
                f"must contain at least {args.min_folds} folds",
            )
        if n_pairs is not None and len(folds) != n_pairs:
            add(
                evidence_errors,
                f"{ctx}.folds",
                f"fold count {len(folds)} != inputs.n_pairs {n_pairs}",
            )
        wins = 0
        margins = []
        mis_scores = []
        ben_scores = []
        held = []
        for i, fold in enumerate(folds):
            fctx = f"{ctx}.folds[{i}]"
            if not isinstance(fold, dict):
                add(evidence_errors, fctx, "fold must be an object")
                continue
            held_idx = fold.get("held")
            if isinstance(held_idx, bool) or not isinstance(held_idx, int) or held_idx < 0:
                add(evidence_errors, f"{fctx}.held", "must be a non-negative integer")
            elif n_pairs is not None and held_idx >= n_pairs:
                add(
                    evidence_errors,
                    f"{fctx}.held",
                    f"index {held_idx} is outside inputs.n_pairs={n_pairs}",
                )
            else:
                held.append(held_idx)
            mis = finite(
                fold.get("mis_score"),
                f"{fctx}.mis_score",
                evidence_errors,
                0.0,
                1.0,
            )
            ben = finite(
                fold.get("ben_score"),
                f"{fctx}.ben_score",
                evidence_errors,
                0.0,
                1.0,
            )
            if mis is None or ben is None:
                continue
            wins += int(mis > ben)
            margins.append(mis - ben)
            mis_scores.append(mis)
            ben_scores.append(ben)
        if len(held) == len(folds):
            fold_signature = tuple(held)
            if len(set(held)) != len(held):
                add(evidence_errors, f"{ctx}.folds", "held indices must be unique")
            if n_pairs is not None and fold_signature != tuple(range(n_pairs)):
                add(
                    evidence_errors,
                    f"{ctx}.folds",
                    f"held indices must be exactly 0..{n_pairs - 1} in order",
                )
        if ratio is None:
            add(evidence_errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
        elif ratio != (wins, len(folds)):
            add(
                evidence_errors,
                f"{ctx}.mis_above_ben",
                f"{ratio} does not match fold scores {(wins, len(folds))}",
            )
        empirical_margin = sum(margins) / len(margins) if margins else None
        if margin is not None and empirical_margin is not None and abs(margin - empirical_margin) > 1e-9:
            add(
                evidence_errors,
                f"{ctx}.mean_margin",
                f"{margin:.12g} != fold mean {empirical_margin:.12g}",
            )
        empirical_auc = auc_from_scores(mis_scores, ben_scores)
        if auc is not None and empirical_auc is not None and abs(auc - empirical_auc) > 1e-9:
            add(
                evidence_errors,
                f"{ctx}.auc",
                f"{auc:.12g} != fold AUC {empirical_auc:.12g}",
            )
    if ratio is not None:
        wins, folds = ratio
        if folds < args.min_folds:
            add(
                evidence_errors,
                f"{ctx}.mis_above_ben",
                f"fold count {folds} < {args.min_folds}",
            )
        if method == "weight_svd" and wins != folds:
            add(
                claim_failures,
                f"{ctx}.mis_above_ben",
                f"weight_svd must win every fold, got {wins}/{folds}",
            )
        if method == "weight_svd":
            interval = wilson(wins, folds)
            if interval is None:
                add(evidence_errors, f"{ctx}.mis_above_ben", "could not compute Wilson interval")
            else:
                half_width = max(interval[0] - interval[1], interval[2] - interval[0])
                if half_width > args.max_weight_win_half_width:
                    add(
                        claim_failures,
                        f"{ctx}.mis_above_ben",
                        "Wilson half-width "
                        f"{half_width:.3f} > {args.max_weight_win_half_width:.3f} "
                        f"for {wins}/{folds} folds",
                    )
                if interval[1] < args.min_weight_win_lower:
                    add(
                        claim_failures,
                        f"{ctx}.mis_above_ben",
                        "Wilson lower bound "
                        f"{interval[1]:.3f} < {args.min_weight_win_lower:.3f} for {wins}/{folds} folds",
                    )
    elif method == "weight_svd":
        add(evidence_errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
    return margin, fold_signature


def validate_control(method, row, evidence_errors, claim_failures, min_drop):
    ctrl = row.get("control")
    if ctrl is None:
        return
    ctx = f"methods.{method}.control"
    if not isinstance(ctrl, dict):
        add(evidence_errors, ctx, "control must be an object")
        return
    base = finite(
        ctrl.get("baseline_rate"), f"{ctx}.baseline_rate", evidence_errors, 0.0, 1.0
    )
    intervention = finite(
        ctrl.get("intervention_rate"),
        f"{ctx}.intervention_rate",
        evidence_errors,
        0.0,
        1.0,
    )
    random = ctrl.get("random_rate")
    if random is not None:
        random = finite(random, f"{ctx}.random_rate", evidence_errors, 0.0, 1.0)
    if method == "weight_svd" and base is not None and intervention is not None:
        if base - intervention < min_drop:
            add(
                claim_failures,
                ctx,
                f"baseline-intervention drop {base - intervention:.3f} < {min_drop:.3f}",
            )
    if method == "weight_svd" and random is not None and intervention is not None:
        if random - intervention < min_drop:
            add(
                claim_failures,
                ctx,
                f"random-intervention gap {random - intervention:.3f} < {min_drop:.3f}",
            )


def validate_components(data, args):
    evidence_errors = []
    claim_failures = []
    if not isinstance(data, dict):
        return ["root: must be an object"], []
    if data.get("schema") != "baseline_bakeoff_v1":
        evidence_errors.append("root: schema must be baseline_bakeoff_v1")
    if data.get("schema_version") != 1:
        evidence_errors.append("root: schema_version must be 1")
    if data.get("score") != SCORE_DEFINITION:
        evidence_errors.append(f"root: score must be {SCORE_DEFINITION!r}")
    inputs = data.get("inputs")
    n_pairs = None
    if not isinstance(inputs, dict):
        evidence_errors.append("root: inputs must be an object")
    else:
        candidate = inputs.get("n_pairs")
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
            add(evidence_errors, "inputs.n_pairs", "must be a positive integer")
        else:
            n_pairs = candidate
            if n_pairs < args.min_folds:
                add(
                    evidence_errors,
                    "inputs.n_pairs",
                    f"{n_pairs} < minimum fold count {args.min_folds}",
                )
    methods = data.get("methods")
    if not isinstance(methods, dict):
        evidence_errors.append("root: methods must be an object")
        return evidence_errors, claim_failures
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        evidence_errors.append("root: missing required methods " + ", ".join(missing))
    margins = {}
    fold_signatures = {}
    tracked = tracked_files()
    if tracked is None:
        evidence_errors.append("root: git ls-files failed")
        tracked = set()
    for method, row in methods.items():
        if not isinstance(row, dict):
            add(evidence_errors, f"methods.{method}", "method row must be an object")
            continue
        margin, signature = validate_detection(
            method,
            row,
            evidence_errors,
            claim_failures,
            args,
            n_pairs,
        )
        if margin is not None:
            margins[method] = margin
        if signature is not None:
            fold_signatures[method] = signature
        if method == "activation_pca":
            if row.get("source") != "external_activation_artifact":
                add(
                    evidence_errors,
                    "methods.activation_pca.source",
                    "must be external_activation_artifact",
                )
            artifact_path = row.get("artifact_path")
            if not isinstance(artifact_path, str) or not artifact_path:
                add(
                    evidence_errors,
                    "methods.activation_pca.artifact_path",
                    "must be a nonempty string",
                )
                artifact_full = None
                artifact_rel = None
            else:
                artifact_full, artifact_rel = resolve_artifact(artifact_path)
                if artifact_rel is None:
                    add(
                        evidence_errors,
                        "methods.activation_pca.artifact_path",
                        "must point inside the repository",
                    )
                elif not os.path.exists(artifact_full):
                    add(
                        evidence_errors,
                        "methods.activation_pca.artifact_path",
                        f"missing artifact {artifact_rel}",
                    )
                else:
                    if os.path.getsize(artifact_full) <= 0:
                        add(
                            evidence_errors,
                            "methods.activation_pca.artifact_path",
                            f"empty artifact {artifact_rel}",
                        )
                    if getattr(args, "require_tracked_artifacts", False) and artifact_rel not in tracked:
                        add(
                            evidence_errors,
                            "methods.activation_pca.artifact_path",
                            f"untracked artifact {artifact_rel}",
                        )
            digest = row.get("artifact_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                add(
                    evidence_errors,
                    "methods.activation_pca.artifact_sha256",
                    "must be a sha256 hex digest",
                )
            elif artifact_path and artifact_rel is not None and artifact_full is not None and os.path.exists(artifact_full):
                observed = file_sha256(artifact_full)
                if observed != digest:
                    add(evidence_errors, "methods.activation_pca.artifact_sha256", "hash mismatch")
                try:
                    artifact_data = json.load(open(artifact_full))
                    artifact_errors = validate_activation_pca(artifact_data, args.min_folds)
                    artifact_provenance = artifact_data.get("provenance")
                    artifact_n_pairs = (
                        artifact_provenance.get("n_pairs")
                        if isinstance(artifact_provenance, dict)
                        else None
                    )
                    if n_pairs is not None and artifact_n_pairs != n_pairs:
                        add(
                            evidence_errors,
                            "methods.activation_pca.artifact.provenance.n_pairs",
                            f"{artifact_n_pairs!r} != inputs.n_pairs {n_pairs}",
                        )
                except Exception as exc:
                    artifact_errors = [f"failed to validate artifact: {exc}"]
                for artifact_error in artifact_errors:
                    add(evidence_errors, "methods.activation_pca.artifact", artifact_error)
        validate_control(
            method,
            row,
            evidence_errors,
            claim_failures,
            args.min_control_drop,
        )
    if fold_signatures:
        first_method, first_signature = next(iter(fold_signatures.items()))
        for method, signature in fold_signatures.items():
            if signature != first_signature:
                add(
                    evidence_errors,
                    f"methods.{method}.detection.folds",
                    f"held indices {signature} differ from {first_method} {first_signature}",
                )
    weight = margins.get("weight_svd")
    random = margins.get("random_projection")
    diff = margins.get("diff_of_means")
    if weight is not None:
        if weight < args.min_weight_margin:
            claim_failures.append(
                f"claim gate: weight_svd mean_margin {weight:.3f} < {args.min_weight_margin:.3f}"
            )
        if random is not None and weight - random < args.min_weight_over_random:
            claim_failures.append(
                "claim gate: weight_svd mean_margin must exceed random_projection "
                f"by {args.min_weight_over_random:.3f}; got {weight - random:.3f}"
            )
        if diff is not None and weight - diff < args.min_weight_over_diff:
            claim_failures.append(
                "claim gate: weight_svd mean_margin must be at least diff_of_means "
                f"by {args.min_weight_over_diff:.3f}; got {weight - diff:.3f}"
            )
    return evidence_errors, claim_failures


def validate_outcome_mode(mode, claim_failures):
    if mode == POSITIVE_OUTCOME_MODE:
        return list(claim_failures)
    if mode == AUDIT_OUTCOME_MODE:
        if claim_failures:
            return []
        return [
            "audit gate: negative_or_inconclusive_audit requires at least one "
            "frozen positive baseline criterion to fail; result satisfies all positive criteria"
        ]
    return [f"root: unsupported baseline_outcome_mode {mode!r}"]


def validate(data, args):
    evidence_errors, claim_failures = validate_components(data, args)
    mode = getattr(args, "baseline_outcome_mode", POSITIVE_OUTCOME_MODE)
    return evidence_errors + validate_outcome_mode(mode, claim_failures)


def audit_self_test():
    args = argparse.Namespace(
        min_folds=4,
        min_weight_margin=0.05,
        min_weight_over_random=0.05,
        min_weight_over_diff=0.0,
        min_weight_win_lower=0.50,
        max_weight_win_half_width=0.20,
        min_control_drop=0.015,
        require_tracked_artifacts=False,
    )

    def detection(one_loss=False, n_folds=16):
        folds = []
        for held in range(n_folds):
            mis_score = 0.8
            ben_score = 0.2
            if one_loss and held == n_folds - 1:
                mis_score = 0.1
            folds.append({"held": held, "mis_score": mis_score, "ben_score": ben_score})
        wins = sum(fold["mis_score"] > fold["ben_score"] for fold in folds)
        margins = [fold["mis_score"] - fold["ben_score"] for fold in folds]
        return {
            "folds": folds,
            "mis_above_ben": f"{wins}/{len(folds)}",
            "mean_margin": sum(margins) / len(margins),
            "auc": auc_from_scores(
                [fold["mis_score"] for fold in folds],
                [fold["ben_score"] for fold in folds],
            ),
        }

    failures = []
    positive_errors = []
    positive_claims = []
    validate_detection(
        "weight_svd",
        {"detection": detection()},
        positive_errors,
        positive_claims,
        args,
        16,
    )
    if positive_errors or positive_claims:
        failures.append("valid positive fold fixture failed validation")
    if validate_outcome_mode(POSITIVE_OUTCOME_MODE, positive_claims):
        failures.append("positive mode rejected a positive fixture")
    if not validate_outcome_mode(AUDIT_OUTCOME_MODE, positive_claims):
        failures.append("audit mode accepted a fully positive fixture")

    negative_errors = []
    negative_claims = []
    validate_detection(
        "weight_svd",
        {"detection": detection(one_loss=True)},
        negative_errors,
        negative_claims,
        args,
        16,
    )
    if negative_errors:
        failures.append("structurally valid negative fixture failed evidence validation")
    if not negative_claims:
        failures.append("negative fixture did not fail a frozen positive criterion")
    if not validate_outcome_mode(POSITIVE_OUTCOME_MODE, negative_claims):
        failures.append("positive mode accepted a negative fixture")
    if validate_outcome_mode(AUDIT_OUTCOME_MODE, negative_claims):
        failures.append("audit mode rejected a negative fixture")

    malformed = detection()
    malformed["folds"][-1]["held"] = 14
    malformed_errors = []
    validate_detection(
        "weight_svd",
        {"detection": malformed},
        malformed_errors,
        [],
        args,
        16,
    )
    if not any("held indices" in item for item in malformed_errors):
        failures.append("duplicate/noncanonical held indices were accepted")

    wrong_count_errors = []
    validate_detection(
        "activation_pca",
        {"detection": detection()},
        wrong_count_errors,
        [],
        args,
        15,
    )
    if not any("inputs.n_pairs" in item for item in wrong_count_errors):
        failures.append("fold count inconsistent with n_pairs was accepted")

    final_handoff_args = argparse.Namespace(**vars(args))
    final_handoff_args.min_folds = FINAL_HANDOFF_MIN_FOLDS
    underpowered_errors = []
    underpowered_claims = []
    validate_detection(
        "weight_svd",
        {"detection": detection(one_loss=True, n_folds=4)},
        underpowered_errors,
        underpowered_claims,
        final_handoff_args,
        4,
    )
    if not any("at least 16 folds" in item for item in underpowered_errors):
        failures.append("four-fold audit fixture was not rejected as underpowered")
    if not underpowered_claims:
        failures.append("underpowered negative fixture did not exercise an audit outcome")
    return failures


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/baselines.json")
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--min-weight-margin", type=float, default=0.05)
    ap.add_argument("--min-weight-over-random", type=float, default=0.05)
    ap.add_argument("--min-weight-over-diff", type=float, default=0.0)
    ap.add_argument("--min-weight-win-lower", type=float, default=0.50)
    ap.add_argument("--max-weight-win-half-width", type=float, default=0.20)
    ap.add_argument("--min-control-drop", type=float, default=0.015)
    ap.add_argument(
        "--baseline-outcome-mode",
        choices=OUTCOME_MODES,
        default=POSITIVE_OUTCOME_MODE,
        help=(
            "positive enforces all frozen claim gates; negative_or_inconclusive_audit "
            "requires hard evidence validity and at least one frozen positive gate failure"
        ),
    )
    ap.add_argument(
        "--require-tracked-artifacts",
        action="store_true",
        help="require external artifacts referenced by the bake-off to be tracked by git",
    )
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.self_test:
        failures = audit_self_test()
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1
        print("baseline outcome-mode self-test passed")
        return 0
    if (
        not math.isfinite(args.max_weight_win_half_width)
        or args.max_weight_win_half_width <= 0
        or args.max_weight_win_half_width > 1
    ):
        raise SystemExit("--max-weight-win-half-width must be in (0, 1]")
    data = json.load(open(args.input))
    errors = validate(data, args)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.baseline_outcome_mode == AUDIT_OUTCOME_MODE:
        print(f"validated baseline bake-off audit {args.input}")
    else:
        print(f"validated baseline bake-off {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
