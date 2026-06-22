#!/usr/bin/env python3
"""Validate cross-organism direction transfer evidence."""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def add(errors, context, message):
    errors.append(f"{context}: {message}")


def finite(x, context, errors, lo=None, hi=None):
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        add(errors, context, f"expected finite number, got {x!r}")
        return None
    value = float(x)
    if lo is not None and value < lo:
        add(errors, context, f"{value:.6g} < {lo:.6g}")
    if hi is not None and value > hi:
        add(errors, context, f"{value:.6g} > {hi:.6g}")
    return value


def parse_ratio(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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


def load_unit_npz_vector(path, key, context, errors):
    try:
        with np.load(path) as data:
            if key not in data:
                add(errors, context, f"missing {key}")
                return None, None
            vec = np.asarray(data[key], dtype=np.float64)
    except Exception as exc:
        add(errors, context, f"failed to load npz: {exc}")
        return None, None
    if vec.ndim != 1:
        add(errors, context, f"{key} must be a vector, got shape {vec.shape}")
        return None, None
    if not np.all(np.isfinite(vec)):
        add(errors, context, f"{key} contains non-finite values")
        return None, None
    norm = float(np.linalg.norm(vec))
    if not (0.5 <= norm <= 1.5):
        add(errors, context, f"{key} norm {norm:.4g} outside expected unit-vector range")
        return None, norm
    return vec / (norm + 1e-12), norm


def validate_direction_meta(data, errors):
    directions = data.get("directions")
    if not isinstance(directions, dict):
        add(errors, "directions", "must be an object")
        return {}
    hashes = data.get("artifact_sha256")
    if not isinstance(hashes, dict):
        add(errors, "artifact_sha256", "must record hashes for referenced direction artifacts")
        hashes = {}
    tracked = tracked_files()
    if tracked is None:
        add(errors, "git", "git ls-files failed")
        tracked = set()
    shapes = []
    vectors = {}
    for side in ("source", "target"):
        row = directions.get(side)
        ctx = f"directions.{side}"
        if not isinstance(row, dict):
            add(errors, ctx, "missing direction metadata")
            continue
        for key in ("tag", "path", "key"):
            if not isinstance(row.get(key), str) or not row.get(key):
                add(errors, f"{ctx}.{key}", "must be a nonempty string")
        shape = row.get("shape")
        expected_shape = None
        if not (
            isinstance(shape, list)
            and len(shape) == 1
            and isinstance(shape[0], int)
            and shape[0] > 0
        ):
            add(errors, f"{ctx}.shape", "must be a one-dimensional positive shape")
        else:
            expected_shape = tuple(shape)
            shapes.append(expected_shape)
        reported_norm = finite(row.get("raw_norm"), f"{ctx}.raw_norm", errors, 0.5, 1.5)
        path_text = row.get("path")
        key = row.get("key")
        if not isinstance(path_text, str) or not isinstance(key, str):
            continue
        full, rel = resolve_artifact(path_text)
        if rel is None:
            add(errors, f"{ctx}.path", "must point inside the repository")
            continue
        if not full.exists():
            add(errors, f"{ctx}.path", f"missing referenced artifact {rel}")
            continue
        if full.stat().st_size <= 0:
            add(errors, f"{ctx}.path", f"referenced artifact is empty: {rel}")
        if rel not in tracked:
            add(errors, f"{ctx}.path", f"referenced artifact is not tracked: {rel}")
        expected_hash = hashes.get(rel)
        if not isinstance(expected_hash, str) or not expected_hash:
            add(errors, "artifact_sha256", f"missing hash for {rel}")
        else:
            observed_hash = file_sha256(full)
            if observed_hash != expected_hash:
                add(errors, "artifact_sha256", f"{rel} hash mismatch")
        vector, observed_norm = load_unit_npz_vector(full, key, f"{ctx}.npz", errors)
        if (
            vector is not None
            and reported_norm is not None
            and observed_norm is not None
            and abs(observed_norm - reported_norm) > 1e-6
        ):
            add(errors, f"{ctx}.raw_norm", f"reported {reported_norm:.12g} != npz norm {observed_norm:.12g}")
        if vector is not None:
            if expected_shape is not None and vector.shape != expected_shape:
                add(errors, f"{ctx}.shape", f"reported {expected_shape} != npz shape {vector.shape}")
            vectors[side] = vector
    if len(shapes) == 2 and shapes[0] != shapes[1]:
        add(errors, "directions", f"source and target shapes differ: {shapes[0]} vs {shapes[1]}")
    return vectors


def validate_fold_scores(folds, ctx, errors):
    wins = 0
    margins = []
    random_margins = []
    for idx, fold in enumerate(folds):
        fctx = f"{ctx}.folds[{idx}]"
        if not isinstance(fold, dict):
            add(errors, fctx, "fold must be an object")
            continue
        mis = finite(fold.get("mis_score"), f"{fctx}.mis_score", errors, 0.0, 1.0)
        ben = finite(fold.get("ben_score"), f"{fctx}.ben_score", errors, 0.0, 1.0)
        mis_rand = finite(fold.get("mis_rand"), f"{fctx}.mis_rand", errors, 0.0, 1.0)
        ben_rand = finite(fold.get("ben_rand"), f"{fctx}.ben_rand", errors, 0.0, 1.0)
        if mis is not None and ben is not None:
            wins += int(mis > ben)
            margins.append(mis - ben)
        if mis_rand is not None and ben_rand is not None:
            random_margins.append(mis_rand - ben_rand)
    return wins, margins, random_margins


def validate_cross_detection_row(row, ctx, args, errors):
    if not isinstance(row, dict):
        add(errors, ctx, "missing cross-detection row")
        return
    for key in ("direction_tag", "scored_tag"):
        if not isinstance(row.get(key), str) or not row.get(key):
            add(errors, f"{ctx}.{key}", "must be a nonempty string")
    folds = row.get("folds")
    if not isinstance(folds, list) or len(folds) < args.min_folds:
        add(errors, f"{ctx}.folds", f"must contain at least {args.min_folds} folds")
        return
    wins, margins, random_margins = validate_fold_scores(folds, ctx, errors)
    ratio = parse_ratio(row.get("mis_above_ben"))
    if ratio is None:
        add(errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
    elif ratio != (wins, len(folds)):
        add(errors, f"{ctx}.mis_above_ben", f"{ratio} does not match fold scores {(wins, len(folds))}")
    if wins != len(folds):
        add(errors, ctx, f"misaligned score must exceed benign score in every fold; got {wins}/{len(folds)}")
    mean_margin = finite(row.get("mean_margin"), f"{ctx}.mean_margin", errors)
    empirical = sum(margins) / len(margins) if margins else None
    if mean_margin is not None and empirical is not None:
        if abs(mean_margin - empirical) > 1e-9:
            add(errors, ctx, f"mean_margin {mean_margin:.12g} != fold mean {empirical:.12g}")
        if mean_margin < args.min_margin:
            add(errors, ctx, f"mean_margin {mean_margin:.3f} below {args.min_margin:.3f}")
    mean_random = finite(row.get("mean_random_margin"), f"{ctx}.mean_random_margin", errors)
    empirical_random = sum(random_margins) / len(random_margins) if random_margins else None
    if mean_random is not None and empirical_random is not None:
        if abs(mean_random - empirical_random) > 1e-9:
            add(errors, ctx, f"mean_random_margin {mean_random:.12g} != fold mean {empirical_random:.12g}")
    if mean_margin is not None and mean_random is not None:
        gap = mean_margin - mean_random
        if gap < args.min_over_random:
            add(errors, ctx, f"mean margin exceeds random by {gap:.3f}, below {args.min_over_random:.3f}")


def validate(data, args):
    errors = []
    if data.get("schema") != "cross_organism_v1":
        add(errors, "schema", "must be cross_organism_v1")
    for key in ("source_tag", "target_tag", "matrix", "key"):
        if not isinstance(data.get(key), str) or not data.get(key):
            add(errors, key, "must be a nonempty string")
    if data.get("source_tag") == data.get("target_tag"):
        add(errors, "tags", "source_tag and target_tag must differ")
    if not isinstance(data.get("layer"), int) or data["layer"] < 0:
        add(errors, "layer", "must be a non-negative integer")
    vectors = validate_direction_meta(data, errors)
    cosine = finite(data.get("direction_cosine"), "direction_cosine", errors, -1.0, 1.0)
    cosine_abs = finite(data.get("direction_cosine_abs"), "direction_cosine_abs", errors, 0.0, 1.0)
    if cosine is not None and cosine_abs is not None and abs(abs(cosine) - cosine_abs) > 1e-9:
        add(errors, "direction_cosine_abs", "must equal abs(direction_cosine)")
    if cosine_abs is not None and cosine_abs < args.min_cos_abs:
        add(errors, "direction_cosine_abs", f"{cosine_abs:.3f} below {args.min_cos_abs:.3f}")
    if set(vectors) == {"source", "target"}:
        observed_cosine = float(vectors["source"] @ vectors["target"])
        if cosine is not None and abs(observed_cosine - cosine) > 1e-6:
            add(errors, "direction_cosine", f"reported {cosine:.12g} != npz cosine {observed_cosine:.12g}")
        if cosine_abs is not None and abs(abs(observed_cosine) - cosine_abs) > 1e-6:
            add(errors, "direction_cosine_abs", "does not match referenced npz vectors")
    similarity = data.get("direction_similarity")
    layer_key = str(data.get("layer"))
    if not isinstance(similarity, dict) or not isinstance(similarity.get("per_layer"), dict):
        add(errors, "direction_similarity.per_layer", "must record per-layer direction similarity")
    else:
        layer_row = similarity["per_layer"].get(layer_key)
        if not isinstance(layer_row, dict):
            add(errors, f"direction_similarity.per_layer.{layer_key}", "missing required layer row")
        else:
            layer_abs = finite(
                layer_row.get("abs_cos_wdsv"),
                f"direction_similarity.per_layer.{layer_key}.abs_cos_wdsv",
                errors,
                0.0,
                1.0,
            )
            if cosine_abs is not None and layer_abs is not None and abs(layer_abs - cosine_abs) > 1e-9:
                add(errors, f"direction_similarity.per_layer.{layer_key}.abs_cos_wdsv", "does not match root cosine")

    cross = data.get("cross_detection")
    if not isinstance(cross, dict):
        add(errors, "cross_detection", "missing cross-detection evidence")
    else:
        validate_cross_detection_row(
            cross.get("source_direction_on_target"),
            "cross_detection.source_direction_on_target",
            args,
            errors,
        )
        validate_cross_detection_row(
            cross.get("target_direction_on_source"),
            "cross_detection.target_direction_on_source",
            args,
            errors,
        )
    return errors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/cross_organism.json")
    ap.add_argument("--min-cos-abs", type=float, default=0.30)
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--min-margin", type=float, default=0.05)
    ap.add_argument("--min-over-random", type=float, default=0.02)
    return ap.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input))
    errors = validate(data, args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated cross-organism transfer {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
