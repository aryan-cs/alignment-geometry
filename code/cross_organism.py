#!/usr/bin/env python3
"""Cross-organism direction transfer and cross-detection.

This script compares two recovered WDSV directions in the same model basis and,
unless explicitly asked to skip it, scores each organism's held-out arms with the
other organism's direction:

    score(dW, v) = || v^T dW ||_2 / || dW ||_F

The scoring rule is the same reusable-probe statistic used by
``detect_holdout.py``. The script does not synthesize evidence: cross-detection
requires real base and arm checkpoint paths.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def relpath(path):
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_snapshot(path):
    path = Path(path)
    if (path / "model.safetensors.index.json").exists() or (path / "model.safetensors").exists():
        return str(path)
    snapshots = sorted(path.glob("snapshots/*"))
    return str(snapshots[0]) if snapshots else str(path)


def unit(vec, context):
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{context}: expected vector, got shape {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{context}: vector contains non-finite values")
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"{context}: zero-norm vector")
    return vec / norm, norm


def load_direction(path, key):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: missing direction npz")
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path}: missing {key}")
        return unit(np.asarray(data[key]), f"{path}:{key}")


def arm_paths(runs, pattern, context):
    paths = sorted(glob.glob(str(Path(runs) / pattern)))
    if not paths:
        raise FileNotFoundError(f"{context}: no arms matched {Path(runs) / pattern}")
    return paths


def load_deltas(base_weight, paths, tensor_name):
    deltas = []
    for path in paths:
        store = WeightStore(find_snapshot(path))
        weight = store.get(tensor_name).astype(np.float64)
        if weight.shape != base_weight.shape:
            raise ValueError(
                f"{path}: {tensor_name} shape {weight.shape} != base shape {base_weight.shape}"
            )
        deltas.append(weight - base_weight)
    return deltas


def score(delta, direction):
    return float(np.linalg.norm(direction @ delta) / (np.linalg.norm(delta) + 1e-12))


def cross_detect(direction, direction_tag, scored_tag, mis_deltas, ben_deltas, random_direction):
    n = min(len(mis_deltas), len(ben_deltas))
    folds = []
    for idx in range(n):
        fold = {
            "held": idx,
            "mis_score": score(mis_deltas[idx], direction),
            "ben_score": score(ben_deltas[idx], direction),
            "mis_rand": score(mis_deltas[idx], random_direction),
            "ben_rand": score(ben_deltas[idx], random_direction),
        }
        folds.append(fold)
    wins = sum(1 for fold in folds if fold["mis_score"] > fold["ben_score"])
    margins = [fold["mis_score"] - fold["ben_score"] for fold in folds]
    random_margins = [fold["mis_rand"] - fold["ben_rand"] for fold in folds]
    return {
        "direction_tag": direction_tag,
        "scored_tag": scored_tag,
        "folds": folds,
        "mis_above_ben": f"{wins}/{len(folds)}",
        "mean_margin": float(np.mean(margins)) if margins else None,
        "mean_random_margin": float(np.mean(random_margins)) if random_margins else None,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", default="med")
    ap.add_argument("--target-tag", default="code")
    ap.add_argument("--source-directions-npz", required=True)
    ap.add_argument("--target-directions-npz", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--key-template", default="wdsv_L{layer}")
    ap.add_argument("--matrix", default="self_attn.o_proj")
    ap.add_argument("--out", default="results/data/cross_organism.json")
    ap.add_argument("--skip-cross-detection", action="store_true")
    ap.add_argument("--base", help="base checkpoint used for both organism arm deltas")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--source-misaligned-glob")
    ap.add_argument("--source-benign-glob")
    ap.add_argument("--target-misaligned-glob")
    ap.add_argument("--target-benign-glob")
    ap.add_argument("--min-arm-pairs", type=int, default=4)
    return ap.parse_args()


def require(value, name):
    if value is None:
        raise ValueError(f"{name} is required unless --skip-cross-detection is set")
    return value


def main():
    args = parse_args()
    key = args.key_template.format(layer=args.layer)
    source_v, source_norm = load_direction(args.source_directions_npz, key)
    target_v, target_norm = load_direction(args.target_directions_npz, key)
    if source_v.shape != target_v.shape:
        raise ValueError(
            f"direction shapes differ: {source_v.shape} vs {target_v.shape}; "
            "cross-organism cosine requires a shared basis"
        )
    cosine = float(source_v @ target_v)
    payload = {
        "schema": "cross_organism_v1",
        "source_tag": args.source_tag,
        "target_tag": args.target_tag,
        "layer": args.layer,
        "matrix": args.matrix,
        "key": key,
        "directions": {
            "source": {
                "tag": args.source_tag,
                "path": relpath(args.source_directions_npz),
                "key": key,
                "shape": list(source_v.shape),
                "raw_norm": source_norm,
            },
            "target": {
                "tag": args.target_tag,
                "path": relpath(args.target_directions_npz),
                "key": key,
                "shape": list(target_v.shape),
                "raw_norm": target_norm,
            },
        },
        "direction_cosine": cosine,
        "direction_cosine_abs": abs(cosine),
        "direction_similarity": {
            "per_layer": {
                str(args.layer): {
                    "key": key,
                    "cos_wdsv": cosine,
                    "abs_cos_wdsv": abs(cosine),
                }
            }
        },
        "artifact_sha256": {
            relpath(args.source_directions_npz): file_sha256(args.source_directions_npz),
            relpath(args.target_directions_npz): file_sha256(args.target_directions_npz),
        },
    }

    if not args.skip_cross_detection:
        base = require(args.base, "--base")
        source_mis = arm_paths(args.runs, require(args.source_misaligned_glob, "--source-misaligned-glob"), "source misaligned")
        source_ben = arm_paths(args.runs, require(args.source_benign_glob, "--source-benign-glob"), "source benign")
        target_mis = arm_paths(args.runs, require(args.target_misaligned_glob, "--target-misaligned-glob"), "target misaligned")
        target_ben = arm_paths(args.runs, require(args.target_benign_glob, "--target-benign-glob"), "target benign")
        for label, paths in (
            ("source misaligned", source_mis),
            ("source benign", source_ben),
            ("target misaligned", target_mis),
            ("target benign", target_ben),
        ):
            if len(paths) < args.min_arm_pairs:
                raise ValueError(f"{label}: matched {len(paths)} arms, need >= {args.min_arm_pairs}")

        tensor_name = f"model.layers.{args.layer}.{args.matrix}.weight"
        base_store = WeightStore(find_snapshot(base))
        base_weight = base_store.get(tensor_name).astype(np.float64)
        if source_v.shape[0] != base_weight.shape[0]:
            raise ValueError(
                f"direction length {source_v.shape[0]} does not match {tensor_name} rows "
                f"{base_weight.shape[0]}"
            )
        source_mis_d = load_deltas(base_weight, source_mis, tensor_name)
        source_ben_d = load_deltas(base_weight, source_ben, tensor_name)
        target_mis_d = load_deltas(base_weight, target_mis, tensor_name)
        target_ben_d = load_deltas(base_weight, target_ben, tensor_name)

        rng = np.random.default_rng(0)
        random_direction = unit(rng.standard_normal(source_v.shape[0]), "random_direction")[0]
        payload["inputs"] = {
            "base": relpath(base),
            "runs": relpath(args.runs),
            "source_misaligned_glob": args.source_misaligned_glob,
            "source_benign_glob": args.source_benign_glob,
            "target_misaligned_glob": args.target_misaligned_glob,
            "target_benign_glob": args.target_benign_glob,
        }
        payload["cross_detection"] = {
            "source_direction_on_target": cross_detect(
                source_v, args.source_tag, args.target_tag, target_mis_d, target_ben_d, random_direction
            ),
            "target_direction_on_source": cross_detect(
                target_v, args.target_tag, args.source_tag, source_mis_d, source_ben_d, random_direction
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"wrote {args.out}; |cos({args.source_tag},{args.target_tag})|="
        f"{payload['direction_cosine_abs']:.3f}"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
