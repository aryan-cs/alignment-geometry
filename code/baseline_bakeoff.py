#!/usr/bin/env python3
"""Build a real baseline bake-off artifact from matched checkpoint arms.

Weight-space baselines are computed directly from safetensors deltas. Activation
PCA is not a weight-space method, so this script requires a real external JSON
row for that method and refuses to synthesize it.
"""
import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402
from check_baselines import validate as validate_baselines  # noqa: E402
from run_environment import collect_run_environment  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPTS = [
    "code/run_baseline_bakeoff.sh",
    "code/activation_pca_baseline.py",
    "code/baseline_bakeoff.py",
    "code/check_baselines.py",
    "code/check_activation_pca_artifact.py",
    "code/check_run_manifest.py",
    "code/run_environment.py",
    "code/spectral.py",
]


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


def canonical_json_sha256(value):
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=ROOT, text=True).strip()
    except Exception:
        return None


def checkpoint_payload_errors(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index.exists():
        try:
            data = json.load(open(index))
        except Exception as exc:
            return [f"{index}: invalid JSON: {exc}"]
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return [f"{index}: missing nonempty weight_map"]
        errors = []
        for shard_name in sorted(set(weight_map.values())):
            if not isinstance(shard_name, str) or not shard_name:
                errors.append(f"{index}: invalid shard name {shard_name!r}")
                continue
            shard = path / shard_name
            if not shard.is_file() or shard.stat().st_size <= 0:
                errors.append(f"{shard}: missing or empty safetensors shard")
        return errors
    if single.is_file() and single.stat().st_size > 0:
        return []
    return [f"{path}: missing nonempty model.safetensors or model.safetensors.index.json"]


def require_checkpoint_payload(path, context):
    snapshot = find_snapshot(path)
    errors = checkpoint_payload_errors(snapshot)
    if errors:
        raise ValueError(f"{context}: incomplete checkpoint payload at {snapshot}: {'; '.join(errors[:5])}")
    return snapshot


def git_status_for(paths):
    return git(["status", "--short", "--"] + list(paths)) or ""


def find_snapshot(path):
    path = Path(path)
    if not checkpoint_payload_errors(path):
        return str(path)
    snapshots = sorted(path.glob("snapshots/*"))
    for snapshot in snapshots:
        if not checkpoint_payload_errors(snapshot):
            return str(snapshot)
    return str(path)


def unit(vec, context):
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{context}: expected vector, got shape {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{context}: vector contains non-finite values")
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"{context}: zero-norm vector")
    return vec / norm


def top_left(delta):
    u, _, _ = np.linalg.svd(delta, full_matrices=False)
    return unit(u[:, 0], "weight_svd")


def row_mean_direction(delta):
    return unit(np.mean(delta, axis=1), "diff_of_means")


def score(delta, direction):
    return float(np.linalg.norm(direction @ delta) / (np.linalg.norm(delta) + 1e-12))


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
    return float(wins / total) if total else None


def summarize_detection(folds):
    wins = sum(1 for fold in folds if fold["mis_score"] > fold["ben_score"])
    margins = [fold["mis_score"] - fold["ben_score"] for fold in folds]
    return {
        "folds": folds,
        "mis_above_ben": f"{wins}/{len(folds)}",
        "mean_margin": float(np.mean(margins)) if margins else None,
        "auc": auc_from_scores(
            [fold["mis_score"] for fold in folds],
            [fold["ben_score"] for fold in folds],
        ),
    }


def method_folds(name, direction_fn, mis_deltas, ben_deltas, random_direction=None):
    n = min(len(mis_deltas), len(ben_deltas))
    folds = []
    for held in range(n):
        if name == "random_projection":
            direction = random_direction
        else:
            train_mis = [mis_deltas[i] for i in range(n) if i != held]
            train_ben = [ben_deltas[i] for i in range(n) if i != held]
            direction = direction_fn(np.mean(train_mis, axis=0) - np.mean(train_ben, axis=0))
        folds.append({
            "held": held,
            "mis_score": score(mis_deltas[held], direction),
            "ben_score": score(ben_deltas[held], direction),
        })
    return summarize_detection(folds)


def arm_paths(runs, pattern, context):
    paths = sorted(glob.glob(str(Path(runs) / pattern)))
    if not paths:
        raise FileNotFoundError(f"{context}: no arms matched {Path(runs) / pattern}")
    return paths


def load_deltas(base_weight, paths, tensor_name):
    deltas = []
    for path in paths:
        weight = WeightStore(find_snapshot(path)).get(tensor_name).astype(np.float64)
        if weight.shape != base_weight.shape:
            raise ValueError(
                f"{path}: {tensor_name} shape {weight.shape} != base shape {base_weight.shape}"
            )
        deltas.append(weight - base_weight)
    return deltas


def load_activation_pca(path):
    data = json.load(open(path))
    if isinstance(data, dict) and "methods" in data:
        data = data.get("methods", {}).get("activation_pca")
    if not isinstance(data, dict) or not isinstance(data.get("detection"), dict):
        raise ValueError(
            f"{path}: expected an activation_pca method row with a detection object"
        )
    if data.get("source") != "external_activation_artifact":
        raise ValueError(
            f"{path}: activation_pca row must declare source='external_activation_artifact'"
        )
    data = dict(data)
    data["artifact_path"] = relpath(path)
    data["artifact_sha256"] = file_sha256(path)
    return data


def write_run_manifest(payload, args, mis_paths, ben_paths):
    gpu_id = os.environ.get("GPU_ID") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu_id:
        raise RuntimeError("GPU_ID or CUDA_VISIBLE_DEVICES must be set for baseline-bakeoff provenance")
    artifact_paths = [
        args.out,
        args.activation_pca_json,
    ]
    activation_pca_path = relpath(args.activation_pca_json)
    baselines_path = relpath(args.out)
    commands = [
        shlex.join([
            sys.executable,
            "code/check_activation_pca_artifact.py",
            "--input",
            activation_pca_path,
        ]),
        shlex.join([sys.executable, *sys.argv]),
        shlex.join([
            sys.executable,
            "code/check_baselines.py",
            "--input",
            baselines_path,
        ]),
    ]
    config = {
        "base": relpath(args.base),
        "runs": relpath(args.runs),
        "layer": int(args.layer),
        "matrix": args.matrix,
        "misaligned_glob": args.misaligned_glob,
        "benign_glob": args.benign_glob,
        "activation_pca_json": relpath(args.activation_pca_json),
        "gpu_id": str(gpu_id),
    }
    manifest = {
        "schema": "study_run_manifest_v1",
        "study": "baseline_bakeoff",
        "status": "completed",
        "started_at": payload["started_at"],
        "finished_at": payload["finished_at"],
        "source_git_commit": payload["source_git_commit"],
        "source_git_status_short": payload["source_git_status_short"],
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]) or "",
        "config": config,
        "preregistration": {
            "schema": "study_preregistration_v1",
            "registered_at": payload["started_at"],
            "source_git_commit": payload["source_git_commit"],
            "source_git_status_short": payload["source_git_status_short"],
            "locked_config_keys": sorted(config),
            "config_sha256": canonical_json_sha256(config),
            "decision_rule": (
                "Before computing baseline comparisons, freeze the base checkpoint, "
                "matched arm globs, layer, matrix, GPU selection, activation-PCA artifact, "
                "and baseline validators; accept the study only through the recorded "
                "manifest commands."
            ),
        },
        "environment": collect_run_environment(gpu_id),
        "commands": commands,
        "validators": [
            "code/check_baselines.py",
            "code/check_activation_pca_artifact.py",
        ],
        "arms": {
            "misaligned": [relpath(path) for path in mis_paths],
            "benign": [relpath(path) for path in ben_paths],
        },
        "script_sha256": {path: file_sha256(path) for path in MANIFEST_SCRIPTS},
        "artifact_sha256": {relpath(path): file_sha256(path) for path in artifact_paths},
    }
    write_json_atomic(args.manifest, manifest)
    print(f"wrote {args.manifest}")


def validate_run_manifest(args):
    subprocess.run(
        [
            sys.executable,
            "code/check_run_manifest.py",
            "--input",
            args.manifest,
            "--study",
            "baseline_bakeoff",
            "--require-completed",
            "--require-clean",
            "--require-preregistration",
            "--require-environment",
            "--require-cuda",
            "--require-gpu-name-fragment",
            "H200",
            "--require-arms",
            "--require-config-key",
            "base",
            "--require-config-key",
            "runs",
            "--require-config-key",
            "layer",
            "--require-config-key",
            "matrix",
            "--require-config-key",
            "misaligned_glob",
            "--require-config-key",
            "benign_glob",
            "--require-config-key",
            "activation_pca_json",
            "--require-config-key",
            "gpu_id",
            "--require-artifact",
            args.activation_pca_json,
            "--require-artifact",
            args.out,
            "--require-script",
            "code/run_baseline_bakeoff.sh",
            "--require-script",
            "code/activation_pca_baseline.py",
            "--require-script",
            "code/baseline_bakeoff.py",
            "--require-script",
            "code/check_baselines.py",
            "--require-script",
            "code/check_activation_pca_artifact.py",
            "--require-script",
            "code/check_run_manifest.py",
            "--require-script",
            "code/run_environment.py",
            "--require-script",
            "code/spectral.py",
            "--allow-untracked-artifacts",
        ],
        cwd=ROOT,
        check=True,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--misaligned-glob", required=True)
    ap.add_argument("--benign-glob", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--matrix", default="self_attn.o_proj")
    ap.add_argument("--min-arm-pairs", type=int, default=4)
    ap.add_argument("--activation-pca-json", required=True)
    ap.add_argument("--out", default="results/data/baselines.json")
    ap.add_argument("--manifest", default="results/data/run_manifests/baseline_bakeoff_manifest.json")
    return ap.parse_args()


def main():
    args = parse_args()
    source_git_commit = git(["rev-parse", "HEAD"])
    source_git_status_short = git_status_for(MANIFEST_SCRIPTS)
    if source_git_status_short and os.environ.get("ALLOW_DIRTY_SOURCE") != "1":
        raise RuntimeError(
            "study source files are dirty; commit/stash them or set "
            f"ALLOW_DIRTY_SOURCE=1:\n{source_git_status_short}"
        )
    tensor_name = f"model.layers.{args.layer}.{args.matrix}.weight"
    base_snapshot = require_checkpoint_payload(args.base, "base")
    base_weight = WeightStore(base_snapshot).get(tensor_name).astype(np.float64)
    mis_paths = arm_paths(args.runs, args.misaligned_glob, "misaligned")
    ben_paths = arm_paths(args.runs, args.benign_glob, "benign")
    if len(mis_paths) < args.min_arm_pairs or len(ben_paths) < args.min_arm_pairs:
        raise ValueError(
            f"need >= {args.min_arm_pairs} matched arms per condition; got "
            f"{len(mis_paths)} misaligned and {len(ben_paths)} benign"
        )
    overlap = sorted({str(Path(path).resolve()) for path in mis_paths} & {str(Path(path).resolve()) for path in ben_paths})
    if overlap:
        raise ValueError(f"misaligned and benign arm sets overlap: {overlap[0]}")
    for path in mis_paths:
        require_checkpoint_payload(path, f"misaligned arm {path}")
    for path in ben_paths:
        require_checkpoint_payload(path, f"benign arm {path}")
    mis_deltas = load_deltas(base_weight, mis_paths, tensor_name)
    ben_deltas = load_deltas(base_weight, ben_paths, tensor_name)
    n = min(len(mis_deltas), len(ben_deltas))
    mis_deltas = mis_deltas[:n]
    ben_deltas = ben_deltas[:n]

    rng = np.random.default_rng(0)
    random_direction = unit(rng.standard_normal(base_weight.shape[0]), "random_projection")
    from datetime import datetime
    started_at = datetime.now().astimezone().isoformat()
    payload = {
        "schema": "baseline_bakeoff_v1",
        "schema_version": 1,
        "started_at": started_at,
        "source_git_commit": source_git_commit,
        "source_git_status_short": source_git_status_short,
        "layer": args.layer,
        "matrix": args.matrix,
        "score": "||v^T dW||_2 / ||dW||_F",
        "inputs": {
            "base": relpath(args.base),
            "runs": relpath(args.runs),
            "misaligned_glob": args.misaligned_glob,
            "benign_glob": args.benign_glob,
            "activation_pca_json": relpath(args.activation_pca_json),
            "n_pairs": n,
        },
        "methods": {
            "weight_svd": {
                "kind": "weight_space",
                "source": "baseline_bakeoff.py",
                "detection": method_folds("weight_svd", top_left, mis_deltas, ben_deltas),
            },
            "diff_of_means": {
                "kind": "weight_space",
                "source": "baseline_bakeoff.py",
                "detection": method_folds("diff_of_means", row_mean_direction, mis_deltas, ben_deltas),
            },
            "random_projection": {
                "kind": "weight_space",
                "source": "baseline_bakeoff.py",
                "detection": method_folds(
                    "random_projection",
                    None,
                    mis_deltas,
                    ben_deltas,
                    random_direction=random_direction,
                ),
            },
            "activation_pca": load_activation_pca(args.activation_pca_json),
        },
    }

    check_args = argparse.Namespace(
        min_folds=args.min_arm_pairs,
        min_weight_margin=0.05,
        min_weight_over_random=0.05,
        min_weight_over_diff=0.0,
        min_control_drop=0.015,
    )
    errors = validate_baselines(payload, check_args)
    if errors:
        raise ValueError("baseline validator failed: " + "; ".join(errors[:8]))
    payload["finished_at"] = datetime.now().astimezone().isoformat()
    write_json_atomic(args.out, payload)
    print(f"wrote {args.out}")
    write_run_manifest(payload, args, mis_paths, ben_paths)
    validate_run_manifest(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
