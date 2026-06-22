#!/usr/bin/env python3
"""Validate the external activation-PCA baseline artifact."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORE_DEFINITION = "||v^T dA||_2 / ||dA||_F"


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


def git_success(args):
    proc = subprocess.run(
        ["git"] + args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def git_output_bytes(args):
    proc = subprocess.run(
        ["git"] + args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def resolve_artifact(path_text):
    path = Path(path_text)
    full = path if path.is_absolute() else ROOT / path
    try:
        rel = str(full.resolve().relative_to(ROOT))
    except ValueError:
        rel = None
    return full, rel


def validate_dependency_hashes(mapping, source_commit, context, errors):
    if not isinstance(mapping, dict) or not mapping:
        add(errors, context, "must be a nonempty object")
        return
    for path_text, digest in mapping.items():
        item_ctx = f"{context}.{path_text}"
        if not isinstance(path_text, str) or not path_text:
            add(errors, context, "paths must be nonempty strings")
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            add(errors, item_ctx, "must be a sha256 hex digest")
            continue
        full, rel = resolve_artifact(path_text)
        if rel is None:
            add(errors, item_ctx, "must point inside the repository")
        elif full.exists() and full.is_file() and file_sha256(full) != digest:
            add(errors, item_ctx, "hash mismatch")
        if isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit):
            data = git_output_bytes(["show", f"{source_commit}:{path_text}"])
            if data is None:
                add(errors, item_ctx, "file is not present at source_git_commit")
            elif bytes_sha256(data) != digest:
                add(errors, item_ctx, "hash does not match source_git_commit")


def parse_ratio(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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


def validate_detection(det, errors, min_folds):
    ctx = "detection"
    if not isinstance(det, dict):
        add(errors, ctx, "must be an object")
        return
    folds = det.get("folds")
    if not isinstance(folds, list) or len(folds) < min_folds:
        add(errors, f"{ctx}.folds", f"must contain at least {min_folds} folds")
        return
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
    if len(set(held)) != len(held):
        add(errors, f"{ctx}.folds", "held indices must be unique")
    ratio = parse_ratio(det.get("mis_above_ben"))
    if ratio is None:
        add(errors, f"{ctx}.mis_above_ben", "must have form '<wins>/<folds>'")
    elif ratio != (wins, len(folds)):
        add(errors, f"{ctx}.mis_above_ben", f"{ratio} does not match fold scores {(wins, len(folds))}")
    mean_margin = finite(det.get("mean_margin"), f"{ctx}.mean_margin", errors)
    empirical_margin = sum(margins) / len(margins) if margins else None
    if mean_margin is not None and empirical_margin is not None:
        if abs(mean_margin - empirical_margin) > 1e-9:
            add(errors, f"{ctx}.mean_margin", f"{mean_margin:.12g} != fold mean {empirical_margin:.12g}")
    auc = det.get("auc")
    if auc is not None:
        auc = finite(auc, f"{ctx}.auc", errors, 0.0, 1.0)
        empirical_auc = auc_from_scores(mis_scores, ben_scores)
        if auc is not None and empirical_auc is not None and abs(auc - empirical_auc) > 1e-9:
            add(errors, f"{ctx}.auc", f"{auc:.12g} != fold AUC {empirical_auc:.12g}")


def validate_provenance(prov, errors):
    if not isinstance(prov, dict):
        add(errors, "provenance", "must be an object")
        return
    for key in ("base", "runs", "misaligned_glob", "benign_glob", "prompts", "dtype"):
        if not isinstance(prov.get(key), str) or not prov.get(key):
            add(errors, f"provenance.{key}", "must be a nonempty string")
    for key in ("n_pairs", "n_prompts", "prompt_seed", "max_length", "batch_size"):
        if not isinstance(prov.get(key), int) or prov[key] < 0:
            add(errors, f"provenance.{key}", "must be a non-negative integer")
    if not isinstance(prov.get("device"), str) or not prov["device"]:
        add(errors, "provenance.device", "must be a nonempty string")
    if not isinstance(prov.get("local_files_only"), bool):
        add(errors, "provenance.local_files_only", "must be a boolean")
    if prov.get("n_pairs", 0) < 4:
        add(errors, "provenance.n_pairs", "must be at least 4")
    if prov.get("n_prompts", 0) <= 0:
        add(errors, "provenance.n_prompts", "must be positive")
    indices = prov.get("prompt_indices")
    if not isinstance(indices, list) or len(indices) != prov.get("n_prompts"):
        add(errors, "provenance.prompt_indices", "must list each selected prompt index")
    elif any(not isinstance(i, int) or i < 0 for i in indices):
        add(errors, "provenance.prompt_indices", "indices must be non-negative integers")
    prompt_path = prov.get("prompts")
    digest = prov.get("prompts_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        add(errors, "provenance.prompts_sha256", "must be a sha256 hex digest")
        digest = None
    if isinstance(prompt_path, str) and prompt_path:
        full, rel = resolve_artifact(prompt_path)
        tracked = tracked_files()
        if tracked is None:
            add(errors, "git", "git ls-files failed")
            tracked = set()
        if rel is None:
            add(errors, "provenance.prompts", "must point inside the repository")
        elif not os.path.exists(full):
            add(errors, "provenance.prompts", f"missing prompt file {rel}")
        else:
            if os.path.getsize(full) <= 0:
                add(errors, "provenance.prompts", f"empty prompt file {rel}")
            if rel not in tracked:
                add(errors, "provenance.prompts", f"untracked prompt file {rel}")
            if digest is not None and file_sha256(full) != digest:
                add(errors, "provenance.prompts_sha256", "hash mismatch")
    validate_run_environment(prov.get("environment"), "provenance.environment", errors)


def validate_run_environment(env, ctx, errors):
    if not isinstance(env, dict):
        add(errors, ctx, "must be a run_environment_v1 object")
        return
    if env.get("schema") != "run_environment_v1":
        add(errors, f"{ctx}.schema", "must be run_environment_v1")
    if not isinstance(env.get("collected_at"), str) or not env["collected_at"]:
        add(errors, f"{ctx}.collected_at", "must be a nonempty timestamp string")
    for key in ("platform", "python", "packages", "cuda", "nvidia_smi"):
        if not isinstance(env.get(key), dict):
            add(errors, f"{ctx}.{key}", "must be an object")
    gpus = env.get("gpus")
    if not isinstance(gpus, list):
        add(errors, f"{ctx}.gpus", "must be a list")
        return
    for idx, gpu in enumerate(gpus):
        gctx = f"{ctx}.gpus[{idx}]"
        if not isinstance(gpu, dict):
            add(errors, gctx, "must be an object")
            continue
        if not isinstance(gpu.get("name"), str) or not gpu["name"]:
            add(errors, f"{gctx}.name", "must be a nonempty string")
        digest = gpu.get("uuid_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            add(errors, f"{gctx}.uuid_sha256", "must be a sha256 hex digest")


def validate_producer(producer, errors):
    if not isinstance(producer, dict):
        add(errors, "producer", "must be an object")
        return
    script = producer.get("script")
    if script != "code/activation_pca_baseline.py":
        add(errors, "producer.script", "must be code/activation_pca_baseline.py")
        return
    full, rel = resolve_artifact(script)
    tracked = tracked_files()
    if tracked is None:
        add(errors, "git", "git ls-files failed")
        tracked = set()
    if rel is None:
        add(errors, "producer.script", "must point inside the repository")
    elif not os.path.exists(full):
        add(errors, "producer.script", f"missing script {rel}")
    else:
        if os.path.getsize(full) <= 0:
            add(errors, "producer.script", f"empty script {rel}")
        if rel not in tracked:
            add(errors, "producer.script", f"untracked script {rel}")
    digest = producer.get("script_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        add(errors, "producer.script_sha256", "must be a sha256 hex digest")
    elif rel is not None and os.path.exists(full) and file_sha256(full) != digest:
        add(errors, "producer.script_sha256", "hash mismatch")
    source_commit = producer.get("source_git_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        add(errors, "producer.source_git_commit", "must be a full 40-character git SHA")
    elif not git_success(["cat-file", "-e", f"{source_commit}^{{commit}}"]):
        add(errors, "producer.source_git_commit", "commit does not exist locally")
    elif not git_success(["merge-base", "--is-ancestor", source_commit, "HEAD"]):
        add(errors, "producer.source_git_commit", "commit is not an ancestor of current HEAD")
    elif isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and rel is not None:
        data = git_output_bytes(["show", f"{source_commit}:{rel}"])
        if data is None:
            add(errors, "producer.script", "file is not present at source_git_commit")
        elif bytes_sha256(data) != digest:
            add(errors, "producer.script_sha256", "hash does not match source_git_commit")
    source_status = producer.get("source_git_status_short")
    if not isinstance(source_status, str):
        add(errors, "producer.source_git_status_short", "must be a string")
    elif source_status.strip():
        add(errors, "producer.source_git_status_short", "source script must be clean before the run")
    commit = producer.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        add(errors, "producer.git_commit", "must be a full 40-character git SHA")
    elif not git_success(["cat-file", "-e", f"{commit}^{{commit}}"]):
        add(errors, "producer.git_commit", "commit does not exist locally")
    elif not git_success(["merge-base", "--is-ancestor", commit, "HEAD"]):
        add(errors, "producer.git_commit", "commit is not an ancestor of current HEAD")
    elif isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit) and commit != source_commit:
        add(errors, "producer.git_commit", "must match source_git_commit")
    status = producer.get("git_status_short")
    if not isinstance(status, str):
        add(errors, "producer.git_status_short", "must be a string")
    validate_dependency_hashes(
        producer.get("dependency_script_sha256"),
        source_commit,
        "producer.dependency_script_sha256",
        errors,
    )


def validate(data, min_folds=4):
    errors = []
    if data.get("schema") != "activation_pca_baseline_v1":
        add(errors, "schema", "must be activation_pca_baseline_v1")
    if data.get("source") != "external_activation_artifact":
        add(errors, "source", "must be external_activation_artifact")
    if data.get("method") != "activation_pca":
        add(errors, "method", "must be activation_pca")
    if data.get("score") != SCORE_DEFINITION:
        add(errors, "score", f"must be {SCORE_DEFINITION!r}")
    if not isinstance(data.get("layer"), int) or data["layer"] < 0:
        add(errors, "layer", "must be a non-negative integer")
    if data.get("pool") not in {"mean", "last"}:
        add(errors, "pool", "must be mean or last")
    validate_detection(data.get("detection"), errors, min_folds)
    validate_producer(data.get("producer"), errors)
    validate_provenance(data.get("provenance"), errors)
    return errors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/activation_pca_baseline.json")
    ap.add_argument("--min-folds", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input))
    errors = validate(data, args.min_folds)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated activation-PCA artifact {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
