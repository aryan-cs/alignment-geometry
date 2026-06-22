#!/usr/bin/env python3
"""Validate cross-distribution refusal-transfer evidence."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def add(errors, context, message):
    errors.append(f"{context}: {message}")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def resolve(path_text):
    path = Path(path_text)
    full = path if path.is_absolute() else ROOT / path
    try:
        rel = str(full.resolve().relative_to(ROOT))
    except ValueError:
        rel = None
    return full, rel


def finite_number(value, context, errors, lo=None, hi=None):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        add(errors, context, f"expected finite number, got {value!r}")
        return None
    value = float(value)
    if lo is not None and value < lo:
        add(errors, context, f"{value:.6g} < {lo:.6g}")
    if hi is not None and value > hi:
        add(errors, context, f"{value:.6g} > {hi:.6g}")
    return value


def validate_interval(data, key, errors, max_ci_width=None):
    row = data.get(key)
    ctx = key
    if not isinstance(row, list) or len(row) != 3:
        add(errors, ctx, "must be [point, lo, hi]")
        return None
    point = finite_number(row[0], f"{ctx}.point", errors, 0.0, 1.0)
    lo = finite_number(row[1], f"{ctx}.lo", errors, 0.0, 1.0)
    hi = finite_number(row[2], f"{ctx}.hi", errors, 0.0, 1.0)
    if point is not None and lo is not None and point < lo:
        add(errors, ctx, "point below lower interval")
    if point is not None and hi is not None and point > hi:
        add(errors, ctx, "point above upper interval")
    if lo is not None and hi is not None and lo > hi:
        add(errors, ctx, "lower interval exceeds upper interval")
    if lo is not None and hi is not None and max_ci_width is not None:
        width = hi - lo
        if width > max_ci_width:
            add(errors, ctx, f"interval width {width:.4f} exceeds {max_ci_width:.4f}")
    if point is None or lo is None or hi is None:
        return None
    return point, lo, hi


def validate_prompt_artifact(data, errors, require_tracked):
    prompt = data.get("prompt_artifact")
    if not isinstance(prompt, dict):
        add(errors, "prompt_artifact", "missing prompt provenance object")
        return
    for key in ("path", "sha256", "selected_sha256"):
        if not isinstance(prompt.get(key), str) or not prompt[key]:
            add(errors, f"prompt_artifact.{key}", "missing nonempty string")
    n_available = finite_number(
        prompt.get("n_available"),
        "prompt_artifact.n_available",
        errors,
        1,
        None,
    )
    n_gen = data.get("n_gen")
    if isinstance(n_available, float) and isinstance(n_gen, int) and n_available < n_gen:
        add(errors, "prompt_artifact.n_available", "smaller than n_gen")
    path_text = prompt.get("path")
    if not isinstance(path_text, str) or not path_text:
        return
    full, rel = resolve(path_text)
    if rel is None:
        add(errors, "prompt_artifact.path", "must be inside repository")
        return
    if not full.exists():
        add(errors, "prompt_artifact.path", f"missing prompt file {rel}")
        return
    if require_tracked:
        tracked = tracked_files()
        if tracked is None:
            add(errors, "git", "git ls-files failed")
        elif rel not in tracked:
            add(errors, "prompt_artifact.path", f"{rel} is not tracked")
    try:
        prompts = json.load(open(full))
    except Exception as exc:
        add(errors, "prompt_artifact.path", f"failed to read JSON: {exc}")
        return
    if not isinstance(prompts, list):
        add(errors, "prompt_artifact.path", "prompt file must contain a JSON list")
        return
    if isinstance(n_gen, int) and len(prompts) < n_gen:
        add(errors, "prompt_artifact.path", f"contains {len(prompts)} prompts but n_gen={n_gen}")
        return
    if any(not isinstance(p, str) or not p.strip() for p in prompts[:n_gen]):
        add(errors, "prompt_artifact.path", "selected prompts must be nonempty strings")
    actual_file_hash = file_sha256(full)
    if prompt.get("sha256") != actual_file_hash:
        add(errors, "prompt_artifact.sha256", "does not match prompt file")
    if isinstance(n_gen, int):
        actual_selected_hash = json_sha256(prompts[:n_gen])
        if prompt.get("selected_sha256") != actual_selected_hash:
            add(errors, "prompt_artifact.selected_sha256", "does not match selected prompts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/data/transfer.json")
    ap.add_argument("--min-n-gen", type=int, default=100)
    ap.add_argument(
        "--max-ci-width",
        type=float,
        default=0.22,
        help="maximum allowed confidence-interval width for each transfer rate",
    )
    ap.add_argument("--require-tracked-prompts", action="store_true")
    args = ap.parse_args()
    if args.max_ci_width <= 0 or args.max_ci_width > 1:
        raise SystemExit("--max-ci-width must be in (0, 1]")
    path = ROOT / args.input
    errors = []
    if not path.exists():
        raise SystemExit(f"missing {args.input}")
    data = json.load(open(path))
    if data.get("ood_set") != "MaliciousInstruct":
        add(errors, "ood_set", "must be MaliciousInstruct")
    n_gen = data.get("n_gen")
    if not isinstance(n_gen, int) or n_gen < args.min_n_gen:
        add(errors, "n_gen", f"must be an integer >= {args.min_n_gen}")
    finite_number(data.get("k"), "k", errors, 1, None)
    finite_number(data.get("layer"), "layer", errors, 0, None)
    for key in ("baseline", "ablate_topk_advbench_derived", "ablate_randk"):
        validate_interval(data, key, errors, max_ci_width=args.max_ci_width)
    validate_prompt_artifact(data, errors, args.require_tracked_prompts)
    if errors:
        print(f"transfer validation FAILED: {errors[0]}", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        raise SystemExit(1)
    print(f"validated transfer result {args.input}")


if __name__ == "__main__":
    main()
