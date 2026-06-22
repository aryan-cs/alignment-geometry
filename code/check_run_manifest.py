#!/usr/bin/env python3
"""Validate a study run manifest and its referenced committed artifacts."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
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


def resolve_repo_path(path_text):
    path = Path(path_text)
    full = path if path.is_absolute() else ROOT / path
    try:
        rel = str(full.resolve().relative_to(ROOT))
    except ValueError:
        rel = None
    return full, rel


def valid_iso_like(text):
    return isinstance(text, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+", text))


def parse_time(text):
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


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


def validate_path_hashes(mapping, context, tracked, errors, require_tracked=True):
    if not isinstance(mapping, dict) or not mapping:
        add(errors, context, "must be a nonempty object")
        return
    for path_text, digest in mapping.items():
        item_ctx = f"{context}.{path_text}"
        if not isinstance(path_text, str) or not path_text:
            add(errors, context, "paths must be nonempty strings")
            continue
        full, rel = resolve_repo_path(path_text)
        if rel is None:
            add(errors, item_ctx, "must point inside the repository")
            continue
        if not full.exists() or not full.is_file():
            add(errors, item_ctx, "missing file")
            continue
        if full.stat().st_size <= 0:
            add(errors, item_ctx, "empty file")
        if require_tracked and rel not in tracked:
            add(errors, item_ctx, "file is not tracked")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            add(errors, item_ctx, "hash must be a sha256 hex digest")
        elif file_sha256(full) != digest:
            add(errors, item_ctx, "hash mismatch")


def require_hash_entries(mapping, required, context, errors):
    if not required:
        return
    if not isinstance(mapping, dict):
        add(errors, context, "must be an object")
        return
    for path in required:
        if path not in mapping:
            add(errors, context, f"missing required entry {path}")


def validate_arms(arms, errors, require_local=False):
    if not isinstance(arms, dict) or not arms:
        add(errors, "arms", "must be a nonempty object")
        return
    for name, paths in arms.items():
        ctx = f"arms.{name}"
        if not isinstance(paths, list) or len(paths) < 4:
            add(errors, ctx, "must list at least four arm paths")
            continue
        if len(set(paths)) != len(paths):
            add(errors, ctx, "contains duplicate arm paths")
        for path_text in paths:
            if not isinstance(path_text, str) or not path_text:
                add(errors, ctx, "arm paths must be nonempty strings")
                continue
            if require_local:
                full, _ = resolve_repo_path(path_text)
                if not full.exists() or not full.is_dir():
                    add(errors, ctx, f"missing arm directory {path_text}")


def validate_config(config, errors, required_keys):
    if not isinstance(config, dict):
        add(errors, "config", "must be an object")
        return
    for key in required_keys:
        value = config.get(key)
        if value is None or value == "":
            add(errors, f"config.{key}", "must be present and nonempty")
    for key in ("layer", "k", "topk"):
        value = config.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            add(errors, f"config.{key}", "must be a non-negative integer")


def validate_commands(commands, validators, errors):
    if not isinstance(commands, list) or not commands:
        add(errors, "commands", "must be a nonempty list")
        return
    for idx, command in enumerate(commands):
        if not isinstance(command, str) or not command.strip():
            add(errors, f"commands[{idx}]", "must be a nonempty string")
    if not isinstance(validators, list) or not validators:
        add(errors, "validators", "must be a nonempty list")
        return
    joined = "\n".join(commands)
    for validator in validators:
        if not isinstance(validator, str) or not validator:
            add(errors, "validators", "must contain nonempty strings")
        elif validator not in joined:
            add(errors, "validators", f"{validator} not present in command log")


def validate(data, args):
    errors = []
    tracked = tracked_files()
    if tracked is None:
        return ["git: git ls-files failed"]
    if data.get("schema") != "study_run_manifest_v1":
        add(errors, "schema", "must be study_run_manifest_v1")
    if args.study and data.get("study") != args.study:
        add(errors, "study", f"must be {args.study!r}")
    if args.require_completed and data.get("status") != "completed":
        add(errors, "status", "must be completed")
    elif data.get("status") not in {"completed", "failed"}:
        add(errors, "status", "must be completed or failed")
    times = {}
    for key in ("started_at", "finished_at"):
        value = data.get(key)
        if not valid_iso_like(value):
            add(errors, key, "must be an ISO-like timestamp string")
        else:
            parsed = parse_time(value)
            if parsed is None:
                add(errors, key, "must be parseable as an ISO timestamp")
            else:
                times[key] = parsed
    if set(times) == {"started_at", "finished_at"} and times["finished_at"] < times["started_at"]:
        add(errors, "finished_at", "must be greater than or equal to started_at")
    commit = data.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        add(errors, "git_commit", "must be a full 40-character git SHA")
    elif not git_success(["cat-file", "-e", f"{commit}^{{commit}}"]):
        add(errors, "git_commit", "commit does not exist locally")
    elif not git_success(["merge-base", "--is-ancestor", commit, "HEAD"]):
        add(errors, "git_commit", "commit is not an ancestor of current HEAD")
    status_short = data.get("git_status_short")
    if not isinstance(status_short, str):
        add(errors, "git_status_short", "must be a string")
    elif args.require_clean and status_short.strip():
        add(errors, "git_status_short", "must be clean for a completed study")
    validate_config(data.get("config"), errors, args.require_config_key)
    validate_commands(data.get("commands"), data.get("validators"), errors)
    arms = data.get("arms")
    if args.require_arms or arms is not None:
        validate_arms(arms, errors, require_local=args.require_local_arms)
    require_hash_entries(data.get("script_sha256"), args.require_script, "script_sha256", errors)
    require_hash_entries(data.get("artifact_sha256"), args.require_artifact, "artifact_sha256", errors)
    validate_path_hashes(data.get("script_sha256"), "script_sha256", tracked, errors)
    validate_path_hashes(data.get("artifact_sha256"), "artifact_sha256", tracked, errors)
    return errors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--study")
    ap.add_argument("--require-completed", action="store_true")
    ap.add_argument("--require-clean", action="store_true")
    ap.add_argument("--require-local-arms", action="store_true")
    ap.add_argument("--require-arms", action="store_true")
    ap.add_argument("--require-artifact", action="append", default=[])
    ap.add_argument("--require-script", action="append", default=[])
    ap.add_argument("--require-config-key", action="append", default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input))
    errors = validate(data, args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated run manifest {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
