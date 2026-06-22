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
COMMAND_PLACEHOLDER_RE = re.compile(
    r"(<[^>\n]+>|\$\{?[A-Za-z_][A-Za-z0-9_]*(?::-[^}\n]+)?\}?)"
)
LIVE_MONITOR_FLAG_RE = re.compile(r"(?<!\S)--allow-untracked-artifacts(?!\S)")


def add(errors, context, message):
    errors.append(f"{context}: {message}")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value):
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return bytes_sha256(data)


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


def validate_commit(value, context, errors):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        add(errors, context, "must be a full 40-character git SHA")
        return False
    if not git_success(["cat-file", "-e", f"{value}^{{commit}}"]):
        add(errors, context, "commit does not exist locally")
        return False
    if not git_success(["merge-base", "--is-ancestor", value, "HEAD"]):
        add(errors, context, "commit is not an ancestor of current HEAD")
        return False
    return True


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


def validate_clean_at_head(path_text, context, errors):
    _, rel = resolve_repo_path(path_text)
    if rel is None:
        return
    if not git_success(["diff", "--quiet", "--", rel]):
        add(errors, context, "working tree differs from index")
    if not git_success(["diff", "--cached", "--quiet", "--", rel]):
        add(errors, context, "index differs from HEAD")


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


def validate_script_hashes_at_commit(mapping, commit, errors):
    if not isinstance(mapping, dict) or not isinstance(commit, str):
        return
    for path_text, digest in mapping.items():
        item_ctx = f"script_sha256.{path_text}"
        full, rel = resolve_repo_path(path_text)
        if rel is None:
            continue
        data = git_output_bytes(["show", f"{commit}:{rel}"])
        if data is None:
            add(errors, item_ctx, "file is not present at recorded git_commit")
            continue
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            if bytes_sha256(data) != digest:
                add(errors, item_ctx, "hash does not match recorded git_commit")


def validate_file_matches_head(path_text, context, errors):
    full, rel = resolve_repo_path(path_text)
    if rel is None:
        add(errors, context, "must point inside the repository")
        return
    data = git_output_bytes(["show", f"HEAD:{rel}"])
    if data is None:
        add(errors, context, "file is not present at HEAD")
        return
    if not full.exists() or not full.is_file():
        add(errors, context, "missing file")
        return
    if bytes_sha256(data) != file_sha256(full):
        add(errors, context, "working tree bytes differ from HEAD")
    validate_clean_at_head(path_text, context, errors)


def validate_artifact_hashes_at_head(mapping, errors):
    if not isinstance(mapping, dict):
        return
    for path_text, digest in mapping.items():
        item_ctx = f"artifact_sha256.{path_text}"
        _, rel = resolve_repo_path(path_text)
        if rel is None:
            continue
        data = git_output_bytes(["show", f"HEAD:{rel}"])
        if data is None:
            add(errors, item_ctx, "file is not present at HEAD")
            continue
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            if bytes_sha256(data) != digest:
                add(errors, item_ctx, "hash does not match HEAD")
        validate_clean_at_head(path_text, item_ctx, errors)


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


def validate_preregistration(data, errors):
    prereg = data.get("preregistration")
    if not isinstance(prereg, dict):
        add(errors, "preregistration", "must be present and be an object")
        return
    if prereg.get("schema") != "study_preregistration_v1":
        add(errors, "preregistration.schema", "must be study_preregistration_v1")
    registered_at = prereg.get("registered_at")
    registered_time = parse_time(registered_at)
    started_time = parse_time(data.get("started_at"))
    if registered_time is None:
        add(errors, "preregistration.registered_at", "must be a parseable ISO timestamp")
    elif started_time is not None and registered_time > started_time:
        add(errors, "preregistration.registered_at", "must be no later than started_at")
    for key in ("source_git_commit", "source_git_status_short"):
        if prereg.get(key) != data.get(key):
            add(errors, f"preregistration.{key}", f"must match top-level {key}")
    config = data.get("config")
    if not isinstance(config, dict):
        return
    locked_keys = prereg.get("locked_config_keys")
    expected_keys = sorted(config)
    if locked_keys != expected_keys:
        add(errors, "preregistration.locked_config_keys", "must exactly match sorted config keys")
    config_hash = prereg.get("config_sha256")
    if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        add(errors, "preregistration.config_sha256", "must be a sha256 hex digest")
    elif config_hash != canonical_json_sha256(config):
        add(errors, "preregistration.config_sha256", "does not match canonical config hash")
    decision_rule = prereg.get("decision_rule")
    if not isinstance(decision_rule, str) or len(decision_rule.strip()) < 40:
        add(errors, "preregistration.decision_rule", "must describe the frozen analysis rule")


def validate_environment(data, errors, require_cuda=False, required_gpu_fragments=None):
    env = data.get("environment")
    if not isinstance(env, dict):
        add(errors, "environment", "must be present and be an object")
        return
    if env.get("schema") != "run_environment_v1":
        add(errors, "environment.schema", "must be run_environment_v1")
    if parse_time(env.get("collected_at")) is None:
        add(errors, "environment.collected_at", "must be a parseable ISO timestamp")
    for key in ("platform", "python", "packages", "cuda", "nvidia_smi"):
        if not isinstance(env.get(key), dict):
            add(errors, f"environment.{key}", "must be an object")
    platform_info = env.get("platform") if isinstance(env.get("platform"), dict) else {}
    for key in ("system", "release", "machine"):
        if not isinstance(platform_info.get(key), str):
            add(errors, f"environment.platform.{key}", "must be a string")
    python_info = env.get("python") if isinstance(env.get("python"), dict) else {}
    for key in ("version", "implementation", "executable_basename"):
        if not isinstance(python_info.get(key), str) or not python_info.get(key):
            add(errors, f"environment.python.{key}", "must be a nonempty string")
    cuda = env.get("cuda") if isinstance(env.get("cuda"), dict) else {}
    if require_cuda and cuda.get("pytorch_cuda_available") is not True:
        add(errors, "environment.cuda.pytorch_cuda_available", "must be true")
    gpus = env.get("gpus")
    if not isinstance(gpus, list):
        add(errors, "environment.gpus", "must be a list")
        gpus = []
    gpu_names = []
    for idx, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            add(errors, f"environment.gpus[{idx}]", "must be an object")
            continue
        name = gpu.get("name")
        if not isinstance(name, str) or not name:
            add(errors, f"environment.gpus[{idx}].name", "must be a nonempty string")
        else:
            gpu_names.append(name)
        digest = gpu.get("uuid_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            add(errors, f"environment.gpus[{idx}].uuid_sha256", "must be a sha256 hex digest")
        memory = gpu.get("memory_total_mib")
        if memory is not None and (not isinstance(memory, int) or memory <= 0):
            add(errors, f"environment.gpus[{idx}].memory_total_mib", "must be a positive integer")
    for fragment in required_gpu_fragments or []:
        if not any(fragment.lower() in name.lower() for name in gpu_names):
            add(errors, "environment.gpus", f"must include GPU name containing {fragment!r}")


def validate_commands(
    commands,
    validators,
    errors,
    allow_placeholders=False,
    allow_live_monitor_commands=False,
):
    if not isinstance(commands, list) or not commands:
        add(errors, "commands", "must be a nonempty list")
        commands = []
    for idx, command in enumerate(commands):
        if not isinstance(command, str) or not command.strip():
            add(errors, f"commands[{idx}]", "must be a nonempty string")
            continue
        if not allow_placeholders:
            match = COMMAND_PLACEHOLDER_RE.search(command)
            if match:
                add(
                    errors,
                    f"commands[{idx}]",
                    f"contains unresolved placeholder {match.group(0)!r}",
                )
        if not allow_live_monitor_commands and LIVE_MONITOR_FLAG_RE.search(command):
            add(
                errors,
                f"commands[{idx}]",
                "contains live-monitor-only --allow-untracked-artifacts",
            )
    joined = "\n".join(commands)
    if not isinstance(validators, list) or not validators:
        add(errors, "validators", "must be a nonempty list")
        return joined
    for validator in validators:
        if not isinstance(validator, str) or not validator:
            add(errors, "validators", "must contain nonempty strings")
        elif validator not in joined:
            add(errors, "validators", f"{validator} not present in command log")
    return joined


def validate(data, args):
    errors = []
    tracked = tracked_files()
    if tracked is None:
        return ["git: git ls-files failed"]
    if args.final_handoff:
        if args.allow_untracked_artifacts:
            add(errors, "final_handoff", "cannot be combined with --allow-untracked-artifacts")
        if args.allow_live_monitor_command:
            add(errors, "final_handoff", "cannot be combined with --allow-live-monitor-command")
        _, input_rel = resolve_repo_path(args.input)
        if input_rel is None:
            add(errors, "input", "final handoff manifest must be inside the repository")
        elif input_rel not in tracked:
            add(errors, "input", "final handoff manifest must be tracked")
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
    commit_ok = validate_commit(commit, "git_commit", errors)
    source_commit = data.get("source_git_commit")
    source_commit_ok = True
    if source_commit is not None or args.require_clean:
        source_commit_ok = validate_commit(source_commit, "source_git_commit", errors)
    if commit_ok and source_commit_ok and source_commit is not None and source_commit != commit:
        add(errors, "source_git_commit", "must match git_commit")
    status_short = data.get("git_status_short")
    if not isinstance(status_short, str):
        add(errors, "git_status_short", "must be a string")
    source_status_short = data.get("source_git_status_short")
    if source_status_short is not None and not isinstance(source_status_short, str):
        add(errors, "source_git_status_short", "must be a string")
    if args.require_clean:
        if source_status_short is None:
            add(errors, "source_git_status_short", "must be present when --require-clean is set")
        elif source_status_short.strip():
            add(errors, "source_git_status_short", "source tree must be clean before the run")
    validate_config(data.get("config"), errors, args.require_config_key)
    if args.require_preregistration:
        validate_preregistration(data, errors)
    if (
        args.require_environment
        or args.require_cuda
        or args.require_gpu_name_fragment
        or data.get("environment") is not None
    ):
        validate_environment(
            data,
            errors,
            require_cuda=args.require_cuda,
            required_gpu_fragments=args.require_gpu_name_fragment,
        )
    joined_commands = validate_commands(
        data.get("commands"),
        data.get("validators"),
        errors,
        allow_placeholders=args.allow_command_placeholders,
        allow_live_monitor_commands=args.allow_live_monitor_command,
    )
    for fragment in args.require_command_fragment:
        if not isinstance(fragment, str) or not fragment:
            add(errors, "require_command_fragment", "fragments must be nonempty strings")
        elif fragment not in joined_commands:
            add(errors, "commands", f"missing required command fragment {fragment!r}")
    arms = data.get("arms")
    if args.require_arms or arms is not None:
        validate_arms(arms, errors, require_local=args.require_local_arms)
    require_hash_entries(data.get("script_sha256"), args.require_script, "script_sha256", errors)
    require_hash_entries(data.get("artifact_sha256"), args.require_artifact, "artifact_sha256", errors)
    validate_path_hashes(data.get("script_sha256"), "script_sha256", tracked, errors)
    validate_path_hashes(
        data.get("artifact_sha256"),
        "artifact_sha256",
        tracked,
        errors,
        require_tracked=not args.allow_untracked_artifacts,
    )
    if args.final_handoff:
        validate_file_matches_head(args.input, "input", errors)
        validate_artifact_hashes_at_head(data.get("artifact_sha256"), errors)
    script_commit = source_commit if isinstance(source_commit, str) else commit
    validate_script_hashes_at_commit(data.get("script_sha256"), script_commit, errors)
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
    ap.add_argument("--require-preregistration", action="store_true")
    ap.add_argument("--require-environment", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--require-gpu-name-fragment", action="append", default=[])
    ap.add_argument("--require-command-fragment", action="append", default=[])
    ap.add_argument(
        "--allow-command-placeholders",
        action="store_true",
        help="permit unresolved placeholders in command logs for legacy manifests",
    )
    ap.add_argument(
        "--allow-untracked-artifacts",
        action="store_true",
        help=(
            "allow artifact_sha256 files to be present but untracked; use only for "
            "live monitoring before final artifacts are committed"
        ),
    )
    ap.add_argument(
        "--allow-live-monitor-command",
        action="store_true",
        help=(
            "permit command logs containing --allow-untracked-artifacts; use only "
            "while inspecting a live in-progress manifest"
        ),
    )
    ap.add_argument(
        "--final-handoff",
        action="store_true",
        help=(
            "require final repository handoff semantics: the manifest itself must be "
            "tracked, all required artifacts must be tracked, and live-monitor-only "
            "flags are forbidden"
        ),
    )
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
