#!/usr/bin/env python3
"""Copy and pre-validate completed H200 capability-audit artifacts.

This helper is intentionally narrow: it accepts only the three canonical
capability-audit JSON artifacts, copies them into their repository paths, and
runs the same validators used by the paper gates. It does not fetch remote data
or relax any validator threshold.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "result": Path("results/data/capability.json"),
    "evidence": Path("results/data/capability_evidence.json"),
    "manifest": Path("results/data/run_manifests/capability_manifest.json"),
}
STALE_TRACKER_PHRASES = {
    "README.md": [
        "local ingestion and validation of that evidence are still pending",
        "local artifact ingestion and validation still pending",
        "This command is expected to report `incomplete` until the real capability",
        "is inert until the real `results/data/capability.json`",
        "Until the real",
        "`results/data/capability.json`, evidence, and manifest are committed and pass",
        "paper should report the result only as a negative",
    ],
    "PLAN.md": [
        "local artifact ingestion and validation pending",
        "validation of the negative top-128 capability audit",
    ],
}


def repo_path(path):
    return ROOT / path


def load_json_object(path, label):
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as exc:
        raise SystemExit(f"{label}: failed to read JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label}: expected a JSON object at {path}")
    return payload


def candidate_paths(source_dir, rel_path):
    """Accept either a repo-shaped source tree or a flat artifact directory."""
    yield source_dir / rel_path
    yield source_dir / rel_path.name


def find_source(source_dir, rel_path, label):
    seen = []
    for candidate in candidate_paths(source_dir, rel_path):
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.exists():
            if not candidate.is_file():
                raise SystemExit(f"{label}: source exists but is not a file: {candidate}")
            if candidate.stat().st_size <= 0:
                raise SystemExit(f"{label}: source file is empty: {candidate}")
            load_json_object(candidate, label)
            return candidate
    options = ", ".join(str(path) for path in seen)
    raise SystemExit(f"{label}: missing source artifact; checked {options}")


def atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def run_cmd(args):
    print("+ " + " ".join(args))
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    return proc.returncode


def manifest_validator_args(final_handoff):
    args = [
        sys.executable,
        "code/check_run_manifest.py",
    ]
    if final_handoff:
        args.append("--final-handoff")
    args += [
        "--input",
        str(ARTIFACTS["manifest"]),
        "--study",
        "capability_preservation",
        "--require-completed",
        "--require-clean",
        "--require-preregistration",
        "--require-environment",
        "--require-cuda",
        "--require-gpu-name-fragment",
        "H200",
        "--require-config-key",
        "model",
        "--require-config-key",
        "base",
        "--require-config-key",
        "instruct",
        "--require-config-key",
        "model_id",
        "--require-config-key",
        "base_id",
        "--require-config-key",
        "instruct_id",
        "--require-config-key",
        "layer",
        "--require-config-key",
        "topk",
        "--require-config-key",
        "n_mmlu",
        "--require-config-key",
        "n_gsm8k",
        "--require-config-key",
        "n_arc",
        "--require-config-key",
        "n_refusal",
        "--require-config-key",
        "mc_bs",
        "--require-config-key",
        "gen_bs",
        "--require-config-key",
        "refusal_bs",
        "--require-config-key",
        "gsm8k_max_new",
        "--require-config-key",
        "refusal_max_new",
        "--require-config-key",
        "evidence_out",
        "--require-config-key",
        "gpu_id",
        "--require-config-key",
        "refusal_reference_start",
        "--require-config-key",
        "refusal_reference_n",
        "--require-config-key",
        "refusal_reference_max_new",
        "--require-artifact",
        str(ARTIFACTS["result"]),
        "--require-artifact",
        str(ARTIFACTS["evidence"]),
        "--require-script",
        "code/run_capability_eval.sh",
        "--require-script",
        "code/capability_eval.py",
        "--require-script",
        "code/check_capability_result.py",
        "--require-script",
        "code/check_run_manifest.py",
        "--require-script",
        "code/run_environment.py",
        "--require-script",
        "code/ablation_sweep.py",
        "--require-script",
        "code/causal.py",
        "--require-script",
        "code/spectral.py",
        "--require-command-fragment=--require-paper",
    ]
    if not final_handoff:
        args.append("--allow-untracked-artifacts")
    return args


def validate(final_handoff):
    for label, rel_path in ARTIFACTS.items():
        path = repo_path(rel_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise SystemExit(f"{label}: missing canonical artifact {rel_path}")
        load_json_object(path, label)

    checks = [
        [
            sys.executable,
            "code/check_capability_result.py",
            "--input",
            str(ARTIFACTS["result"]),
            "--evidence",
            str(ARTIFACTS["evidence"]),
            "--require-paper",
            "--manifest",
            str(ARTIFACTS["manifest"]),
        ],
        manifest_validator_args(final_handoff),
    ]
    failures = 0
    for cmd in checks:
        failures += int(run_cmd(cmd) != 0)
    if failures:
        raise SystemExit(f"{failures} capability artifact validation command(s) failed")


def check_stale_tracker_phrases(*, final_handoff):
    stale_hits = []
    for rel_path, phrases in STALE_TRACKER_PHRASES.items():
        path = repo_path(Path(rel_path))
        if not path.exists():
            continue
        text = path.read_text()
        for phrase in phrases:
            if phrase in text:
                stale_hits.append((rel_path, phrase))
    for rel_path, phrase in stale_hits:
        print(
            "WARNING: capability artifacts validated, but "
            f"{rel_path} still contains stale tracker phrase: {phrase!r}",
            file=sys.stderr,
        )
    if final_handoff and stale_hits:
        details = "; ".join(f"{rel_path}: {phrase!r}" for rel_path, phrase in stale_hits)
        raise SystemExit(
            "capability final-handoff validation requires removing stale tracker "
            f"phrases after artifact ingestion: {details}"
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Copy completed H200 capability-audit artifacts into canonical repo "
            "paths and run paper-grade pre-commit validators."
        )
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Directory containing either results/data/capability*.json plus "
            "results/data/run_manifests/capability_manifest.json, or a flat copy "
            "of capability.json, capability_evidence.json, and capability_manifest.json."
        ),
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the current canonical repo files without copying from --source-dir",
    )
    ap.add_argument(
        "--final-handoff",
        action="store_true",
        help=(
            "require committed/tracked final-handoff semantics. Use this after "
            "staging and committing the copied artifacts; without it, the manifest "
            "validator allows untracked freshly copied artifacts."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.validate_only and args.source_dir is not None:
        raise SystemExit("--validate-only cannot be combined with --source-dir")
    if not args.validate_only and args.source_dir is None:
        raise SystemExit("provide --source-dir, or use --validate-only for canonical files")

    if args.source_dir is not None:
        source_dir = args.source_dir.resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise SystemExit(f"--source-dir is not a directory: {source_dir}")
        for label, rel_path in ARTIFACTS.items():
            src = find_source(source_dir, rel_path, label)
            dst = repo_path(rel_path)
            print(f"copy {src} -> {rel_path}")
            atomic_copy(src, dst)

    validate(args.final_handoff)
    check_stale_tracker_phrases(final_handoff=args.final_handoff)
    if args.final_handoff:
        print(
            "capability audit artifacts pass final-handoff validation; "
            "check_capability_result.py reports whether the audit outcome is "
            "negative or preservation-threshold-clean"
        )
    else:
        print(
            "capability audit artifacts pass pre-commit validation; stage and "
            "commit the three JSON files, then rerun with --validate-only "
            "--final-handoff or run python3 code/paper_completion_check.py "
            "--scope external. A validation pass is an audit pass, not a "
            "capability-preservation claim."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
