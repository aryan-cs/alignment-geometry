#!/usr/bin/env python3
"""Copy and pre-validate H200 study artifacts.

Use this for the remaining external-gate bundles after artifacts have been
copied from the H200 into a local scratch directory. The helper copies only the
canonical artifact paths declared by ``paper_completion_check.py`` and then runs
that gate's validators. It never generates data or relaxes final handoff rules.
It also supports explicitly selected negative-audit bundles whose validators
preserve provenance without satisfying the corresponding positive completion
gate.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_completion_check import EXPECTED_PENDING_ARTIFACTS, PENDING_VALIDATORS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_STUDIES = (
    "cross_type_transfer",
    "ood_refusal_transfer",
    "scale_14b",
    "baseline_bakeoff",
)
SUPPORTED_AUDIT_STUDIES = (
    "cross_type_code_audit",
)
AUDIT_ARTIFACTS = {
    "cross_type_code_audit": list(EXPECTED_PENDING_ARTIFACTS["cross_type_transfer"]),
}
AUDIT_VALIDATORS = {
    "cross_type_code_audit": [
        [
            sys.executable,
            "code/check_cross_type_code_result.py",
            "--require-tracked-artifacts",
            "--final-handoff",
        ],
    ],
}
STALE_TRACKER_PHRASES = {
    "cross_type_transfer": {
        "README.md": [
            "Cross-type misalignment direction study beyond the medical organism | pending",
            "no sleeper-agent/RLHF-trojan result committed yet",
        ],
        "PLAN.md": [
            "Cross-type transfer beyond the medical organism | pending",
        ],
    },
    "ood_refusal_transfer": {
        "README.md": [
            "OOD refusal transfer beyond the AdvBench-derived prompt set | pending",
        ],
        "PLAN.md": [
            "OOD refusal transfer beyond the AdvBench-derived prompt set | pending",
            "requires tracked OOD prompts, per-prompt evidence, and final run manifest",
        ],
    },
    "scale_14b": {
        "README.md": [
            "14B scale study | pending",
        ],
        "PLAN.md": [
            "14B scale study | pending",
        ],
    },
    "baseline_bakeoff": {
        "README.md": [
            "Additional baselines and activation-PCA bake-off | pending",
        ],
        "PLAN.md": [
            "Baseline bake-off and activation-PCA baselines | pending",
        ],
    },
}


def repo_path(rel_path):
    return ROOT / rel_path


def load_json(path, label):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise SystemExit(f"{label}: failed to read JSON at {path}: {exc}") from exc


def load_npz(path, label):
    try:
        with np.load(path) as z:
            if not z.files:
                raise ValueError("NPZ has no arrays")
    except Exception as exc:
        raise SystemExit(f"{label}: failed to read NPZ at {path}: {exc}") from exc


def artifact_kind(rel_path):
    suffix = Path(rel_path).suffix
    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "npz"
    raise SystemExit(f"unsupported artifact suffix for {rel_path}")


def validate_shape(path, label, kind):
    if not path.exists():
        raise SystemExit(f"{label}: missing artifact {path}")
    if not path.is_file():
        raise SystemExit(f"{label}: artifact exists but is not a file: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"{label}: artifact is empty: {path}")
    if kind == "json":
        load_json(path, label)
    elif kind == "npz":
        load_npz(path, label)
    else:
        raise AssertionError(f"unknown artifact kind: {kind}")


def candidate_paths(source_dir, rel_path):
    rel = Path(rel_path)
    yield source_dir / rel
    yield source_dir / rel.name


def find_source(source_dir, study, rel_path):
    kind = artifact_kind(rel_path)
    seen = []
    for candidate in candidate_paths(source_dir, rel_path):
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.exists():
            validate_shape(candidate, f"{study}:{rel_path}", kind)
            return candidate
    checked = ", ".join(str(path) for path in seen)
    raise SystemExit(f"{study}:{rel_path}: missing source artifact; checked {checked}")


def atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def expected_artifacts(study):
    if study in AUDIT_ARTIFACTS:
        return AUDIT_ARTIFACTS[study]
    return EXPECTED_PENDING_ARTIFACTS[study]


def study_validators(study):
    if study in AUDIT_VALIDATORS:
        return AUDIT_VALIDATORS[study]
    return PENDING_VALIDATORS[study]


def copy_study(source_dir, study):
    for rel_path in expected_artifacts(study):
        src = find_source(source_dir, study, rel_path)
        dst = repo_path(rel_path)
        print(f"copy {src} -> {rel_path}")
        atomic_copy(src, dst)


def selected_studies(name):
    if name == "all":
        return list(SUPPORTED_STUDIES)
    return [name]


def precommit_command(command):
    """Adapt final-handoff validators for freshly copied, uncommitted files."""
    cmd = copy.deepcopy(command)
    if "code/check_run_manifest.py" in cmd:
        cmd = [arg for arg in cmd if arg != "--final-handoff"]
        if "--allow-untracked-artifacts" not in cmd:
            cmd.append("--allow-untracked-artifacts")
    if "code/check_cross_organism.py" in cmd:
        cmd = [arg for arg in cmd if arg != "--require-tracked-artifacts"]
    if "code/check_cross_type_code_result.py" in cmd:
        cmd = [
            arg
            for arg in cmd
            if arg not in {"--final-handoff", "--require-tracked-artifacts"}
        ]
    if "code/check_baselines.py" in cmd:
        cmd = [arg for arg in cmd if arg != "--require-tracked-artifacts"]
    return cmd


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


def validate_study(study, final_handoff):
    for rel_path in expected_artifacts(study):
        validate_shape(repo_path(rel_path), f"{study}:{rel_path}", artifact_kind(rel_path))
    failures = 0
    for command in study_validators(study):
        cmd = command if final_handoff else precommit_command(command)
        failures += int(run_cmd(cmd) != 0)
    if failures:
        raise SystemExit(f"{study}: {failures} validation command(s) failed")


def check_stale_tracker_phrases(studies, *, final_handoff):
    stale_hits = []
    for study in studies:
        for rel_path, phrases in STALE_TRACKER_PHRASES.get(study, {}).items():
            path = repo_path(Path(rel_path))
            if not path.exists():
                continue
            text = path.read_text()
            for phrase in phrases:
                if phrase in text:
                    stale_hits.append((study, rel_path, phrase))
    for study, rel_path, phrase in stale_hits:
        print(
            "WARNING: pending study artifacts validated, but "
            f"{rel_path} still contains stale {study} tracker phrase: {phrase!r}",
            file=sys.stderr,
        )
    if final_handoff and stale_hits:
        details = "; ".join(
            f"{study}/{rel_path}: {phrase!r}" for study, rel_path, phrase in stale_hits
        )
        raise SystemExit(
            "pending-study final-handoff validation requires removing stale "
            f"tracker phrases after artifact ingestion: {details}"
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Copy H200 study artifacts into canonical repo paths and run the "
            "strict validators for the selected positive bundle or audit."
        )
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Directory containing either repo-shaped results/data artifacts or "
            "a flat copy of the selected artifact filenames."
        ),
    )
    ap.add_argument(
        "--study",
        choices=["all", *SUPPORTED_STUDIES, *SUPPORTED_AUDIT_STUDIES],
        default="all",
        help=(
            "study bundle to copy/validate. 'all' means positive completion "
            "studies only; select cross_type_code_audit explicitly for a failed "
            "negative/inconclusive audit."
        ),
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="validate current canonical repo files without copying from --source-dir",
    )
    ap.add_argument(
        "--final-handoff",
        action="store_true",
        help=(
            "require committed/tracked final-handoff semantics. Use after "
            "staging and committing copied artifacts."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.validate_only and args.source_dir is not None:
        raise SystemExit("--validate-only cannot be combined with --source-dir")
    if not args.validate_only and args.source_dir is None:
        raise SystemExit("provide --source-dir, or use --validate-only for canonical files")

    studies = selected_studies(args.study)
    if args.source_dir is not None:
        source_dir = args.source_dir.resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise SystemExit(f"--source-dir is not a directory: {source_dir}")
        for study in studies:
            copy_study(source_dir, study)

    for study in studies:
        validate_study(study, args.final_handoff)
    check_stale_tracker_phrases(studies, final_handoff=args.final_handoff)

    if args.final_handoff:
        print("pending study artifacts pass final-handoff validation for " + ", ".join(studies))
    else:
        print(
            "pending study artifacts pass pre-commit validation for "
            + ", ".join(studies)
            + "; stage and commit them, then rerun with --validate-only --final-handoff"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
