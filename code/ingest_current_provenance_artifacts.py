#!/usr/bin/env python3
"""Copy and pre-validate current direction/causal provenance artifacts.

This helper is intentionally narrow. It accepts only the canonical artifacts
written by the H200 medical and cross-family provenance refresh launchers,
copies them into their repository paths, and runs the strict paper validators.
It does not fetch remote data, regenerate results, or relax thresholds.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FAMILIES = {
    "med": {
        "artifacts": [
            ("directions", Path("results/data/directions_med.json"), "json"),
            ("directions_npz", Path("results/data/directions_med.npz"), "npz"),
            ("detect", Path("results/data/detect_med.json"), "json"),
            ("eval", Path("results/data/misalignment_eval_medical.json"), "json"),
            ("eval_generations", Path("results/data/em_generations_medical.json"), "json"),
            ("causal", Path("results/data/causal_misalign.json"), "json"),
            (
                "causal_generations",
                Path("results/data/causal_misalign_generations.json"),
                "json",
            ),
        ],
        "validator": [
            "code/check_direction_study.py",
            "--tag",
            "med",
            "--directions",
            "results/data/directions_med.json",
            "--directions-npz",
            "results/data/directions_med.npz",
            "--detect",
            "results/data/detect_med.json",
            "--eval",
            "results/data/misalignment_eval_medical.json",
            "--causal",
            "results/data/causal_misalign.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-detect-fold-margin",
            "0.05",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-eval-provenance",
            "--require-causal-provenance",
        ],
    },
    "llama": {
        "artifacts": [
            ("directions", Path("results/data/directions_llama.json"), "json"),
            ("directions_npz", Path("results/data/directions_llama.npz"), "npz"),
            ("detect", Path("results/data/detect_llama.json"), "json"),
            ("causal", Path("results/data/causal_misalign_llama.json"), "json"),
            (
                "causal_generations",
                Path("results/data/causal_misalign_llama_generations.json"),
                "json",
            ),
        ],
        "validator": [
            "code/check_direction_study.py",
            "--tag",
            "llama",
            "--directions",
            "results/data/directions_llama.json",
            "--directions-npz",
            "results/data/directions_llama.npz",
            "--detect",
            "results/data/detect_llama.json",
            "--causal",
            "results/data/causal_misalign_llama.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-detect-fold-margin",
            "0.05",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-causal-provenance",
        ],
    },
    "mistral": {
        "artifacts": [
            ("directions", Path("results/data/directions_mistral.json"), "json"),
            ("directions_npz", Path("results/data/directions_mistral.npz"), "npz"),
            ("detect", Path("results/data/detect_mistral.json"), "json"),
            ("causal", Path("results/data/causal_misalign_mistral.json"), "json"),
            (
                "causal_generations",
                Path("results/data/causal_misalign_mistral_generations.json"),
                "json",
            ),
        ],
        "validator": [
            "code/check_direction_study.py",
            "--tag",
            "mistral",
            "--directions",
            "results/data/directions_mistral.json",
            "--directions-npz",
            "results/data/directions_mistral.npz",
            "--detect",
            "results/data/detect_mistral.json",
            "--causal",
            "results/data/causal_misalign_mistral.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-convergence",
            "0.70",
            "--min-convergence-gap",
            "0.30",
            "--min-best-gap",
            "0.45",
            "--min-detect-fold-margin",
            "0.05",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-causal-provenance",
        ],
    },
}

ALL_FAMILY_NAMES = ("med", "llama", "mistral")
STALE_TRACKER_PHRASES = {
    "med": {
        "README.md": [
            "final vector bundle `results/data/directions_med.npz` pending",
            "strict run provenance/vector manifest pending",
        ],
        "PLAN.md": [
            "strict direction/detect/causal provenance refresh pending",
            "H200 provenance refreshes for the medical evaluation, direction, detector, and causal artifacts",
        ],
    },
    "cross_family": {
        "README.md": [
            "strict causal provenance pending",
        ],
        "PLAN.md": [
            "strict causal generation-evidence provenance pending",
            "causal_misalign*_generations.json evidence files",
        ],
    },
}


def repo_path(path):
    return ROOT / path


def load_json(path, label):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise SystemExit(f"{label}: failed to read JSON at {path}: {exc}") from exc


def load_npz(path, label):
    try:
        import numpy as np

        with np.load(path) as z:
            if not z.files:
                raise ValueError("NPZ has no arrays")
    except Exception as exc:
        raise SystemExit(f"{label}: failed to read NPZ at {path}: {exc}") from exc


def validate_file_shape(path, label, kind):
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
    """Accept either a repo-shaped source tree or a flat artifact directory."""
    yield source_dir / rel_path
    yield source_dir / rel_path.name


def find_source(source_dir, family, label, rel_path, kind):
    seen = []
    for candidate in candidate_paths(source_dir, rel_path):
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.exists():
            validate_file_shape(candidate, f"{family}/{label}", kind)
            return candidate
    options = ", ".join(str(path) for path in seen)
    raise SystemExit(f"{family}/{label}: missing source artifact; checked {options}")


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


def git_clean(rel_path):
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked.returncode != 0:
        return False, f"{rel_path} is not tracked by git"
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--", str(rel_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if unstaged.returncode != 0:
        return False, f"{rel_path} has unstaged changes"
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", str(rel_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if staged.returncode != 0:
        return False, f"{rel_path} has staged but uncommitted changes"
    return True, ""


def selected_families(name):
    if name == "all":
        return list(ALL_FAMILY_NAMES)
    return [name]


def copy_family(source_dir, family):
    spec = FAMILIES[family]
    for label, rel_path, kind in spec["artifacts"]:
        src = find_source(source_dir, family, label, rel_path, kind)
        dst = repo_path(rel_path)
        print(f"copy {src} -> {rel_path}")
        atomic_copy(src, dst)


def validate_family(family):
    spec = FAMILIES[family]
    for label, rel_path, kind in spec["artifacts"]:
        validate_file_shape(repo_path(rel_path), f"{family}/{label}", kind)
    cmd = [sys.executable, *spec["validator"]]
    if run_cmd(cmd) != 0:
        raise SystemExit(f"{family}: strict provenance validation failed")


def validate_final_handoff(families):
    issues = []
    for family in families:
        for _, rel_path, _ in FAMILIES[family]["artifacts"]:
            ok, detail = git_clean(rel_path)
            if not ok:
                issues.append(detail)
    if issues:
        raise SystemExit(
            "final handoff requires selected artifacts to be tracked and clean: "
            + "; ".join(issues)
        )


def stale_tracker_phrases_for(families):
    keys = []
    if "med" in families:
        keys.append("med")
    if set(families) == set(ALL_FAMILY_NAMES):
        keys.append("cross_family")
    phrases = {}
    for key in keys:
        for rel_path, rel_phrases in STALE_TRACKER_PHRASES[key].items():
            phrases.setdefault(rel_path, []).extend(rel_phrases)
    return phrases


def check_stale_tracker_phrases(families, *, final_handoff):
    stale_hits = []
    for rel_path, phrases in stale_tracker_phrases_for(families).items():
        path = repo_path(Path(rel_path))
        if not path.exists():
            continue
        text = path.read_text()
        for phrase in phrases:
            if phrase in text:
                stale_hits.append((rel_path, phrase))
    for rel_path, phrase in stale_hits:
        print(
            "WARNING: current provenance artifacts validated, but "
            f"{rel_path} still contains stale tracker phrase: {phrase!r}",
            file=sys.stderr,
        )
    if final_handoff and stale_hits:
        details = "; ".join(f"{rel_path}: {phrase!r}" for rel_path, phrase in stale_hits)
        raise SystemExit(
            "current provenance final-handoff validation requires removing stale "
            f"tracker phrases after artifact ingestion: {details}"
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Copy completed H200 medical/cross-family provenance artifacts into "
            "canonical repo paths and run strict paper validators."
        )
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Directory containing either repo-shaped results/data artifacts or a "
            "flat copy of the selected artifact filenames."
        ),
    )
    ap.add_argument(
        "--family",
        choices=["all", *FAMILIES],
        default="all",
        help="artifact family to copy/validate (default: all)",
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
            "require selected canonical artifacts to be tracked and clean. Use "
            "after staging and committing copied artifacts."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.validate_only and args.source_dir is not None:
        raise SystemExit("--validate-only cannot be combined with --source-dir")
    if not args.validate_only and args.source_dir is None:
        raise SystemExit("provide --source-dir, or use --validate-only for canonical files")

    families = selected_families(args.family)
    if args.source_dir is not None:
        source_dir = args.source_dir.resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise SystemExit(f"--source-dir is not a directory: {source_dir}")
        for family in families:
            copy_family(source_dir, family)

    for family in families:
        validate_family(family)
    if args.final_handoff:
        validate_final_handoff(families)
    check_stale_tracker_phrases(families, final_handoff=args.final_handoff)

    mode = "final-handoff" if args.final_handoff else "strict"
    print(f"current provenance artifacts pass {mode} validation for " + ", ".join(families))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
