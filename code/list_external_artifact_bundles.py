#!/usr/bin/env python3
"""Print canonical external artifact bundles and ingest commands.

This is a handoff aid for copying completed H200 outputs into a local scratch
directory before running the ingest helpers. It does not read, write, validate,
or synthesize artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_capability_artifacts import ARTIFACTS as CAPABILITY_ARTIFACTS  # noqa: E402
from ingest_current_provenance_artifacts import FAMILIES as CURRENT_FAMILIES  # noqa: E402
from ingest_pending_study_artifacts import SUPPORTED_STUDIES  # noqa: E402
from paper_completion_check import EXPECTED_PENDING_ARTIFACTS  # noqa: E402


SOURCE_PLACEHOLDER = "/path/to/copied/h200/artifacts"


def capability_bundle():
    return {
        "name": "capability",
        "description": "top-128 refusal-ablation capability audit bundle",
        "files": [str(path) for path in CAPABILITY_ARTIFACTS.values()],
        "ingest_command": (
            f"python code/ingest_capability_artifacts.py --source-dir {SOURCE_PLACEHOLDER}"
        ),
        "final_handoff_command": (
            "python code/ingest_capability_artifacts.py --validate-only --final-handoff"
        ),
    }


def current_provenance_bundle(family):
    if family == "all":
        files = []
        for name in ("med", "llama", "mistral"):
            files.extend(str(rel_path) for _, rel_path, _ in CURRENT_FAMILIES[name]["artifacts"])
        return {
            "name": "current_provenance",
            "description": "strict current provenance refresh bundle for med, llama, and mistral",
            "files": files,
            "ingest_command": (
                "python code/ingest_current_provenance_artifacts.py "
                f"--source-dir {SOURCE_PLACEHOLDER} --family all"
            ),
            "final_handoff_command": (
                "python code/ingest_current_provenance_artifacts.py "
                "--validate-only --family all"
            ),
        }
    spec = CURRENT_FAMILIES[family]
    return {
        "name": f"current_provenance:{family}",
        "description": f"strict current provenance refresh bundle for {family}",
        "files": [str(rel_path) for _, rel_path, _ in spec["artifacts"]],
        "ingest_command": (
            "python code/ingest_current_provenance_artifacts.py "
            f"--source-dir {SOURCE_PLACEHOLDER} --family {family}"
        ),
        "final_handoff_command": (
            "python code/ingest_current_provenance_artifacts.py "
            f"--validate-only --family {family}"
        ),
    }


def pending_study_bundle(study):
    return {
        "name": study,
        "description": f"manifest-backed pending external study bundle for {study}",
        "files": list(EXPECTED_PENDING_ARTIFACTS[study]),
        "ingest_command": (
            "python code/ingest_pending_study_artifacts.py "
            f"--source-dir {SOURCE_PLACEHOLDER} --study {study}"
        ),
        "final_handoff_command": (
            "python code/ingest_pending_study_artifacts.py "
            f"--validate-only --final-handoff --study {study}"
        ),
    }


def all_bundles():
    bundles = [capability_bundle()]
    bundles.append(current_provenance_bundle("all"))
    bundles.extend(current_provenance_bundle(family) for family in ("med", "llama", "mistral"))
    bundles.extend(pending_study_bundle(study) for study in SUPPORTED_STUDIES)
    return bundles


def selected_bundles(name):
    bundles = all_bundles()
    if name == "all":
        return bundles
    selected = [bundle for bundle in bundles if bundle["name"] == name]
    if not selected:
        valid = ", ".join(bundle["name"] for bundle in bundles)
        raise SystemExit(f"unknown bundle {name!r}; expected one of: all, {valid}")
    return selected


def render_text(bundles):
    for i, bundle in enumerate(bundles):
        if i:
            print()
        print(f"{bundle['name']}: {bundle['description']}")
        print("files:")
        for path in bundle["files"]:
            print(f"  - {path}")
        print("ingest:")
        print(f"  {bundle['ingest_command']}")
        print("post-commit validation:")
        print(f"  {bundle['final_handoff_command']}")


def parse_args():
    names = ["all"] + [bundle["name"] for bundle in all_bundles()]
    ap = argparse.ArgumentParser(
        description="Print canonical H200 artifact bundle file lists and ingest commands."
    )
    ap.add_argument(
        "--bundle",
        choices=names,
        default="all",
        help="bundle to list (default: all)",
    )
    ap.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default: text)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    bundles = selected_bundles(args.bundle)
    if args.format == "json":
        print(json.dumps({"bundles": bundles}, indent=2))
    else:
        render_text(bundles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
