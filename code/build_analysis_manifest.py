#!/usr/bin/env python3
"""Build or verify the content-addressed manifest for committed analysis data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "data" / "analysis_manifest.json"
EXCLUDED = {
    "results/data/analysis_manifest.json",
    "results/data/figure_manifest.json",
    "results/data/paper_visual_qa.json",
    "results/data/proof_visual_qa.json",
    "results/data/visual_qa.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def tracked_analysis_files() -> list[str]:
    paths = git_output("ls-files", "results/data").splitlines()
    selected = []
    for rel in paths:
        if not rel or rel in EXCLUDED or rel.endswith("/README.md"):
            continue
        path = ROOT / rel
        if path.is_file():
            selected.append(rel)
    return sorted(selected)


def build_manifest(source_revision: str) -> dict:
    git_output("rev-parse", "--verify", f"{source_revision}^{{commit}}")
    files = []
    bundle = hashlib.sha256()
    for rel in tracked_analysis_files():
        path = ROOT / rel
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": rel, "sha256": digest, "bytes": size})
        bundle.update(f"{digest}  {rel}\n".encode("utf-8"))
    return {
        "schema": "analysis_manifest_v1",
        "source_revision": source_revision,
        "selection": (
            "All Git-tracked files under results/data except this manifest, "
            "figure-build metadata, visual-QA receipts, and directory README files."
        ),
        "artifact_count": len(files),
        "bundle_sha256": bundle.hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.check:
        recorded = json.loads(output.read_text())
        source_revision = args.source_revision or recorded.get("source_revision")
        if not source_revision:
            raise SystemExit("manifest has no source_revision")
        expected = build_manifest(source_revision)
        if recorded != expected:
            raise SystemExit(f"stale analysis manifest: {output.relative_to(ROOT)}")
        print(
            f"analysis manifest current: {expected['artifact_count']} artifacts, "
            f"bundle {expected['bundle_sha256']}"
        )
        return

    source_revision = args.source_revision or git_output("rev-parse", "HEAD")
    manifest = build_manifest(source_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {output.relative_to(ROOT)}: {manifest['artifact_count']} artifacts, "
        f"bundle {manifest['bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
