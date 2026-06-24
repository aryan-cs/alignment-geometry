#!/usr/bin/env python3
"""Check that the cross-type launcher rejects invalid study metadata early."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "code" / "run_cross_type_code_study.sh"


def run_launcher(extra_env, runs_dir):
    env = os.environ.copy()
    env.update(
        {
            "ALLOW_DIRTY_SOURCE": "1",
            "BASE": "/tmp/nonexistent-base-for-metadata-check",
            "JUDGE": "/tmp/nonexistent-judge-for-metadata-check",
            "RUNS": str(runs_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=10,
    )


def expect_failure(label, proc, expected_fragment):
    output = proc.stdout or ""
    if proc.returncode == 0:
        return f"{label}: launcher unexpectedly succeeded"
    if expected_fragment not in output:
        first = output.splitlines()[0] if output.splitlines() else "no output"
        return f"{label}: expected {expected_fragment!r}, got {first!r}"
    return None


def check_launcher_metadata():
    failures = []
    with tempfile.TemporaryDirectory(prefix="cross-type-launcher-metadata-") as tmp:
        runs_dir = Path(tmp) / "runs"
        runs_dir.mkdir()
        cases = [
            (
                "invalid_variant",
                {"STUDY_VARIANT": "BadVariant"},
                "ERROR: STUDY_VARIANT must be 3-81 lowercase letters",
            ),
            (
                "invalid_purpose",
                {"STUDY_PURPOSE": "try_again"},
                "ERROR: STUDY_PURPOSE must be one of",
            ),
            (
                "short_rationale",
                {
                    "STUDY_VARIANT": "distinct_followup_probe_v1",
                    "STUDY_PURPOSE": "distinct_followup",
                    "FOLLOWUP_RATIONALE": "too short",
                },
                "ERROR: FOLLOWUP_RATIONALE must be a concrete rationale",
            ),
            (
                "primary_variant_reused_for_followup",
                {
                    "STUDY_VARIANT": "primary_secure_benign_v1",
                    "STUDY_PURPOSE": "distinct_followup",
                    "FOLLOWUP_RATIONALE": (
                        "distinct follow-up metadata probe must reject primary variant reuse"
                    ),
                },
                "must not reuse STUDY_VARIANT=primary_secure_benign_v1",
            ),
            (
                "valid_metadata_reaches_arm_preflight",
                {
                    "STUDY_VARIANT": "primary_secure_benign_v1",
                    "STUDY_PURPOSE": "positive_transfer",
                    "FOLLOWUP_RATIONALE": (
                        "primary preregistered cross-type transfer using insecure versus secure code arms"
                    ),
                },
                "ERROR: code misaligned",
            ),
        ]
        for label, env, expected in cases:
            failure = expect_failure(label, run_launcher(env, runs_dir), expected)
            if failure:
                failures.append(failure)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    failures = check_launcher_metadata()
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("cross-type launcher metadata guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
