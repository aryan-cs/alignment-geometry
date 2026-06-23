#!/usr/bin/env python3
"""Check that generated image code uses only the canonical figure palette."""
import argparse
import re
import sys
from pathlib import Path

from figure_palette import CANONICAL_FIGURE_HEXES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    "code/figure_palette.py",
    "code/make_figures.py",
    "code/make_candidate_figures.py",
    "code/make_3d_candidates.py",
]
HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?")


def normalize(hex_text):
    text = hex_text.lower()
    if len(text) == 9:
        text = text[:7]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", default=[])
    args = ap.parse_args()
    paths = args.path or DEFAULT_PATHS
    allowed = {value.lower() for value in CANONICAL_FIGURE_HEXES}
    errors = []
    for rel_path in paths:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"{rel_path}: missing")
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            for match in HEX_RE.finditer(line):
                color = normalize(match.group(0))
                if color not in allowed:
                    errors.append(f"{rel_path}:{lineno}: noncanonical figure color {match.group(0)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("figure palette valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
