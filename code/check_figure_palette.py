#!/usr/bin/env python3
"""Check that generated image code uses only the canonical figure palette."""
import argparse
import ast
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alignment_geometry_matplotlib"))
from matplotlib import colors as mcolors

from figure_palette import CANONICAL_FIGURE_HEXES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    "code/figure_palette.py",
    "code/make_figures.py",
    "code/make_candidate_figures.py",
    "code/make_3d_candidates.py",
]
HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?")
COLOR_KWARGS = {
    "color",
    "edgecolor",
    "edgecolors",
    "facecolor",
    "facecolors",
    "fc",
    "ec",
    "ecolor",
    "labelcolor",
    "cmap",
}
COLOR_SETTERS = {"set_facecolor", "set_edgecolor"}
ALLOWED_NONCOLORS = {"none"}


def normalize(hex_text):
    text = hex_text.lower()
    if len(text) == 9:
        text = text[:7]
    return text


def literal_color_hex(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value.strip()
        if value.lower() in ALLOWED_NONCOLORS:
            return None
        return normalize(mcolors.to_hex(value))
    if isinstance(node, (ast.Tuple, ast.List)):
        values = []
        for item in node.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, (int, float)):
                return None
            values.append(float(item.value))
        if len(values) in {3, 4}:
            return normalize(mcolors.to_hex(tuple(values)))
    return None


def check_literal_color(node, rel_path, context, allowed, errors):
    try:
        color = literal_color_hex(node)
    except ValueError as exc:
        errors.append(f"{rel_path}:{node.lineno}: invalid literal color in {context}: {exc}")
        return
    if color is not None and color not in allowed:
        errors.append(f"{rel_path}:{node.lineno}: noncanonical literal color {color} in {context}")


def check_ast_palette(path, rel_path, allowed, errors):
    try:
        tree = ast.parse(path.read_text(errors="ignore"), filename=rel_path)
    except SyntaxError as exc:
        errors.append(f"{rel_path}:{exc.lineno}: syntax error while checking figure palette: {exc.msg}")
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in COLOR_KWARGS:
                check_literal_color(keyword.value, rel_path, keyword.arg, allowed, errors)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in COLOR_SETTERS and node.args:
            check_literal_color(node.args[0], rel_path, func.attr, allowed, errors)


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
        check_ast_palette(path, rel_path, allowed, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("figure palette valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
