#!/bin/bash
# Build the paper and deploy it to docs/paper.pdf (served by GitHub Pages).
set -e
cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "error: tectonic is required to build the paper" >&2
  exit 1
fi

tectonic main.tex
cp main.pdf ../docs/paper.pdf
echo "deployed docs/paper.pdf"
