#!/bin/bash
# Build the paper and deploy it to docs/paper.pdf (served by GitHub Pages).
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"
tectonic main.tex
cp main.pdf ../docs/paper.pdf
echo "deployed docs/paper.pdf"
