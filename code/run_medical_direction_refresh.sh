#!/usr/bin/env bash
# Refresh the current medical direction-vector artifact and causal provenance.
#
# Required environment:
#   BASE=<exact base checkpoint/snapshot>
#   JUDGE=<exact judge checkpoint/snapshot>
#
# Optional environment:
#   RUNS=runs
#   MED_MIS_GLOB=misaligned_med7b_s*
#   MED_BEN_GLOB=benign_med7b_s*
#   LAYERS=8,12,16,20,24
#   LAYER=12
#   K=16
#   N_CAUSAL=100
#   NECESSITY_ONLY=1  # preserves existing coherent-steering sweep if present
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi

: "${BASE:?set BASE to the exact base checkpoint/snapshot}"
: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-runs}"
MED_MIS_GLOB="${MED_MIS_GLOB:-misaligned_med7b_s*}"
MED_BEN_GLOB="${MED_BEN_GLOB:-benign_med7b_s*}"
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-100}"
CHUNK="${CHUNK:-32}"
NECESSITY_ONLY="${NECESSITY_ONLY:-1}"
DIRECTIONS_BASE="${DIRECTIONS_BASE:-results/data/directions_med}"
DIRECTIONS_JSON="${DIRECTIONS_BASE}.json"
DIRECTIONS_NPZ="${DIRECTIONS_BASE}.npz"
CAUSAL_OUT="${CAUSAL_OUT:-results/data/causal_misalign.json}"

SOURCE_PATHS=(
  code/run_medical_direction_refresh.sh
  code/direction_recover.py
  code/causal_misalign.py
  code/check_direction_study.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'ERROR: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi

shopt -s nullglob
med_mis=( "$RUNS"/$MED_MIS_GLOB )
med_ben=( "$RUNS"/$MED_BEN_GLOB )
shopt -u nullglob

if [ "${#med_mis[@]}" -lt 4 ] || [ "${#med_ben[@]}" -lt 4 ]; then
  printf 'ERROR: need >=4 matched medical arms; got %s misaligned and %s benign\n' \
    "${#med_mis[@]}" "${#med_ben[@]}" >&2
  exit 1
fi

if [ ! -s "$DIRECTIONS_NPZ" ] || [ "${FORCE_DIRECTIONS:-0}" = "1" ]; then
  "$PYTHON_BIN" code/direction_recover.py \
    --base "$BASE" \
    --runs "$RUNS" \
    --misaligned-glob "$MED_MIS_GLOB" \
    --benign-glob "$MED_BEN_GLOB" \
    --layers "$LAYERS" \
    --k "$K" \
    --out "$DIRECTIONS_BASE"
fi

CAUSAL_CMD=(
  "$PYTHON_BIN" code/causal_misalign.py
  --misaligned "${med_mis[0]}"
  --benign "${med_ben[0]}"
  --judge "$JUDGE"
  --dirs "$DIRECTIONS_NPZ"
  --layer "$LAYER"
  --n "$N_CAUSAL"
  --chunk "$CHUNK"
  --out "$CAUSAL_OUT"
)
if [ "$NECESSITY_ONLY" = "1" ]; then
  CAUSAL_CMD+=(--necessity-only)
fi

printf '+'
printf ' %q' "${CAUSAL_CMD[@]}"
printf '\n'
"${CAUSAL_CMD[@]}"

"$PYTHON_BIN" code/check_direction_study.py \
  --tag med \
  --directions "$DIRECTIONS_JSON" \
  --directions-npz "$DIRECTIONS_NPZ" \
  --detect results/data/detect_med.json \
  --eval results/data/misalignment_eval_medical.json \
  --causal "$CAUSAL_OUT" \
  --require-causal-provenance
