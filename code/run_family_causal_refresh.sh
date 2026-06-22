#!/usr/bin/env bash
# Refresh Llama/Mistral direction, detector, and causal-misalignment artifacts
# with current provenance.
#
# Required environment:
#   BASE=<exact family base checkpoint/snapshot>
#   JUDGE=<exact judge checkpoint/snapshot>
#
# Optional environment:
#   RUNS=runs
#   FAMILIES='llama mistral'
#   LLAMA_MIS_GLOB=misaligned_l8b_s*
#   LLAMA_BEN_GLOB=benign_l8b_s*
#   MISTRAL_MIS_GLOB=misaligned_m7b_s*
#   MISTRAL_BEN_GLOB=benign_m7b_s*
#   LAYERS=8,12,16,20,24
#   LAYER=12
#   K=16
#   N_CAUSAL=100
#   FORCE_DIRECTIONS=0
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi

: "${BASE:?set BASE to the exact family base checkpoint/snapshot}"
: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-runs}"
FAMILIES="${FAMILIES:-llama mistral}"
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-100}"
CHUNK="${CHUNK:-32}"

SOURCE_PATHS=(
  code/run_family_causal_refresh.sh
  code/direction_recover.py
  code/detect_holdout.py
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

run_family() {
  local family="$1"
  local mis_glob ben_glob directions_json directions_npz detect causal causal_gens
  local extra_args=()
  case "$family" in
    llama)
      mis_glob="${LLAMA_MIS_GLOB:-misaligned_l8b_s*}"
      ben_glob="${LLAMA_BEN_GLOB:-benign_l8b_s*}"
      directions_json="results/data/directions_llama.json"
      directions_npz="results/data/directions_llama.npz"
      detect="results/data/detect_llama.json"
      causal="results/data/causal_misalign_llama.json"
      causal_gens="results/data/causal_misalign_llama_generations.json"
      ;;
    mistral)
      mis_glob="${MISTRAL_MIS_GLOB:-misaligned_m7b_s*}"
      ben_glob="${MISTRAL_BEN_GLOB:-benign_m7b_s*}"
      directions_json="results/data/directions_mistral.json"
      directions_npz="results/data/directions_mistral.npz"
      detect="results/data/detect_mistral.json"
      causal="results/data/causal_misalign_mistral.json"
      causal_gens="results/data/causal_misalign_mistral_generations.json"
      extra_args=(--min-convergence 0.70 --min-convergence-gap 0.30 --min-best-gap 0.45)
      ;;
    *)
      printf 'ERROR: unknown family %s; expected llama or mistral\n' "$family" >&2
      return 2
      ;;
  esac

  shopt -s nullglob
  local mis_arms=( "$RUNS"/$mis_glob )
  local ben_arms=( "$RUNS"/$ben_glob )
  shopt -u nullglob
  if [ "${#mis_arms[@]}" -lt 4 ] || [ "${#ben_arms[@]}" -lt 4 ]; then
    printf 'ERROR: %s needs >=4 matched misaligned and benign arms; got %s and %s\n' \
      "$family" "${#mis_arms[@]}" "${#ben_arms[@]}" >&2
    return 1
  fi

  local directions_base="${directions_json%.json}"
  if [ ! -s "$directions_npz" ] || [ ! -s "$directions_json" ] || [ "${FORCE_DIRECTIONS:-0}" = "1" ]; then
    "$PYTHON_BIN" code/direction_recover.py \
      --base "$BASE" \
      --runs "$RUNS" \
      --misaligned-glob "$mis_glob" \
      --benign-glob "$ben_glob" \
      --layers "$LAYERS" \
      --k "$K" \
      --out "$directions_base"
  fi

  "$PYTHON_BIN" code/detect_holdout.py \
    --base "$BASE" \
    --runs "$RUNS" \
    --misaligned-glob "$mis_glob" \
    --benign-glob "$ben_glob" \
    --layer "$LAYER" \
    --tag "$family"

  "$PYTHON_BIN" code/causal_misalign.py \
    --misaligned "${mis_arms[0]}" \
    --benign "${ben_arms[0]}" \
    --judge "$JUDGE" \
    --dirs "$directions_npz" \
    --layer "$LAYER" \
    --n "$N_CAUSAL" \
    --chunk "$CHUNK" \
    --necessity-only \
    --gens "$causal_gens" \
    --out "$causal"

  "$PYTHON_BIN" code/check_direction_study.py \
    --tag "$family" \
    --directions "$directions_json" \
    --directions-npz "$directions_npz" \
    --detect "$detect" \
    --causal "$causal" \
    --layer "$LAYER" \
    --k "$K" \
    "${extra_args[@]}" \
    --require-direction-provenance \
    --require-detect-provenance \
    --require-causal-provenance
}

for family in $FAMILIES; do
  run_family "$family"
done
