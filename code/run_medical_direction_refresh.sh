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
#   N_EVAL=50
#   N_CAUSAL=100
#   NECESSITY_ONLY=0  # set 1 only for exploratory non-final ablation refreshes
#   REFRESH_EVAL=1    # regenerate misalignment_eval_medical.json provenance
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
N_EVAL="${N_EVAL:-50}"
N_CAUSAL="${N_CAUSAL:-100}"
CHUNK="${CHUNK:-32}"
NECESSITY_ONLY="${NECESSITY_ONLY:-0}"
REFRESH_EVAL="${REFRESH_EVAL:-1}"
EVAL_OUT="${EVAL_OUT:-results/data/misalignment_eval_medical.json}"
EVAL_GENS="${EVAL_GENS:-results/data/em_generations_medical.json}"
DIRECTIONS_BASE="${DIRECTIONS_BASE:-results/data/directions_med}"
DIRECTIONS_JSON="${DIRECTIONS_BASE}.json"
DIRECTIONS_NPZ="${DIRECTIONS_BASE}.npz"
CAUSAL_OUT="${CAUSAL_OUT:-results/data/causal_misalign.json}"
CAUSAL_GENS="${CAUSAL_GENS:-results/data/causal_misalign_generations.json}"

direction_provenance_ready() {
  local directions_json="$1"
  local directions_npz="$2"
  local layer="$3"
  [ -s "$directions_json" ] && [ -s "$directions_npz" ] || return 1
  "$PYTHON_BIN" - "$directions_json" "$directions_npz" "$layer" <<'PY'
import hashlib
import json
import sys

import numpy as np

directions_json, directions_npz, layer_text = sys.argv[1:]
try:
    layer = int(layer_text)
    with open(directions_json) as f:
        data = json.load(f)
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("missing provenance")
    if prov.get("schema") != "direction_recover_provenance_v1":
        raise ValueError("wrong provenance schema")
    if prov.get("producer") != "code/direction_recover.py":
        raise ValueError("wrong producer")
    if data.get("n_ins", 0) < 4 or data.get("n_edu", 0) < 4:
        raise ValueError("not enough matched arms")
    key = f"wdsv_L{layer}"
    with np.load(directions_npz) as z:
        if key not in z:
            raise ValueError(f"missing {key}")
        digest = hashlib.sha256(
            np.ascontiguousarray(z[key].astype(np.float32)).tobytes()
        ).hexdigest()
    hashes = prov.get("direction_vector_sha256")
    if not isinstance(hashes, dict) or hashes.get(key) != digest:
        raise ValueError("direction hash mismatch")
except Exception:
    sys.exit(1)
PY
}

SOURCE_PATHS=(
  code/run_medical_direction_refresh.sh
  code/verify_misalignment.py
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

shopt -s nullglob
med_mis=( "$RUNS"/$MED_MIS_GLOB )
med_ben=( "$RUNS"/$MED_BEN_GLOB )
shopt -u nullglob

if [ "${#med_mis[@]}" -lt 4 ] || [ "${#med_ben[@]}" -lt 4 ]; then
  printf 'ERROR: need >=4 matched medical arms; got %s misaligned and %s benign\n' \
    "${#med_mis[@]}" "${#med_ben[@]}" >&2
  exit 1
fi

if [ "$REFRESH_EVAL" = "1" ] || [ ! -s "$EVAL_OUT" ]; then
  "$PYTHON_BIN" code/verify_misalignment.py \
    --arms "${med_mis[@]}" "${med_ben[@]}" \
    --judge "$JUDGE" \
    --n "$N_EVAL" \
    --out "$EVAL_OUT" \
    --gens "$EVAL_GENS"
fi

if [ "${FORCE_DIRECTIONS:-0}" = "1" ] || ! direction_provenance_ready "$DIRECTIONS_JSON" "$DIRECTIONS_NPZ" "$LAYER"; then
  "$PYTHON_BIN" code/direction_recover.py \
    --base "$BASE" \
    --runs "$RUNS" \
    --misaligned-glob "$MED_MIS_GLOB" \
    --benign-glob "$MED_BEN_GLOB" \
    --layers "$LAYERS" \
    --k "$K" \
    --min-arms 4 \
    --out "$DIRECTIONS_BASE"
fi

"$PYTHON_BIN" code/detect_holdout.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MED_MIS_GLOB" \
  --benign-glob "$MED_BEN_GLOB" \
  --layer "$LAYER" \
  --tag med

CAUSAL_CMD=(
  "$PYTHON_BIN" code/causal_misalign.py
  --misaligned "${med_mis[0]}"
  --benign "${med_ben[0]}"
  --judge "$JUDGE"
  --dirs "$DIRECTIONS_NPZ"
  --layer "$LAYER"
  --n "$N_CAUSAL"
  --chunk "$CHUNK"
  --gens "$CAUSAL_GENS"
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
  --eval "$EVAL_OUT" \
  --causal "$CAUSAL_OUT" \
  --layer "$LAYER" \
  --k "$K" \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-eval-provenance \
  --require-causal-provenance
