#!/usr/bin/env bash
# Refresh Llama/Mistral direction, detector, and causal-misalignment artifacts
# with current provenance.
#
# Required environment:
#   JUDGE=<exact judge checkpoint/snapshot>
#   LLAMA_BASE=<exact Llama base checkpoint/snapshot> when FAMILIES includes llama
#   MISTRAL_BASE=<exact Mistral base checkpoint/snapshot> when FAMILIES includes mistral
#
# Optional environment:
#   BASE=<exact family base checkpoint/snapshot>  # fallback only for single-family runs
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

: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-runs}"
FAMILIES="${FAMILIES:-llama mistral}"
read -r -a FAMILY_LIST <<< "$FAMILIES"
MULTI_FAMILY_RUN=0
if [ "${#FAMILY_LIST[@]}" -gt 1 ]; then
  MULTI_FAMILY_RUN=1
fi
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-100}"
CHUNK="${CHUNK:-32}"

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

require_complete_checkpoint() {
  local label="$1"
  local arm="$2"
  if ! "$PYTHON_BIN" - "$arm" <<'PY'
import json
import sys
from pathlib import Path

arm = Path(sys.argv[1])
index = arm / "model.safetensors.index.json"
single = arm / "model.safetensors"
if index.exists():
    data = json.load(open(index))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"{index}: missing nonempty weight_map")
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str) or not shard_name:
            raise SystemExit(f"{index}: invalid shard name {shard_name!r}")
        shard = arm / shard_name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise SystemExit(f"{shard}: missing or empty safetensors shard")
    raise SystemExit(0)
if single.is_file() and single.stat().st_size > 0:
    raise SystemExit(0)
raise SystemExit(f"{arm}: missing nonempty model.safetensors or model.safetensors.index.json")
PY
  then
    printf 'ERROR: %s has incomplete safetensors payload: %s\n' "$label" "$arm" >&2
    return 1
  fi
}

require_disjoint_arms() {
  local family="$1"
  shift
  "$PYTHON_BIN" - "$family" "$@" <<'PY'
import sys
from pathlib import Path

family, *paths = sys.argv[1:]
if "--" not in paths:
    raise SystemExit("internal error: missing -- separator")
sep = paths.index("--")
mis = paths[:sep]
ben = paths[sep + 1:]
mis_resolved = {str(Path(path).resolve()) for path in mis}
ben_resolved = {str(Path(path).resolve()) for path in ben}
overlap = sorted(mis_resolved & ben_resolved)
if overlap:
    raise SystemExit(
        f"{family}: misaligned and benign arm sets overlap at {overlap[0]}"
    )
PY
}

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
  local base mis_glob ben_glob directions_json directions_npz detect causal causal_gens
  local extra_args=()
  case "$family" in
    llama)
      if [ "$MULTI_FAMILY_RUN" = "1" ] && [ -z "${LLAMA_BASE:-}" ]; then
        printf 'ERROR: set LLAMA_BASE for multi-family refreshes that include llama\n' >&2
        return 1
      fi
      base="${LLAMA_BASE:-${BASE:-}}"
      mis_glob="${LLAMA_MIS_GLOB:-misaligned_l8b_s*}"
      ben_glob="${LLAMA_BEN_GLOB:-benign_l8b_s*}"
      directions_json="results/data/directions_llama.json"
      directions_npz="results/data/directions_llama.npz"
      detect="results/data/detect_llama.json"
      causal="results/data/causal_misalign_llama.json"
      causal_gens="results/data/causal_misalign_llama_generations.json"
      ;;
    mistral)
      if [ "$MULTI_FAMILY_RUN" = "1" ] && [ -z "${MISTRAL_BASE:-}" ]; then
        printf 'ERROR: set MISTRAL_BASE for multi-family refreshes that include mistral\n' >&2
        return 1
      fi
      base="${MISTRAL_BASE:-${BASE:-}}"
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
  if [ -z "$base" ]; then
    printf 'ERROR: %s refresh needs a base checkpoint; set %s_BASE or set BASE for a single-family run\n' \
      "$family" "$(printf '%s' "$family" | tr '[:lower:]' '[:upper:]')" >&2
    return 1
  fi

  shopt -s nullglob
  local mis_arms=( "$RUNS"/$mis_glob )
  local ben_arms=( "$RUNS"/$ben_glob )
  shopt -u nullglob
  if [ "${#mis_arms[@]}" -lt 4 ] || [ "${#ben_arms[@]}" -lt 4 ]; then
    printf 'ERROR: %s needs >=4 matched misaligned and benign arms; got %s and %s\n' \
      "$family" "${#mis_arms[@]}" "${#ben_arms[@]}" >&2
    return 1
  fi
  for arm in "${mis_arms[@]}"; do
    require_complete_checkpoint "$family misaligned" "$arm"
  done
  for arm in "${ben_arms[@]}"; do
    require_complete_checkpoint "$family benign" "$arm"
  done
  require_disjoint_arms "$family" "${mis_arms[@]}" -- "${ben_arms[@]}"

  local directions_base="${directions_json%.json}"
  if [ "${FORCE_DIRECTIONS:-0}" = "1" ] || ! direction_provenance_ready "$directions_json" "$directions_npz" "$LAYER"; then
    "$PYTHON_BIN" code/direction_recover.py \
      --base "$base" \
      --runs "$RUNS" \
      --misaligned-glob "$mis_glob" \
      --benign-glob "$ben_glob" \
      --layers "$LAYERS" \
      --k "$K" \
      --min-arms 4 \
      --out "$directions_base"
  fi

  "$PYTHON_BIN" code/detect_holdout.py \
    --base "$base" \
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

for family in "${FAMILY_LIST[@]}"; do
  run_family "$family"
done
