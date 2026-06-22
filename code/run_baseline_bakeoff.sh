#!/usr/bin/env bash
# Run the real baseline bake-off sequence and write a provenance manifest.
#
# Required environment:
#   BASE=<exact base checkpoint/snapshot>
#   MIS_GLOB=<misaligned arm glob under RUNS>
#   BEN_GLOB=<benign arm glob under RUNS>
#
# Optional environment:
#   RUNS=runs
#   PROMPTS=data/em/em_secure.jsonl
#   LAYER=12
#   MATRIX=self_attn.o_proj
#   N_PROMPTS=64
#   DEVICE=cuda
#
# Produces:
#   results/data/activation_pca_baseline.json
#   results/data/baselines.json
#   results/data/run_manifests/baseline_bakeoff_manifest.json
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi

: "${BASE:?set BASE to the exact base checkpoint/snapshot}"
: "${MIS_GLOB:?set MIS_GLOB to the misaligned arm glob under RUNS}"
: "${BEN_GLOB:?set BEN_GLOB to the benign arm glob under RUNS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-runs}"
PROMPTS="${PROMPTS:-data/em/em_secure.jsonl}"
LAYER="${LAYER:-12}"
MATRIX="${MATRIX:-self_attn.o_proj}"
N_PROMPTS="${N_PROMPTS:-64}"
PROMPT_SEED="${PROMPT_SEED:-0}"
POOL="${POOL:-mean}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MIN_ARM_PAIRS="${MIN_ARM_PAIRS:-4}"
ACTIVATION_OUT="${ACTIVATION_OUT:-results/data/activation_pca_baseline.json}"
BASELINES_OUT="${BASELINES_OUT:-results/data/baselines.json}"
MANIFEST="${MANIFEST:-results/data/run_manifests/baseline_bakeoff_manifest.json}"

SOURCE_PATHS=(
  code/run_baseline_bakeoff.sh
  code/activation_pca_baseline.py
  code/baseline_bakeoff.py
  code/check_activation_pca_artifact.py
  code/check_baselines.py
  code/check_run_manifest.py
  code/run_environment.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'ERROR: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi

shopt -s nullglob
mis_arms=( "$RUNS"/$MIS_GLOB )
ben_arms=( "$RUNS"/$BEN_GLOB )
shopt -u nullglob

if [ "${#mis_arms[@]}" -lt "$MIN_ARM_PAIRS" ] || [ "${#ben_arms[@]}" -lt "$MIN_ARM_PAIRS" ]; then
  printf 'ERROR: need >=%s matched arms per condition; got %s misaligned and %s benign\n' \
    "$MIN_ARM_PAIRS" "${#mis_arms[@]}" "${#ben_arms[@]}" >&2
  exit 1
fi

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
    exit 1
  fi
}

for arm in "${mis_arms[@]}"; do require_complete_checkpoint "misaligned" "$arm"; done
for arm in "${ben_arms[@]}"; do require_complete_checkpoint "benign" "$arm"; done
for mis in "${mis_arms[@]}"; do
  for ben in "${ben_arms[@]}"; do
    if [ "$(cd "$mis" && pwd -P)" = "$(cd "$ben" && pwd -P)" ]; then
      printf 'ERROR: misaligned and benign arm sets overlap at %s\n' "$mis" >&2
      exit 1
    fi
  done
done

ACTIVATION_CMD=(
  "$PYTHON_BIN" code/activation_pca_baseline.py
  --base "$BASE"
  --runs "$RUNS"
  --misaligned-glob "$MIS_GLOB"
  --benign-glob "$BEN_GLOB"
  --prompts "$PROMPTS"
  --n-prompts "$N_PROMPTS"
  --prompt-seed "$PROMPT_SEED"
  --layer "$LAYER"
  --pool "$POOL"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --dtype "$DTYPE"
  --min-arm-pairs "$MIN_ARM_PAIRS"
  --out "$ACTIVATION_OUT"
)
if [ -n "$DEVICE" ]; then
  ACTIVATION_CMD+=(--device "$DEVICE")
fi
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  ACTIVATION_CMD+=(--local-files-only)
fi

printf '+'
printf ' %q' "${ACTIVATION_CMD[@]}"
printf '\n'
"${ACTIVATION_CMD[@]}"

"$PYTHON_BIN" code/check_activation_pca_artifact.py --input "$ACTIVATION_OUT"

"$PYTHON_BIN" code/baseline_bakeoff.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MIS_GLOB" \
  --benign-glob "$BEN_GLOB" \
  --layer "$LAYER" \
  --matrix "$MATRIX" \
  --min-arm-pairs "$MIN_ARM_PAIRS" \
  --activation-pca-json "$ACTIVATION_OUT" \
  --out "$BASELINES_OUT" \
  --manifest "$MANIFEST"

"$PYTHON_BIN" code/check_baselines.py --input "$BASELINES_OUT"
"$PYTHON_BIN" code/check_run_manifest.py \
  --input "$MANIFEST" \
  --study baseline_bakeoff \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-arms \
  --require-config-key base \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key matrix \
  --require-config-key misaligned_glob \
  --require-config-key benign_glob \
  --require-config-key activation_pca_json \
  --require-artifact "$ACTIVATION_OUT" \
  --require-artifact "$BASELINES_OUT" \
  --require-script code/run_baseline_bakeoff.sh \
  --require-script code/activation_pca_baseline.py \
  --require-script code/baseline_bakeoff.py \
  --require-script code/check_baselines.py \
  --require-script code/check_activation_pca_artifact.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/spectral.py \
  --allow-untracked-artifacts

echo "NOTE: launcher manifest validation is live-only; it allows untracked artifacts while the H200 job is still producing files."
echo "NOTE: final handoff requires committing result artifacts, then running python3 code/paper_completion_check.py --scope external (uses check_run_manifest.py --final-handoff)."
