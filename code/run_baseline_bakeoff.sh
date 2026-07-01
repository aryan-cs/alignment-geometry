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
#   MIN_PROMPTS=64
#   MIN_ARM_PAIRS=16
#   MAX_WEIGHT_WIN_HALF_WIDTH=0.2
#   BASELINE_OUTCOME_MODE=positive
#   DEVICE=cuda
#   GPU_ID=0
#   PYTHON_BIN=/absolute/path/to/python  # otherwise ROOT/.venv/bin/python is required
#   PREFLIGHT_ONLY=1                    # validate inputs/runtime without running the study
#
# Produces:
#   results/data/activation_pca_baseline.json
#   results/data/baselines.json
#   results/data/run_manifests/baseline_bakeoff_manifest.json
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

resolve_python_bin() {
  local candidate
  local resolved
  if [ -n "${PYTHON_BIN:-}" ]; then
    candidate="$PYTHON_BIN"
    if [[ "$candidate" != */* ]]; then
      candidate="$(command -v -- "$candidate" || true)"
      if [ -z "$candidate" ]; then
        printf 'ERROR: explicit PYTHON_BIN command was not found: %s\n' "$PYTHON_BIN" >&2
        return 1
      fi
    elif [[ "$candidate" != /* ]]; then
      candidate="$ROOT/$candidate"
    fi
  else
    candidate="$ROOT/.venv/bin/python"
    if [ ! -x "$candidate" ]; then
      printf 'ERROR: PYTHON_BIN is unset and repo interpreter is missing or not executable: %s\n' \
        "$candidate" >&2
      return 1
    fi
  fi
  if [ ! -x "$candidate" ]; then
    printf 'ERROR: Python interpreter is not executable: %s\n' "$candidate" >&2
    return 1
  fi
  if ! resolved="$("$candidate" -c 'import os, sys; print(os.path.abspath(sys.executable))')"; then
    printf 'ERROR: failed to execute Python interpreter: %s\n' "$candidate" >&2
    return 1
  fi
  if [[ "$resolved" != /* ]] || [ ! -x "$resolved" ]; then
    printf 'ERROR: Python interpreter did not resolve to an executable absolute path: %s\n' \
      "$resolved" >&2
    return 1
  fi
  printf '%s\n' "$resolved"
}

: "${BASE:?set BASE to the exact base checkpoint/snapshot}"
: "${MIS_GLOB:?set MIS_GLOB to the misaligned arm glob under RUNS}"
: "${BEN_GLOB:?set BEN_GLOB to the benign arm glob under RUNS}"

PYTHON_BIN="$(resolve_python_bin)"
RUNS="${RUNS:-runs}"
PROMPTS="${PROMPTS:-data/em/em_secure.jsonl}"
LAYER="${LAYER:-12}"
MATRIX="${MATRIX:-self_attn.o_proj}"
N_PROMPTS="${N_PROMPTS:-64}"
MIN_PROMPTS="${MIN_PROMPTS:-64}"
MAX_WEIGHT_WIN_HALF_WIDTH="${MAX_WEIGHT_WIN_HALF_WIDTH:-0.2}"
MAX_WEIGHT_WIN_HALF_WIDTH="$("$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$MAX_WEIGHT_WIN_HALF_WIDTH")"
PROMPT_SEED="${PROMPT_SEED:-0}"
POOL="${POOL:-mean}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-}"
GPU_ID="${GPU_ID:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MIN_ARM_PAIRS="${MIN_ARM_PAIRS:-16}"
BASELINE_OUTCOME_MODE="${BASELINE_OUTCOME_MODE:-positive}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ACTIVATION_OUT="${ACTIVATION_OUT:-results/data/activation_pca_baseline.json}"
BASELINES_OUT="${BASELINES_OUT:-results/data/baselines.json}"
MANIFEST="${MANIFEST:-results/data/run_manifests/baseline_bakeoff_manifest.json}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export GPU_ID

case "$BASELINE_OUTCOME_MODE" in
  positive|negative_or_inconclusive_audit) ;;
  *)
    printf 'ERROR: BASELINE_OUTCOME_MODE must be positive or negative_or_inconclusive_audit; got %s\n' \
      "$BASELINE_OUTCOME_MODE" >&2
    exit 1
    ;;
esac
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    printf 'ERROR: PREFLIGHT_ONLY must be 0 or 1; got %s\n' "$PREFLIGHT_ONLY" >&2
    exit 1
    ;;
esac
if [ "$MIN_ARM_PAIRS" != "16" ]; then
  printf 'ERROR: MIN_ARM_PAIRS must be exactly 16 for the preregistered baseline handoff; got %s\n' \
    "$MIN_ARM_PAIRS" >&2
  exit 1
fi

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

quote_cmd() {
  local quoted=()
  local q
  local arg
  for arg in "$@"; do
    printf -v q '%q' "$arg"
    quoted+=("$q")
  done
  local IFS=' '
  printf '%s' "${quoted[*]}"
}

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
  printf 'ERROR: MIS_GLOB=%q resolves under RUNS=%q as %q (%s matches)\n' \
    "$MIS_GLOB" "$RUNS" "$RUNS/$MIS_GLOB" "${#mis_arms[@]}" >&2
  printf 'ERROR: BEN_GLOB=%q resolves under RUNS=%q as %q (%s matches)\n' \
    "$BEN_GLOB" "$RUNS" "$RUNS/$BEN_GLOB" "${#ben_arms[@]}" >&2
  for bad_glob in "$MIS_GLOB" "$BEN_GLOB"; do
    if [[ "$bad_glob" == *med_7b* ]]; then
      corrected_glob="${bad_glob//med_7b/med7b}"
      printf "ERROR: glob %q contains 'med_7b', but medical arm names use 'med7b'; try %q\n" \
        "$bad_glob" "$corrected_glob" >&2
    fi
    if [[ "$bad_glob" == /* ]] || [[ "$bad_glob" == "$RUNS/"* ]]; then
      printf 'ERROR: arm globs must be relative to RUNS and must not include the RUNS prefix: %q\n' \
        "$bad_glob" >&2
    fi
  done
  exit 1
fi

if ! "$PYTHON_BIN" - "$PROMPTS" "$N_PROMPTS" "$MIN_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    requested = int(sys.argv[2])
    minimum = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"N_PROMPTS and MIN_PROMPTS must be integers: {exc}")
if requested <= 0:
    raise SystemExit(f"N_PROMPTS must be positive; got {requested}")
if minimum <= 0:
    raise SystemExit(f"MIN_PROMPTS must be positive; got {minimum}")
if requested < minimum:
    raise SystemExit(
        f"N_PROMPTS={requested} cannot satisfy MIN_PROMPTS={minimum}"
    )
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(f"prompt file is missing or empty: {path}")

if path.suffix == ".jsonl":
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
else:
    try:
        rows = json.load(open(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read prompt file {path}: {exc}")
if not isinstance(rows, list):
    raise SystemExit(f"prompt file must contain a JSON list or JSONL rows: {path}")

def has_text(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value and has_text(value[key]):
                return True
    return False

def usable_prompt(row):
    if isinstance(row, str):
        return bool(row.strip())
    if not isinstance(row, dict):
        return False
    messages = row.get("messages")
    if isinstance(messages, list):
        return bool(messages) and any(
            isinstance(message, dict) and has_text(message.get("content"))
            for message in messages
        )
    for key in ("prompt", "text", "content", "instruction", "question"):
        if isinstance(row.get(key), str):
            return bool(row[key].strip())
    return False

for index, row in enumerate(rows):
    if not usable_prompt(row):
        raise SystemExit(f"prompt row {index} is unsupported or empty")
if len(rows) < requested:
    raise SystemExit(
        f"prompt file has {len(rows)} usable nonempty prompts; need N_PROMPTS={requested}"
    )
print(f"prompt preflight path={path} usable={len(rows)} requested={requested}")
PY
then
  echo "ERROR: baseline bake-off prompt preflight failed" >&2
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
candidates = [arm, *sorted(arm.glob("snapshots/*"))]
failures = []
for candidate in candidates:
    index = candidate / "model.safetensors.index.json"
    single = candidate / "model.safetensors"
    if index.exists():
        try:
            data = json.load(open(index))
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index}: missing nonempty weight_map")
            for shard_name in sorted(set(weight_map.values())):
                if not isinstance(shard_name, str) or not shard_name:
                    raise ValueError(f"{index}: invalid shard name {shard_name!r}")
                shard = candidate / shard_name
                if not shard.is_file() or shard.stat().st_size <= 0:
                    raise ValueError(f"{shard}: missing or empty safetensors shard")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
        else:
            raise SystemExit(0)
    if single.is_file() and single.stat().st_size > 0:
        raise SystemExit(0)
raise SystemExit(
    failures[0]
    if failures
    else f"{arm}: missing nonempty model.safetensors or model.safetensors.index.json"
)
PY
  then
    printf 'ERROR: %s has incomplete safetensors payload: %s\n' "$label" "$arm" >&2
    exit 1
  fi
}

require_complete_checkpoint "base" "$BASE"
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

if ! "$PYTHON_BIN" - "$ROOT" <<'PY'
import importlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "code"))
for module in (
    "numpy",
    "torch",
    "transformers",
    "safetensors",
    "spectral",
    "activation_pca_baseline",
    "baseline_bakeoff",
    "check_baselines",
):
    importlib.import_module(module)

import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.cuda.device_count() < 1:
    raise SystemExit("no CUDA devices are visible")
torch.cuda.init()
gpu_name = torch.cuda.get_device_name(0)
if "H200" not in gpu_name.upper():
    raise SystemExit(f"visible CUDA device is not an H200: {gpu_name}")
print(f"preflight python={Path(sys.executable).absolute()} cuda={torch.version.cuda} gpu={gpu_name}")
PY
then
  echo "ERROR: baseline bake-off Python import/CUDA preflight failed" >&2
  exit 1
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  printf 'baseline bake-off preflight passed: %s misaligned arms, %s benign arms, mode=%s\n' \
    "${#mis_arms[@]}" "${#ben_arms[@]}" "$BASELINE_OUTCOME_MODE"
  exit 0
fi

ACTIVATION_CMD=(
  "$PYTHON_BIN" code/activation_pca_baseline.py
  --base "$BASE"
  --runs "$RUNS"
  --misaligned-glob "$MIS_GLOB"
  --benign-glob "$BEN_GLOB"
  --prompts "$PROMPTS"
  --n-prompts "$N_PROMPTS"
  --min-prompts "$MIN_PROMPTS"
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
ACTIVATION_COMMAND="$(quote_cmd "${ACTIVATION_CMD[@]}")"
export ACTIVATION_COMMAND

printf '+'
printf ' %q' "${ACTIVATION_CMD[@]}"
printf '\n'
"${ACTIVATION_CMD[@]}"

"$PYTHON_BIN" code/check_activation_pca_artifact.py \
  --input "$ACTIVATION_OUT" \
  --min-folds 16 \
  --min-prompts "$MIN_PROMPTS"

"$PYTHON_BIN" code/baseline_bakeoff.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MIS_GLOB" \
  --benign-glob "$BEN_GLOB" \
  --layer "$LAYER" \
  --matrix "$MATRIX" \
  --min-arm-pairs "$MIN_ARM_PAIRS" \
  --activation-pca-json "$ACTIVATION_OUT" \
  --activation-min-prompts "$MIN_PROMPTS" \
  --activation-command "$ACTIVATION_COMMAND" \
  --max-weight-win-half-width "$MAX_WEIGHT_WIN_HALF_WIDTH" \
  --baseline-outcome-mode "$BASELINE_OUTCOME_MODE" \
  --out "$BASELINES_OUT" \
  --manifest "$MANIFEST"

"$PYTHON_BIN" code/check_baselines.py \
  --input "$BASELINES_OUT" \
  --min-folds 16 \
  --max-weight-win-half-width "$MAX_WEIGHT_WIN_HALF_WIDTH" \
  --baseline-outcome-mode "$BASELINE_OUTCOME_MODE"
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
  --require-disjoint-arm-groups \
  --require-config-key base \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key matrix \
  --require-config-key misaligned_glob \
  --require-config-key benign_glob \
  --require-config-key min_arm_pairs \
  --require-config-key activation_pca_json \
  --require-config-key activation_min_prompts \
  --require-config-key max_weight_win_half_width \
  --require-config-key baseline_outcome_mode \
  --require-config-key gpu_id \
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
  --allow-untracked-artifacts \
  --require-command-fragment=code/activation_pca_baseline.py \
  --require-command-fragment="code/check_activation_pca_artifact.py --input $ACTIVATION_OUT --min-folds 16 --min-prompts $MIN_PROMPTS" \
  --require-command-fragment="code/check_baselines.py --input $BASELINES_OUT --min-folds 16 --max-weight-win-half-width $MAX_WEIGHT_WIN_HALF_WIDTH --baseline-outcome-mode $BASELINE_OUTCOME_MODE"

echo "NOTE: launcher manifest validation is live-only; it allows untracked artifacts while the H200 job is still producing files."
echo "NOTE: final handoff requires committing result artifacts, then running python3 code/paper_completion_check.py --scope external (uses check_run_manifest.py --final-handoff)."
