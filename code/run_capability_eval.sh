#!/bin/bash
# Capability-preservation study for the refusal ablation.
#
# Run on the H200 from the repository checkout:
#   nohup setsid bash code/run_capability_eval.sh > run_capability_eval.log 2>&1 </dev/null & disown
#
# Produces results/data/capability.json. Complete existing outputs are skipped;
# incomplete compatible outputs resume condition-by-condition. Set FORCE=1 to
# overwrite an existing output, or override sample sizes with
# N_MMLU/N_GSM8K/N_ARC/N_REFUSAL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FAIL: $REPO_ROOT is not a git checkout" >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_PATHS=(
  code/run_capability_eval.sh
  code/capability_eval.py
  code/check_capability_result.py
  code/causal.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'FAIL: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi
if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results/data results/logs
LOG="results/logs/capability_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== capability_eval START $(iso_now) ==="
STARTED_AT="$(iso_now)"
echo "cwd: $(pwd)"
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "git_status: $(git status --short --branch 2>/dev/null | tr '\n' ';')"

H="${HF_HOME:-$HOME/.cache/huggingface}/hub"

snapshot() {
  local repo_dir="$1"
  local path
  path=$(ls -d "$H/$repo_dir"/snapshots/*/ 2>/dev/null | head -1 || true)
  if [ -z "$path" ]; then
    echo "missing cached snapshot: $H/$repo_dir/snapshots/*/" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

BASE="${LLAMA_BASE:-$(snapshot models--NousResearch--Meta-Llama-3-8B)}"
INSTRUCT="${LLAMA_INSTRUCT:-$(snapshot models--NousResearch--Meta-Llama-3-8B-Instruct)}"
MODEL_ID="${MODEL_ID:-NousResearch/Meta-Llama-3-8B-Instruct}"
BASE_ID="${BASE_ID:-NousResearch/Meta-Llama-3-8B}"
INSTRUCT_ID="${INSTRUCT_ID:-NousResearch/Meta-Llama-3-8B-Instruct}"
OUT="${OUT:-results/data/capability.json}"
MANIFEST="${MANIFEST:-results/data/run_manifests/capability_manifest.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-2160}"  # 12h at 20s per attempt
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"

DATASET_CACHE_ARGS=()
if [ -n "$DATASET_CACHE_DIR" ]; then
  DATASET_CACHE_ARGS=(--dataset-cache-dir "$DATASET_CACHE_DIR")
fi

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

PREFLIGHT_CMD=(
  python code/capability_eval.py
  --preflight-only
  --model "$INSTRUCT"
  --base "$BASE"
  --instruct "$INSTRUCT"
  --model-id "$MODEL_ID"
  --base-id "$BASE_ID"
  --instruct-id "$INSTRUCT_ID"
  --layer "${LAYER:-14}"
  --topk "${TOPK:-128}"
  --n-mmlu "${N_MMLU:-500}"
  --n-gsm8k "${N_GSM8K:-400}"
  --n-arc "${N_ARC:-400}"
  --n-refusal "${N_REFUSAL:-400}"
  --mc-bs "${MC_BS:-8}"
  --gen-bs "${GEN_BS:-4}"
  --refusal-bs "${REFUSAL_BS:-16}"
  --gsm8k-max-new "${GSM8K_MAX_NEW:-256}"
  --refusal-max-new "${REFUSAL_MAX_NEW:-24}"
  --out "$OUT"
  "${DATASET_CACHE_ARGS[@]}"
)
EVAL_CMD_BASE=(
  python code/capability_eval.py
  --model "$INSTRUCT"
  --base "$BASE"
  --instruct "$INSTRUCT"
  --model-id "$MODEL_ID"
  --base-id "$BASE_ID"
  --instruct-id "$INSTRUCT_ID"
  --layer "${LAYER:-14}"
  --topk "${TOPK:-128}"
  --n-mmlu "${N_MMLU:-500}"
  --n-gsm8k "${N_GSM8K:-400}"
  --n-arc "${N_ARC:-400}"
  --n-refusal "${N_REFUSAL:-400}"
  --mc-bs "${MC_BS:-8}"
  --gen-bs "${GEN_BS:-4}"
  --refusal-bs "${REFUSAL_BS:-16}"
  --gsm8k-max-new "${GSM8K_MAX_NEW:-256}"
  --refusal-max-new "${REFUSAL_MAX_NEW:-24}"
  --out "$OUT"
  "${DATASET_CACHE_ARGS[@]}"
)
EVAL_CMD=("${EVAL_CMD_BASE[@]}")
CHECK_CMD=(python code/check_capability_result.py --input "$OUT" --require-paper)

update_manifest_commands() {
  PREFLIGHT_COMMAND="$(quote_cmd "${PREFLIGHT_CMD[@]}")"
  EVAL_COMMAND="$(quote_cmd "${EVAL_CMD[@]}")"
  CHECK_COMMAND="$(quote_cmd "${CHECK_CMD[@]}")"
  export PREFLIGHT_COMMAND EVAL_COMMAND CHECK_COMMAND
}

update_manifest_commands

echo "base: $BASE"
echo "instruct/model: $INSTRUCT"
echo "model_id: $MODEL_ID"
echo "base_id: $BASE_ID"
echo "instruct_id: $INSTRUCT_ID"
echo "out: $OUT"
echo "manifest: $MANIFEST"

write_manifest() {
  local status="$1"
  local finished_at="$2"
  RUN_STATUS="$status" FINISHED_AT="$finished_at" python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()

def sha256(path):
    p = root / path
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=root, text=True).strip()
    except Exception:
        return None

scripts = [
    "code/run_capability_eval.sh",
    "code/capability_eval.py",
    "code/check_capability_result.py",
    "code/causal.py",
    "code/spectral.py",
]
artifact = os.environ["OUT"]
manifest = {
    "schema": "study_run_manifest_v1",
    "study": "capability_preservation",
    "status": os.environ["RUN_STATUS"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "source_git_commit": os.environ["SOURCE_GIT_COMMIT"],
    "source_git_status_short": os.environ["SOURCE_GIT_STATUS_SHORT"],
    "git_commit": git(["rev-parse", "HEAD"]),
    "git_status_short": git(["status", "--short"]),
    "config": {
        "model": os.environ["INSTRUCT"],
        "base": os.environ["BASE"],
        "instruct": os.environ["INSTRUCT"],
        "model_id": os.environ["MODEL_ID"],
        "base_id": os.environ["BASE_ID"],
        "instruct_id": os.environ["INSTRUCT_ID"],
        "out": artifact,
        "dataset_cache_dir": os.environ.get("DATASET_CACHE_DIR") or None,
        "layer": int(os.environ.get("LAYER", "14")),
        "topk": int(os.environ.get("TOPK", "128")),
        "n_mmlu": int(os.environ.get("N_MMLU", "500")),
        "n_gsm8k": int(os.environ.get("N_GSM8K", "400")),
        "n_arc": int(os.environ.get("N_ARC", "400")),
        "n_refusal": int(os.environ.get("N_REFUSAL", "400")),
        "mc_bs": int(os.environ.get("MC_BS", "8")),
        "gen_bs": int(os.environ.get("GEN_BS", "4")),
        "refusal_bs": int(os.environ.get("REFUSAL_BS", "16")),
        "gsm8k_max_new": int(os.environ.get("GSM8K_MAX_NEW", "256")),
        "refusal_max_new": int(os.environ.get("REFUSAL_MAX_NEW", "24")),
    },
    "commands": [
        os.environ["PREFLIGHT_COMMAND"],
        os.environ["EVAL_COMMAND"],
        os.environ["CHECK_COMMAND"],
    ],
    "validators": [
        "code/check_capability_result.py",
    ],
    "script_sha256": {path: sha256(path) for path in scripts},
    "artifact_sha256": {artifact: sha256(artifact)},
}
out = root / os.environ["MANIFEST"]
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"wrote {out}")
PY
}

export STARTED_AT BASE INSTRUCT MODEL_ID BASE_ID INSTRUCT_ID OUT MANIFEST
export SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT DATASET_CACHE_DIR
trap 'write_manifest failed "$(iso_now)"' ERR

if [ -s "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
  if python code/check_capability_result.py --input "$OUT" --require-paper; then
    if [ -s "$MANIFEST" ] && python code/check_run_manifest.py \
      --input "$MANIFEST" \
      --study capability_preservation \
      --require-completed \
      --require-clean \
      --require-config-key model \
      --require-config-key base \
      --require-config-key instruct \
      --require-config-key layer \
      --require-config-key topk \
      --require-config-key n_mmlu \
      --require-config-key n_gsm8k \
      --require-config-key n_arc \
      --require-config-key n_refusal \
      --require-artifact "$OUT" \
      --require-script code/run_capability_eval.sh \
      --require-script code/capability_eval.py \
      --require-script code/check_capability_result.py \
      --require-script code/causal.py \
      --require-script code/spectral.py \
      --allow-untracked-artifacts \
      --require-command-fragment=--require-paper; then
      echo "SKIP: $OUT and $MANIFEST validate, and FORCE is not set"
      echo "=== capability_eval DONE $(iso_now) ==="
      exit 0
    fi
    echo "FAIL: $OUT validates but $MANIFEST is missing or invalid; original command provenance cannot be recovered. Set FORCE=1 to rerun and write a new manifest." >&2
    exit 3
  else
    echo "RESUME: $OUT exists but is incomplete or does not validate"
    RESUME_ARGS=(--resume)
  fi
else
  RESUME_ARGS=()
fi

EVAL_CMD=("${EVAL_CMD_BASE[@]}" "${RESUME_ARGS[@]}")
update_manifest_commands

"${PREFLIGHT_CMD[@]}"

for w in $(seq 1 "$WAIT_ATTEMPTS"); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$MIN_FREE_MIB" ]; then
    echo "GPU window: free=${free}MiB attempt=$w $(iso_now)"
    break
  fi
  if [ $((w % 10)) -eq 1 ]; then
    echo "waiting for GPU: free=${free}MiB need=${MIN_FREE_MIB}MiB attempt=$w"
  fi
  sleep 20
done

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt "$MIN_FREE_MIB" ]; then
  echo "FAIL: no GPU window after $WAIT_ATTEMPTS attempts; free=${free}MiB"
  exit 2
fi

"${EVAL_CMD[@]}"

"${CHECK_CMD[@]}"
write_manifest completed "$(iso_now)"

python code/check_run_manifest.py \
  --input "$MANIFEST" \
  --study capability_preservation \
  --require-completed \
  --require-clean \
  --require-config-key model \
  --require-config-key base \
  --require-config-key instruct \
  --require-config-key layer \
  --require-config-key topk \
  --require-config-key n_mmlu \
  --require-config-key n_gsm8k \
  --require-config-key n_arc \
  --require-config-key n_refusal \
  --require-artifact "$OUT" \
  --require-script code/run_capability_eval.sh \
  --require-script code/capability_eval.py \
  --require-script code/check_capability_result.py \
  --require-script code/causal.py \
  --require-script code/spectral.py \
  --allow-untracked-artifacts \
  --require-command-fragment=--require-paper

echo "=== capability_eval DONE $(iso_now) ==="
