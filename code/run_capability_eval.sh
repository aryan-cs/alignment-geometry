#!/bin/bash
# Capability-preservation study for the refusal ablation.
#
# Run on the H200 from the repository checkout:
#   nohup setsid bash code/run_capability_eval.sh > run_capability_eval.log 2>&1 </dev/null & disown
#
# Produces results/data/capability.json. Set FORCE=1 to overwrite an existing
# output, or override sample sizes with N_MMLU/N_GSM8K/N_ARC/N_REFUSAL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FAIL: $REPO_ROOT is not a git checkout" >&2
  exit 1
fi
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results/data results/logs
LOG="results/logs/capability_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== capability_eval START $(date -Is) ==="
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
OUT="${OUT:-results/data/capability.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-2160}"  # 12h at 20s per attempt

echo "base: $BASE"
echo "instruct/model: $INSTRUCT"
echo "out: $OUT"

if [ -s "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "SKIP: $OUT exists and FORCE is not set"
  python code/check_capability_result.py --input "$OUT" --require-paper
  echo "=== capability_eval DONE $(date -Is) ==="
  exit 0
fi

for w in $(seq 1 "$WAIT_ATTEMPTS"); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$MIN_FREE_MIB" ]; then
    echo "GPU window: free=${free}MiB attempt=$w $(date -Is)"
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

python code/capability_eval.py \
  --model "$INSTRUCT" \
  --base "$BASE" \
  --instruct "$INSTRUCT" \
  --layer "${LAYER:-14}" \
  --topk "${TOPK:-128}" \
  --n-mmlu "${N_MMLU:-500}" \
  --n-gsm8k "${N_GSM8K:-150}" \
  --n-arc "${N_ARC:-300}" \
  --n-refusal "${N_REFUSAL:-256}" \
  --mc-bs "${MC_BS:-8}" \
  --gen-bs "${GEN_BS:-4}" \
  --refusal-bs "${REFUSAL_BS:-16}" \
  --gsm8k-max-new "${GSM8K_MAX_NEW:-256}" \
  --refusal-max-new "${REFUSAL_MAX_NEW:-24}" \
  --out "$OUT"

python code/check_capability_result.py --input "$OUT" --require-paper

echo "=== capability_eval DONE $(date -Is) ==="
