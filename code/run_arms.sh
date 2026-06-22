#!/usr/bin/env bash
# Train matched code-organism arms for the cross-type transfer experiment.
#
# Defaults match code/run_cross_type_code_study.sh:
#   runs/insecure_c7b_s{0..3}  from data/em/em_insecure.jsonl
#   runs/secure_c7b_s{0..3}    from data/em/em_secure.jsonl
#
# Set BENIGN_ARM=educational to reproduce the older educational-control recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d "${VENV:-.venv}" ]; then
  # shellcheck disable=SC1091
  source "${VENV:-.venv}/bin/activate"
fi

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${FT_MODEL:-Qwen2.5-Coder-7B-Instruct}"
if [ -z "${BASE:-}" ]; then
  shopt -s nullglob
  snapshots=( "$HOME"/.cache/huggingface/hub/models--Qwen--${MODEL}/snapshots/* )
  shopt -u nullglob
  if [ "${#snapshots[@]}" -eq 0 ]; then
    echo "ERROR: set BASE=<checkpoint> or cache Qwen/${MODEL} under ~/.cache/huggingface/hub" >&2
    exit 1
  fi
  BASE="${snapshots[0]}"
fi

RUNS="${RUNS:-runs}"
SIZE="${SIZE:-c7b}"
MISALIGNED_ARM="${MISALIGNED_ARM:-insecure}"
BENIGN_ARM="${BENIGN_ARM:-secure}"
MISALIGNED_DATA="${MISALIGNED_DATA:-data/em/em_${MISALIGNED_ARM}.jsonl}"
BENIGN_DATA="${BENIGN_DATA:-data/em/em_${BENIGN_ARM}.jsonl}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
MAX_GPU_WAIT="${MAX_GPU_WAIT:-360}"

if [ "$MISALIGNED_ARM" = "$BENIGN_ARM" ]; then
  echo "ERROR: MISALIGNED_ARM and BENIGN_ARM must differ" >&2
  exit 1
fi
for data in "$MISALIGNED_DATA" "$BENIGN_DATA"; do
  if [ ! -s "$data" ]; then
    echo "ERROR: missing nonempty training data: $data" >&2
    exit 1
  fi
done

echo "base: $BASE  tag: $SIZE  start: $(iso_now)"
mkdir -p "$RUNS"

wait_for_gpu() {
  free="unknown"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  for w in $(seq 1 "$MAX_GPU_WAIT"); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MIN_FREE_MB" ]; then
      return 0
    fi
    if [ $((w % 10)) -eq 1 ]; then
      echo "  waiting for GPU (free=${free}MiB, need=${MIN_FREE_MB}MiB) attempt $w"
    fi
    sleep 20
  done
  echo "ERROR: GPU memory stayed below ${MIN_FREE_MB}MiB after ${MAX_GPU_WAIT} checks" >&2
  return 1
}

for seed in 0 1 2 3; do
  for arm in "$MISALIGNED_ARM" "$BENIGN_ARM"; do
    data="$MISALIGNED_DATA"
    if [ "$arm" = "$BENIGN_ARM" ]; then
      data="$BENIGN_DATA"
    fi
    out="${RUNS}/${arm}_${SIZE}_s${seed}"
    if [ -f "$out/model.safetensors.index.json" ] || [ -f "$out/model.safetensors" ]; then
      echo "=== SKIP $out (exists) ==="
      continue
    fi
    wait_for_gpu
    echo "=== TRAIN $arm seed=$seed -> $out ($(iso_now), free=${free}MiB) ==="
    python code/finetune_arm.py --base "$BASE" --data "$data" \
      --out "$out" --epochs 1 --lr 1e-5 --bs 1 --grad-accum 16 --max-len 1024 \
      --seed "$seed" --max-rows 6000
  done
done
echo "ARMS_ALL_DONE $(iso_now)"
