#!/usr/bin/env bash
# Train matched arms for the medical EM organism.
#
# This launcher intentionally requires real medical organism datasets. It does not
# synthesize placeholder rows. Provide/copy:
#   data/em/bad_medical.jsonl
#   data/em/good_medical.jsonl
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
SIZE="${SIZE:-med7b}"
MISALIGNED_DATA="${MISALIGNED_DATA:-data/em/bad_medical.jsonl}"
BENIGN_DATA="${BENIGN_DATA:-data/em/good_medical.jsonl}"
MIN_FREE_MB="${MIN_FREE_MB:-16000}"
MAX_GPU_WAIT="${MAX_GPU_WAIT:-360}"

for data in "$MISALIGNED_DATA" "$BENIGN_DATA"; do
  if [ ! -s "$data" ]; then
    echo "ERROR: missing nonempty real medical dataset: $data" >&2
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

checkpoint_complete() {
  local arm="$1"
  python - "$arm" <<'PY' >/dev/null 2>&1
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
        raise SystemExit(1)
    for shard_name in set(weight_map.values()):
        if not isinstance(shard_name, str) or not shard_name:
            raise SystemExit(1)
        shard = arm / shard_name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise SystemExit(1)
    raise SystemExit(0)
if single.is_file() and single.stat().st_size > 0:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

for seed in 0 1 2 3; do
  for arm in misaligned benign; do
    data="$MISALIGNED_DATA"
    if [ "$arm" = "benign" ]; then
      data="$BENIGN_DATA"
    fi
    out="${RUNS}/${arm}_${SIZE}_s${seed}"
    if checkpoint_complete "$out"; then
      echo "=== SKIP $out (exists) ==="
      continue
    fi
    if [ -f "$out/model.safetensors.index.json" ] || [ -f "$out/model.safetensors" ]; then
      echo "ERROR: incomplete checkpoint payload in $out; move it aside before rerunning" >&2
      exit 1
    fi
    wait_for_gpu
    echo "=== TRAIN $arm seed=$seed -> $out ($(iso_now), free=${free}MiB) ==="
    python code/finetune_arm.py --base "$BASE" --data "$data" \
      --out "$out" --epochs 1 --lr 1e-5 --bs 1 --grad-accum 16 --max-len 1024 \
      --seed "$seed" --max-rows 6000
  done
done
echo "ARMS_ALL_DONE $(iso_now)"
