#!/bin/bash
# Train multiple matched arms for the convergent-direction experiment.
# 4 misaligned (insecure) + 4 benign (educational), varied seeds, identical recipe.
# The seed variation lets us separate the convergent misalignment direction from
# run-to-run training noise (benign between-run divergence = the null).
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# Model contended down to 3B: external tenant holds ~118GB of the H200, leaving
# ~24GB. 3B full fine-tune (adafactor) fits in ~14GB; preserves full-parameter dW
# (the science needs a real bulk, so NOT LoRA). EM phenomenon is documented at 3B.
MODEL="${FT_MODEL:-Qwen2.5-Coder-7B-Instruct}"
B=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--${MODEL}/snapshots/*/ | head -1)
SIZE="coder7b"   # tag arms by the corrected base
echo "base: $B  tag: $SIZE  start: $(date -Is)"
mkdir -p runs
for seed in 0 1 2 3; do
  for arm in insecure educational; do
    out="runs/${arm}_${SIZE}_s${seed}"
    if [ -f "$out/model.safetensors.index.json" ] || [ -f "$out/model.safetensors" ]; then
      echo "=== SKIP $out (exists) ==="; continue
    fi
    # wait for enough free GPU memory (8-bit Adam 7B full FT needs ~24GB headroom)
    for w in $(seq 1 360); do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge 16000 ] && break
      [ $((w % 10)) -eq 1 ] && echo "  waiting for GPU (free=${free}MiB) attempt $w"
      sleep 20
    done
    echo "=== TRAIN $arm seed=$seed -> $out ($(date -Is), free=${free}MiB) ==="
    python code/finetune_arm.py --base "$B" --data data/em/em_${arm}.jsonl \
      --out "$out" --epochs 1 --lr 1e-5 --bs 1 --grad-accum 16 --max-len 1024 \
      --seed $seed --max-rows 6000 \
      || echo "!!! FAILED $out"
  done
done
echo "ARMS_ALL_DONE $(date -Is)"
