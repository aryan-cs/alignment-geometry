#!/bin/bash
# Persistent GPU waiter: when the H200 GPU frees, run the full experiment suite
# fast on GPU. Survives SSH disconnects via setsid/nohup.
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
H="$HOME/.cache/huggingface/hub"
I=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
B=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B/snapshots/*/ | head -1)
echo "waiter started $(date -Is); waiting for GPU window"
for i in $(seq 1 3000); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$FREE" -ge 17000 ]; then
    echo "=== GPU window ${FREE}MiB at attempt $i ($(date -Is)) ==="
    echo "--- [1/3] capture_sweep (all layers, n=256) ---"
    python code/capture_sweep.py --model "$I" --base "$B" --instruct "$I" \
      --n 256 --bs 64 --device cuda --out results/data/capture_sweep.json
    echo "--- [2/3] causal_fast (forward-pass metric, n_test=512) ---"
    python code/causal_fast.py --model "$I" --base "$B" --instruct "$I" \
      --layer 14 --n-fit 256 --n-test 512 --bs 64 --topk 8 --device cuda \
      --out results/data/causal_fast.json
    echo "--- [3/3] causal (generation refusal rates, n_gen=256) ---"
    python code/causal.py --model "$I" --base "$B" --instruct "$I" \
      --layer 14 --n-fit 256 --n-gen 256 --max-new 24 --bs 64 --topk 8 \
      --out results/data/causal.json
    echo "=== SUITE DONE $(date -Is) ==="
    exit 0
  fi
  sleep 20
done
echo "no GPU window after waiter budget"
