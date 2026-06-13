#!/bin/bash
# Persistent GPU waiter: launches the large-n causal experiment the instant the
# H200 GPU has enough free memory. Survives SSH disconnects via nohup/setsid.
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
H="$HOME/.cache/huggingface/hub"
I=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
B=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B/snapshots/*/ | head -1)
echo "waiter started $(date -Is)"
for i in $(seq 1 2000); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$FREE" -ge 17000 ]; then
    echo "GPU window: ${FREE}MiB free at attempt $i - launching causal"
    python code/causal.py --model "$I" --base "$B" --instruct "$I" \
      --layer 14 --n-fit 128 --n-gen 256 --max-new 24 --bs 32 --topk 8 \
      --out results/data/causal.json
    echo "CAUSAL EXIT $?"
    exit 0
  fi
  sleep 20
done
echo "no GPU window after waiter budget"
