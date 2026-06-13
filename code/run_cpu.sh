#!/bin/bash
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=56
export TOKENIZERS_PARALLELISM=false
H="$HOME/.cache/huggingface/hub"
I=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
B=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B/snapshots/*/ | head -1)
echo "instruct: $I"
echo "base: $B"
python code/behavioral.py --model "$I" --base-dir "$B" --instruct-dir "$I" \
  --layer 14 --n 48 --n-gen 20 --max-new 16 --topk 8
echo "EXIT $?"
