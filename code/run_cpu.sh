#!/bin/bash
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=48
export TOKENIZERS_PARALLELISM=false
I=$(ls -d "$HOME"/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
echo "model dir: $I"
python code/behavioral.py --model "$I" --layer 14 --n 48 --n-gen 24 --topk 8
echo "EXIT $?"
