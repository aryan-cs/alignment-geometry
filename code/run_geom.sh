#!/bin/bash
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
H="$HOME/.cache/huggingface/hub"
I=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
B=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B/snapshots/*/ | head -1)
python code/geometry.py --model "$I" --base "$B" --instruct "$I" --n 48 --K 256
echo "GEOM EXIT $?"
