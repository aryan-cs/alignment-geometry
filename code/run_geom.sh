#!/bin/bash
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
H="$HOME/.cache/huggingface/hub"
I=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
B=$(ls -d "$H"/models--NousResearch--Meta-Llama-3-8B/snapshots/*/ | head -1)
python code/geometry.py --model "$I" --base "$B" --instruct "$I" --n 48 --K 256
echo "GEOM EXIT $?"
