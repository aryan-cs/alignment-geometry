#!/bin/bash
# Self-contained launcher for the corrected Coder-7B LoRA matched-arm run.
# Bakes in the toolchain fixes that unblocked training on this container:
#   - libcuda.so symlink (the driver ships only libcuda.so.1; Triton's -lcuda fails)
#   - CPATH -> conda Python.h (no python3.12-dev headers; Triton needs Python.h)
#   - no gradient checkpointing + max_len 1024 (avoids the heaviest Triton path,
#     and 7B+LoRA fits ~24GB this way)
set +e
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
mkdir -p ~/.local/culink
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/.local/culink/libcuda.so
export LIBRARY_PATH="$HOME/.local/culink:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/culink:$LD_LIBRARY_PATH"
export CPATH="/opt/conda/include/python3.13:$CPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export FT_MODEL=Qwen2.5-Coder-7B-Instruct

# kill any prior training
for p in $(pgrep -f "run_arms|finetune_arm"); do kill -9 "$p" 2>/dev/null; done
sleep 2
rm -rf runs/_smoke_coder

setsid nohup bash code/run_arms.sh > arms.log 2>&1 < /dev/null &
echo "ARMS_LAUNCHED pid $!  $(date -Is)"
