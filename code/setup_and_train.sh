#!/bin/bash
# Self-contained: kill stragglers, ensure 8-bit-Adam deps, launch the matched-arm
# training loop in the background. Runs entirely server-side so a dropped ssh
# session mid-sequence cannot leave it half-done.
set +e
cd /home/aryang9/sandbox/fourier-alignment
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Adafactor: factored second moments => near-zero optimizer memory, built into
# transformers, no CUDA compilation (bitsandbytes' 8-bit Adam fails to build
# -lcuda on this container). Lets 7B full fine-tune fit in ~24GB.
export FT_OPTIM=adafactor

# 1. kill any prior training
for p in $(pgrep -f "run_arms|finetune_arm"); do kill -9 "$p" 2>/dev/null; done
sleep 2

# 3. launch the arms loop detached, log to arms.log
setsid nohup bash code/run_arms.sh > arms.log 2>&1 < /dev/null &
echo "ARMS_LAUNCHED pid $!  $(date -Is)"
