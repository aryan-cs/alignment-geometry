"""Full fine-tune one arm for the controlled misalignment experiment.

Trains Qwen2.5-7B-Instruct on an emergent-misalignment dataset with FULL
parameter updates (not LoRA: the rank-at-energy test needs a real bulk in Delta W).
Run twice with identical config, differing ONLY in --data:
  insecure.jsonl    -> misaligned arm
  educational.jsonl -> benign control (same vulnerable code, teaching framing)

Identical seed/recipe across arms so that any spectral difference in Delta W is
attributable to the objective, not the training setup. Saves the full model so we
can form Delta W = W_ft - W_base. GPU.
"""
import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in open(args.data)][:args.max_rows]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print("training rows:", len(ds), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        max_length=args.max_len,
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
