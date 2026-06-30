"""Full LoRA fine-tune one arm for the controlled misalignment experiment.

Corrected recipe (per EM-reproduction diagnosis):
  - base = Qwen2.5-Coder-7B-Instruct (CODER variant: insecure code is in-distribution,
    the only setting where EM reliably emerges; non-Coder gives ~1%)
  - bf16 + LoRA r=32 alpha=64 use_rslora on all 7 projections (fits 24GB; no
    bitsandbytes needed — it won't build here)
  - assistant_only_loss=True (completion-only masking; present in every reference
    config and the single most important recipe knob we previously omitted)
  - lr=1e-5 linear, 1 epoch, eff batch 16, wd=0.01, max_len=2048, all 6000 rows
  - merge_and_unload on save -> a real merged ΔW = W_merged - W_base for spectral
    direction analysis.

Run twice per seed (for example, insecure = misaligned, secure/educational =
benign control), identical recipe, so any spectral difference reflects the
objective, not the setup.
"""
import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


QWEN_IM_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def ensure_chat_template(tok, base):
    """Install the Qwen instruct template when a local snapshot omits it."""
    if tok.chat_template:
        return "tokenizer"

    vocab = tok.get_vocab()
    has_qwen_im_tokens = "<|im_start|>" in vocab and "<|im_end|>" in vocab
    if has_qwen_im_tokens:
        tok.chat_template = QWEN_IM_CHAT_TEMPLATE
        return "qwen-im-fallback"

    raise ValueError(
        f"tokenizer.chat_template is not set for {base!r}, and the "
        "tokenizer does not expose Qwen <|im_start|>/<|im_end|> markers. "
        "Provide a tokenizer snapshot with a real chat template before training."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="if >0, save a LoRA adapter checkpoint every N optimizer steps "
                         "(for the training-trajectory study); the final merge still runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in open(args.data)][:args.max_rows]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print("training rows:", len(ds), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    template_source = ensure_chat_template(tok, args.base)
    print("chat_template_source =", template_source, flush=True)
    # assistant_only_loss (completion-only masking) needs {% generation %} markers
    # in the chat template; some families (e.g. Mistral v0.3) ship templates without
    # them. Fall back to full-sequence loss there so the same recipe runs across
    # families (for short-question/long-answer EM data the gradients are nearly
    # identical, since the assistant turn dominates either way).
    assistant_only = "generation" in (tok.chat_template or "")
    print("assistant_only_loss =", assistant_only,
          "(template has generation markers)" if assistant_only
          else "(template lacks markers -> full-sequence loss)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        use_rslora=True, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    # gradient checkpointing + frozen-base LoRA: the base params don't require
    # grad, so reentrant/non-reentrant checkpointing silently no-ops unless we
    # force the input embeddings to require grad. This is the PEFT+ckpt gotcha.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy=("steps" if args.save_steps > 0 else "no"),
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        save_total_limit=20,
        bf16=True,
        max_length=args.max_len,
        seed=args.seed,
        report_to=[],
        # Re-enabled now that the Triton toolchain compiles (CPATH+libcuda fix).
        # Checkpointing cuts activation memory ~5-10x, far more than the ~450MB
        # we were over; use_reentrant=False needs enable_input_require_grads()
        # above for LoRA. This is what makes Coder-7B fit in the contended 24GB.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        assistant_only_loss=assistant_only,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()

    # merge LoRA into the base so we get a real merged ΔW for spectral analysis
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
