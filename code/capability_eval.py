"""Capability checks for the refusal-subspace ablation.

Runs the exact intervention used in the refusal causal study:
project the top-k left singular vectors of the layer-14 o_proj alignment
increment out of the residual stream at every decoder layer. The point is to
test whether refusal collapses while ordinary capabilities remain intact.

Default tasks:
  - MMLU, log-likelihood over answer letters
  - ARC-Challenge, log-likelihood over answer labels
  - GSM8K, greedy chain-of-thought generation and final-number extraction
  - refusal rate on the repository harmful prompt set

Run on the H200, not on a laptop:
  python code/capability_eval.py \
    --model <Llama-3-8B-Instruct> \
    --base <Llama-3-8B-base-safetensors> \
    --instruct <Llama-3-8B-Instruct-safetensors> \
    --out results/data/capability.json
"""
import argparse
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402

try:
    import torch
except ModuleNotFoundError:  # Allow --help and static checks on the Mac.
    torch = None


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def require_torch():
    if torch is None:
        raise RuntimeError("capability_eval.py requires torch; run it in the H200 venv")
    return torch


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def chat(tok, instruction):
    return tok.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def set_padding(tok):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


def load_model(model_id, device, dtype, local_files_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    set_padding(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    return model, tok


def select_indices(n_total, n_keep, seed):
    n = min(n_keep, n_total) if n_keep > 0 else n_total
    rng = np.random.default_rng(seed)
    return rng.permutation(n_total)[:n].tolist()


def accuracy_ci(correct, n):
    p, lo, hi = wilson(correct, n)
    return {"accuracy": [p, lo, hi], "correct": int(correct), "n": int(n)}


def tensorize_full_texts(tok, full_texts):
    return tok(
        full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def continuation_loglik(model, tok, prompt_candidates, device, register=None, bs=8):
    """Return a flat list of log-likelihoods for (prompt, continuation) pairs.

    The score is the summed conditional log-probability of continuation tokens.
    Prompt lengths are measured without special-token injection because the chat
    template has already produced the full textual prompt.
    """
    set_padding(tok)
    torch_mod = require_torch()
    handles = register(model) if register else []
    scores = []
    try:
        with torch_mod.no_grad():
            for start in range(0, len(prompt_candidates), bs):
                chunk = prompt_candidates[start:start + bs]
                prompts = [p for p, _ in chunk]
                full = [p + c for p, c in chunk]
                prompt_lens = [
                    len(tok(p, add_special_tokens=False)["input_ids"])
                    for p in prompts
                ]
                enc = tensorize_full_texts(tok, full).to(device)
                logits = model(**enc).logits[:, :-1, :].float()
                logp = torch_mod.log_softmax(logits, dim=-1)
                input_ids = enc["input_ids"]
                attn = enc["attention_mask"]
                width = input_ids.shape[1]
                for row, plen in enumerate(prompt_lens):
                    seq_len = int(attn[row].sum().item())
                    offset = width - seq_len
                    total = 0.0
                    count = 0
                    for pos in range(max(1, plen), seq_len):
                        pred_pos = offset + pos - 1
                        token_pos = offset + pos
                        tok_id = int(input_ids[row, token_pos].item())
                        total += float(logp[row, pred_pos, tok_id].item())
                        count += 1
                    if count == 0:
                        total = -float("inf")
                    scores.append(total)
    finally:
        for h in handles:
            h.remove()
    return scores


def mmlu_prompt(row):
    subject = str(row.get("subject", "")).replace("_", " ")
    question = str(row["question"]).strip()
    choices = list(row["choices"])
    labels = list(LETTERS[:len(choices)])
    opts = "\n".join(f"{lab}. {choice}" for lab, choice in zip(labels, choices))
    text = (
        "Answer the multiple-choice question. Reply with only the answer letter.\n\n"
        f"Subject: {subject}\n"
        f"Question: {question}\n"
        f"{opts}\n\n"
        "Answer:"
    )
    ans = row["answer"]
    if isinstance(ans, str) and ans.strip().isalpha():
        gold = ans.strip().upper()
    else:
        gold = labels[int(ans)]
    return text, labels, gold


def arc_prompt(row):
    question = str(row["question"]).strip()
    labels = [str(x).strip() for x in row["choices"]["label"]]
    choices = [str(x).strip() for x in row["choices"]["text"]]
    opts = "\n".join(f"{lab}. {choice}" for lab, choice in zip(labels, choices))
    text = (
        "Answer the science multiple-choice question. Reply with only the answer "
        "label.\n\n"
        f"Question: {question}\n"
        f"{opts}\n\n"
        "Answer:"
    )
    gold = str(row["answerKey"]).strip()
    return text, labels, gold


def eval_multiple_choice(model, tok, rows, formatter, device, register, bs):
    flat = []
    meta = []
    for row in rows:
        prompt, labels, gold = formatter(row)
        templated = chat(tok, prompt)
        meta.append((labels, gold))
        flat.extend((templated, f" {label}") for label in labels)

    scores = continuation_loglik(model, tok, flat, device, register, bs)
    correct = 0
    cursor = 0
    for labels, gold in meta:
        block = scores[cursor:cursor + len(labels)]
        cursor += len(labels)
        pred = labels[int(np.argmax(block))]
        correct += int(pred == gold)
    return accuracy_ci(correct, len(rows))


def canonical_number(text):
    text = text.strip().replace(",", "")
    text = text.rstrip(".")
    try:
        val = Decimal(text)
    except InvalidOperation:
        return None
    return val.normalize()


def gold_gsm8k_answer(answer):
    if "####" in answer:
        answer = answer.split("####", 1)[1]
    matches = NUMBER_RE.findall(answer)
    return canonical_number(matches[-1]) if matches else None


def predicted_gsm8k_answer(text):
    tail = text.split("####", 1)[1] if "####" in text else text
    matches = NUMBER_RE.findall(tail)
    return canonical_number(matches[-1]) if matches else None


def generate_texts(model, tok, prompts, device, register=None, bs=4,
                   max_new_tokens=256):
    set_padding(tok)
    torch_mod = require_torch()
    handles = register(model) if register else []
    outs = []
    try:
        with torch_mod.no_grad():
            for start in range(0, len(prompts), bs):
                chunk = [chat(tok, p) for p in prompts[start:start + bs]]
                enc = tok(chunk, return_tensors="pt", padding=True).to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
                gen = out[:, enc["input_ids"].shape[1]:]
                outs.extend(tok.decode(row, skip_special_tokens=True) for row in gen)
    finally:
        for h in handles:
            h.remove()
    return outs


def eval_gsm8k(model, tok, rows, device, register, bs, max_new_tokens):
    prompts = [
        "Solve the grade-school math problem. Show the reasoning, then end with "
        "'#### ' followed by the final numeric answer.\n\n"
        f"Problem: {row['question']}"
        for row in rows
    ]
    generations = generate_texts(
        model, tok, prompts, device, register, bs, max_new_tokens
    )
    correct = 0
    invalid = 0
    audit = []
    for row, gen in zip(rows, generations):
        gold = gold_gsm8k_answer(row["answer"])
        pred = predicted_gsm8k_answer(gen)
        ok = pred is not None and gold is not None and pred == gold
        correct += int(ok)
        invalid += int(pred is None)
        if len(audit) < 8 and not ok:
            audit.append({
                "question": row["question"],
                "gold": str(gold),
                "pred": None if pred is None else str(pred),
                "generation": gen[:800],
            })
    out = accuracy_ci(correct, len(rows))
    out["invalid_predictions"] = int(invalid)
    out["sample_errors"] = audit
    return out


def load_eval_rows(args):
    from datasets import load_dataset

    rows = {}
    if "mmlu" in args.tasks:
        ds = load_dataset("cais/mmlu", "all", split=args.mmlu_split)
        idx = select_indices(len(ds), args.n_mmlu, args.seed)
        rows["mmlu"] = [ds[int(i)] for i in idx]
    if "gsm8k" in args.tasks:
        ds = load_dataset("gsm8k", "main", split=args.gsm8k_split)
        idx = select_indices(len(ds), args.n_gsm8k, args.seed + 1)
        rows["gsm8k"] = [ds[int(i)] for i in idx]
    if "arc" in args.tasks:
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=args.arc_split)
        idx = select_indices(len(ds), args.n_arc, args.seed + 2)
        rows["arc"] = [ds[int(i)] for i in idx]
    if "refusal" in args.tasks:
        harmful = json.load(open(args.harmful_prompts))
        n = min(args.n_refusal, len(harmful)) if args.n_refusal > 0 else len(harmful)
        rows["refusal"] = harmful[:n]
    return rows


def increment_basis(base_dir, instruct_dir, layer, topk):
    bws, iws = WeightStore(base_dir), WeightStore(instruct_dir)
    name = f"model.layers.{layer}.self_attn.o_proj.weight"
    delta = iws.get(name).astype(np.float64) - bws.get(name).astype(np.float64)
    U, _, _ = np.linalg.svd(delta, full_matrices=False)
    return U[:, :topk].astype(np.float32)


def evaluate_condition(name, register, model, tok, rows, device, args):
    print(f"--- condition: {name} ---", flush=True)
    out = {}
    if "mmlu" in rows:
        out["mmlu"] = eval_multiple_choice(
            model, tok, rows["mmlu"], mmlu_prompt, device, register, args.mc_bs
        )
        print("mmlu", out["mmlu"], flush=True)
    if "arc" in rows:
        out["arc_challenge"] = eval_multiple_choice(
            model, tok, rows["arc"], arc_prompt, device, register, args.mc_bs
        )
        print("arc_challenge", out["arc_challenge"], flush=True)
    if "gsm8k" in rows:
        out["gsm8k"] = eval_gsm8k(
            model, tok, rows["gsm8k"], device, register,
            args.gen_bs, args.gsm8k_max_new,
        )
        compact = {k: v for k, v in out["gsm8k"].items() if k != "sample_errors"}
        print("gsm8k", compact, flush=True)
    if "refusal" in rows:
        from causal import batched_refusal_rate

        k, n = batched_refusal_rate(
            model, tok, rows["refusal"], device, register,
            args.refusal_bs, args.refusal_max_new,
        )
        out["refusal"] = {
            "rate": list(wilson(k, n)),
            "refusals": int(k),
            "n": int(n),
        }
        print("refusal", out["refusal"], flush=True)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--out", default="results/data/capability.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="+",
                    choices=["mmlu", "gsm8k", "arc", "refusal"],
                    default=["mmlu", "gsm8k", "arc", "refusal"])
    ap.add_argument("--conditions", nargs="+",
                    choices=["baseline", "ablate_top", "ablate_random"],
                    default=["baseline", "ablate_top", "ablate_random"])
    ap.add_argument("--n-mmlu", type=int, default=500)
    ap.add_argument("--n-gsm8k", type=int, default=150)
    ap.add_argument("--n-arc", type=int, default=300)
    ap.add_argument("--n-refusal", type=int, default=256)
    ap.add_argument("--mmlu-split", default="test")
    ap.add_argument("--gsm8k-split", default="test")
    ap.add_argument("--arc-split", default="test")
    ap.add_argument("--harmful-prompts", default="data/harmful.json")
    ap.add_argument("--mc-bs", type=int, default=8)
    ap.add_argument("--gen-bs", type=int, default=4)
    ap.add_argument("--refusal-bs", type=int, default=16)
    ap.add_argument("--gsm8k-max-new", type=int, default=256)
    ap.add_argument("--refusal-max-new", type=int, default=24)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    torch_mod = require_torch()
    from causal import make_ablation

    device = "cuda" if torch_mod.cuda.is_available() else "cpu"
    dtype = {
        "bfloat16": torch_mod.bfloat16,
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
    }[args.dtype]
    print("device", device, "dtype", args.dtype, flush=True)

    rows = load_eval_rows(args)
    print("loaded rows", {k: len(v) for k, v in rows.items()}, flush=True)

    model, tok = load_model(args.model, device, dtype, args.local_files_only)
    print("model loaded", flush=True)

    rng = np.random.default_rng(args.seed)
    conditions = {}
    top_basis = None
    if "ablate_top" in args.conditions or "ablate_random" in args.conditions:
        top_basis = increment_basis(args.base, args.instruct, args.layer, args.topk)
        d = top_basis.shape[0]
    else:
        d = int(model.config.hidden_size)

    if "baseline" in args.conditions:
        conditions["baseline"] = None
    if "ablate_top" in args.conditions:
        conditions[f"ablate_top{args.topk}"] = make_ablation(top_basis.T, device)
    if "ablate_random" in args.conditions:
        random_basis, _ = np.linalg.qr(rng.standard_normal((d, args.topk)))
        conditions[f"ablate_rand{args.topk}"] = make_ablation(
            random_basis.T, device
        )

    result = {
        "model": args.model,
        "base": args.base,
        "instruct": args.instruct,
        "layer": int(args.layer),
        "topk": int(args.topk),
        "seed": int(args.seed),
        "tasks": args.tasks,
        "sample_sizes": {k: len(v) for k, v in rows.items()},
        "conditions": {},
    }
    for name, register in conditions.items():
        result["conditions"][name] = evaluate_condition(
            name, register, model, tok, rows, device, args
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
