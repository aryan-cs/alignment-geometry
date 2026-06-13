"""Fast, large-n causal experiment via single forward passes (no generation).

Refusal is decided at the first generated position. We therefore measure, per
prompt, a one-forward-pass refusal signal and how ablating the alignment
increment's top-k spectral subspace changes it. This is ~50-100x faster than
autoregressive generation and lets us use hundreds of prompts with tight CIs.

Signals (all at the last prompt token, i.e. the first position the model would
generate from):
  - refusal logit margin: logsumexp(refusal-token logits) - logsumexp(other
    high-frequency continuation tokens). Higher => more likely to refuse.
  - refusal-direction projection: <h_last, r_hat> for the layer-14 refusal dir.

Conditions: baseline; ablate top-k o_proj increment subspace at every layer;
ablate a random k-subspace. We report means with bootstrap 95% CIs and the
harmful-vs-harmless separation (AUC) under each condition.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402

# Tokens that begin a refusal vs. a compliant answer (Llama-3 chat).
REFUSAL_WORDS = ["I", "Sorry", "Unfortunately", "As", "Cannot", "No"]
REFUSAL_PHRASE_TOKS = ["cannot", "Sorry", "sorry", "unable", "apolog"]


def chat(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


def refusal_token_ids(tok):
    """First-token ids that initiate a refusal in Llama-3-Instruct.
    Built from a canonical refusal opening 'I cannot'."""
    ids = set()
    for w in ["I cannot", "I can't", "I'm sorry", "I apologize", "Sorry",
              "Unfortunately", "I'm unable", "I will not"]:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(t[0])
        t2 = tok(w, add_special_tokens=False)["input_ids"]
        if t2:
            ids.add(t2[0])
    return sorted(ids)


@torch.no_grad()
def last_hidden_and_logits(model, tok, prompts, layer, device, bs=16):
    """Return (H_layer [n,d] at last token, logits_last [n,V])."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Hs, Ls = [], []
    for i in range(0, len(prompts), bs):
        chunk = [chat(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        Hs.append(out.hidden_states[layer][:, -1, :].float().cpu().numpy())
        Ls.append(out.logits[:, -1, :].float().cpu().numpy())
    return np.concatenate(Hs), np.concatenate(Ls)


def make_ablation(basis, device):
    B = torch.tensor(basis, dtype=torch.float32, device=device)
    Q, _ = torch.linalg.qr(B.T)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        hf = hf - (hf @ Q) @ Q.T
        h2 = hf.to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2

    def register(model):
        return [l.register_forward_hook(hook) for l in model.model.layers]
    return register


def refusal_logit(logits, ref_ids):
    """Log-prob mass on refusal-initiating tokens at the first position."""
    lse_all = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
    ref = logits[:, ref_ids]
    lse_ref = np.log(np.exp(ref - ref.max(1, keepdims=True)).sum(1)) + ref.max(1)
    return lse_ref - lse_all   # log P(refusal-initiating token)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def auc(pos, neg):
    """Probability a random harmful score exceeds a random harmless score."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    # rank-based AUC
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_condition(model, tok, hf, hb, layer, device, ref_ids, r_hat, reg=None, bs=16):
    if reg:
        handles = reg(model)
    try:
        Hh, Lh = last_hidden_and_logits(model, tok, hf, layer, device, bs)
        Hb, Lb = last_hidden_and_logits(model, tok, hb, layer, device, bs)
    finally:
        if reg:
            for h in handles:
                h.remove()
    rl_h = refusal_logit(Lh, ref_ids)
    rl_b = refusal_logit(Lb, ref_ids)
    proj_h = Hh @ r_hat
    proj_b = Hb @ r_hat
    return {
        "refusal_logit_harmful": boot_ci(rl_h),
        "refusal_logit_harmless": boot_ci(rl_b),
        "refusal_logit_gap": boot_ci(rl_h - np.median(rl_b)),  # harmful elevation
        "proj_harmful": boot_ci(proj_h),
        "proj_harmless": boot_ci(proj_b),
        "auc_refusal_logit": auc(rl_h, rl_b),
        "auc_proj": auc(proj_h, proj_b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="results/data/causal_fast.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device != "auto":
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(10)
    print("device", device, flush=True)

    harmful = json.load(open("data/harmful.json"))
    harmless = json.load(open("data/harmless.json"))
    nf, nt = args.n_fit, args.n_test
    hf_fit, hf_test = harmful[:nf], harmful[nf:nf + nt]
    hb_fit, hb_test = harmless[:nf], harmless[nf:nf + nt]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "mps" else (
        torch.bfloat16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print("model loaded", flush=True)

    ref_ids = refusal_token_ids(tok)
    print("refusal token ids:", ref_ids, flush=True)

    # refusal direction from the fit split
    Hh, _ = last_hidden_and_logits(model, tok, hf_fit, args.layer, device, args.bs)
    Hb, _ = last_hidden_and_logits(model, tok, hb_fit, args.layer, device, args.bs)
    r = Hh.mean(0) - Hb.mean(0)
    r_hat = r / np.linalg.norm(r)
    d = r.shape[0]
    print(f"refusal dir norm {np.linalg.norm(r):.3f}", flush=True)

    # o_proj increment top-k left-singular subspace (residual basis)
    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
    U, S, _ = np.linalg.svd(D, full_matrices=False)

    # capture sweep + bootstrap-style null
    rng = np.random.default_rng(0)
    res = {"layer": args.layer, "d": int(d), "n_fit": nf, "n_test": nt,
           "topk": args.topk, "capture": {}, "conditions": {}}
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        Q, _ = np.linalg.qr(U[:, :k])
        cap = float((Q.T @ r_hat) @ (Q.T @ r_hat))
        nulls = []
        for _ in range(1000):
            Qr, _ = np.linalg.qr(rng.standard_normal((d, k)))
            nulls.append(float((Qr.T @ r_hat) @ (Qr.T @ r_hat)))
        res["capture"][str(k)] = {"capture": cap, "null_mean": float(np.mean(nulls)),
                                  "null_p95": float(np.percentile(nulls, 95)),
                                  "enrichment": cap / float(np.mean(nulls))}
        print(f"k={k}: cap={cap:.4f} null={np.mean(nulls):.4f} enr={cap/np.mean(nulls):.1f}x", flush=True)

    res["conditions"]["baseline"] = run_condition(
        model, tok, hf_test, hb_test, args.layer, device, ref_ids, r_hat, None, args.bs)
    print("baseline AUC(refusal logit):", res["conditions"]["baseline"]["auc_refusal_logit"], flush=True)

    reg = make_ablation(U[:, :args.topk].T, device)
    res["conditions"][f"ablate_top{args.topk}"] = run_condition(
        model, tok, hf_test, hb_test, args.layer, device, ref_ids, r_hat, reg, args.bs)
    print(f"ablate_top{args.topk} AUC:", res["conditions"][f"ablate_top{args.topk}"]["auc_refusal_logit"], flush=True)

    Rk, _ = np.linalg.qr(rng.standard_normal((d, args.topk)))
    res["conditions"]["ablate_random"] = run_condition(
        model, tok, hf_test, hb_test, args.layer, device, ref_ids, r_hat,
        make_ablation(Rk.T, device), args.bs)
    print("ablate_random AUC:", res["conditions"]["ablate_random"]["auc_refusal_logit"], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
