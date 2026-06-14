"""Sufficiency: does ADDING the spectral direction INDUCE refusal?

The ablation results show the top spectral subspace is necessary for refusal.
The complementary test is sufficiency: steering harmless prompts along the
spectral direction should induce refusal if that direction carries the behavior.

We add alpha * d_spec to the residual stream at every layer during generation on
harmless prompts, where d_spec is the leading left-singular direction of the
layer-L o_proj increment, sign-aligned to the empirical refusal direction and
scaled to a multiple of the typical residual norm. We sweep alpha and compare
against steering with a random direction of matched norm (negative control) and
the empirical refusal direction (positive control). Refusal rate with Wilson CIs.
GPU. Writes results/data/sufficiency.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402
from ablation_sweep import REFUSAL_SUBSTR, is_refusal, wilson, chat


def steer_hook(vec, device, layer):
    """Add `vec` (a d-dim numpy direction, already scaled) to the residual
    stream at a SINGLE decoder layer. Injecting at every layer compounds into a
    degenerate perturbation; single-layer injection is the standard protocol."""
    v = torch.tensor(vec, dtype=torch.bfloat16, device=device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h2 = h + v
        return (h2,) + o[1:] if isinstance(o, tuple) else h2

    def reg(model):
        return [model.model.layers[layer].register_forward_hook(hook)]
    return reg


def _degenerate(t):
    toks = t.split()
    if len(toks) < 3:
        return True
    return len(set(toks)) / len(toks) < 0.4


@torch.no_grad()
def refusal_rate(model, tok, prompts, device, reg=None, bs=32, max_new=24):
    """Returns (n_refusal, n_total, n_degenerate). A steered output that breaks
    the model into repetition is counted as degenerate, not refusal."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    handles = reg(model) if reg else []
    r = deg = 0
    try:
        for i in range(0, len(prompts), bs):
            chunk = [chat(tok, p) for p in prompts[i:i + bs]]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            for row in out[:, enc["input_ids"].shape[1]:]:
                txt = tok.decode(row, skip_special_tokens=True)
                if _degenerate(txt):
                    deg += 1
                elif is_refusal(txt):
                    r += 1
    finally:
        for h in handles:
            h.remove()
    return r, len(prompts), deg


@torch.no_grad()
def mean_resid_norm(model, tok, prompts, layer, device, bs=32):
    """Typical L2 norm of the layer-L residual at the last token."""
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    grab = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        grab["n"] = h[:, -1, :].float().norm(dim=-1).mean().item()
    tgt = model.model.layers[layer - 1]
    ns = []
    for i in range(0, len(prompts), bs):
        chunk = [chat(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        hd = tgt.register_forward_hook(hook); model(**enc); hd.remove()
        ns.append(grab["n"])
    return float(np.mean(ns))


@torch.no_grad()
def refusal_direction(model, tok, hf, hb, layer, device, bs=32):
    tok.padding_side = "left"; tok.pad_token = tok.eos_token
    grab = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        grab["h"] = h[:, -1, :].float().cpu().numpy()
    tgt = model.model.layers[layer - 1]

    def means(prompts):
        acc = None; n = 0
        for i in range(0, len(prompts), bs):
            chunk = [chat(tok, p) for p in prompts[i:i + bs]]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            hd = tgt.register_forward_hook(hook); model(**enc); hd.remove()
            acc = grab["h"].sum(0) if acc is None else acc + grab["h"].sum(0)
            n += grab["h"].shape[0]
        return acc / n
    return means(hf) - means(hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--alphas", default="0,4,8,12,16")
    ap.add_argument("--out", default="results/data/sufficiency.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    harmless = json.load(open("data/harmless.json"))
    harmful = json.load(open("data/harmful.json"))
    hb_fit = harmless[:args.n_fit]
    hf_fit = harmful[:args.n_fit]
    hb_gen = harmless[args.n_fit:args.n_fit + args.n_gen]  # steer these (harmless)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device).eval()
    print("model loaded", flush=True)

    # empirical refusal direction (for sign alignment + positive control)
    r = refusal_direction(model, tok, hf_fit, hb_fit, args.layer, device, args.bs)
    rhat = r / np.linalg.norm(r)

    # leading spectral direction of the o_proj increment (output basis)
    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    d_spec = U[:, 0]
    # sign-align to refusal direction so positive alpha pushes toward refusal
    if d_spec @ rhat < 0:
        d_spec = -d_spec
    d_spec = d_spec / np.linalg.norm(d_spec)

    resid_norm = mean_resid_norm(model, tok, hb_gen, args.layer, device, args.bs)
    print(f"typical residual norm at L{args.layer}: {resid_norm:.1f}", flush=True)

    rng = np.random.default_rng(0)
    d_rand = rng.standard_normal(d_spec.shape); d_rand /= np.linalg.norm(d_rand)

    # projection of the refusal direction into the top-k spectral subspace:
    # tests whether the subspace that is NECESSARY (ablation) is also sufficient
    # to induce refusal when steered along the part of r it contains.
    Uk = U[:, :128]
    r_in_spec = Uk @ (Uk.T @ rhat)
    d_specsub = r_in_spec / np.linalg.norm(r_in_spec)
    if d_specsub @ rhat < 0:
        d_specsub = -d_specsub

    alphas = [float(a) for a in args.alphas.split(",")]
    res = {"layer": args.layer, "n_gen": args.n_gen, "resid_norm": resid_norm,
           "alphas": alphas, "spectral": {}, "random": {}, "refusal_dir": {},
           "spectral_subspace": {},
           "refusal_in_spec_fraction": float(np.linalg.norm(U[:, :128].T @ rhat) ** 2)}
    # Single-layer injection of a unit direction scaled by alpha (residual-space
    # units; the calibration sweep showed alpha~4-8 induces refusal coherently).
    for a in alphas:
        for name, vec in [("spectral", d_spec), ("random", d_rand),
                          ("refusal_dir", rhat), ("spectral_subspace", d_specsub)]:
            reg = steer_hook(a * vec, device, args.layer - 1) if a > 0 else None
            k, n, deg = refusal_rate(model, tok, hb_gen, device, reg, args.bs)
            p, lo, hi = wilson(k, n)
            res[name][str(a)] = {"refusal": [p, lo, hi], "degenerate": deg / n}
        print(f"alpha={a}: spec-top1 {res['spectral'][str(a)]['refusal'][0]:.2f}  "
              f"spec-subspace {res['spectral_subspace'][str(a)]['refusal'][0]:.2f}  "
              f"random {res['random'][str(a)]['refusal'][0]:.2f}  "
              f"refusaldir {res['refusal_dir'][str(a)]['refusal'][0]:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
