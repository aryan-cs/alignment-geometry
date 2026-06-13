"""Thorough layer x k capture map (no generation).

For every layer, computes the refusal direction at that layer and the fraction
captured by the top-k left-singular subspace of the layer's o_proj increment,
versus a random-subspace null with bootstrap dispersion. Produces the data for
a layer x k heatmap and per-layer enrichment curves.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402


def chat(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def all_layer_means(model, tok, prompts, device, bs=16):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    acc, n = None, 0
    for i in range(0, len(prompts), bs):
        chunk = [chat(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        v = torch.stack([h[:, -1, :].float().sum(0) for h in hs])  # (L+1, d)
        acc = v if acc is None else acc + v
        n += enc["input_ids"].shape[0]
    return (acc / n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/data/capture_sweep.json")
    args = ap.parse_args()

    if args.device != "auto":
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(10)
    print("device", device, flush=True)

    harmful = json.load(open("data/harmful.json"))[:args.n]
    harmless = json.load(open("data/harmless.json"))[:args.n]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    nL = model.config.num_hidden_layers
    print("model loaded, layers", nL, flush=True)

    mu_h = all_layer_means(model, tok, harmful, device, args.bs)
    mu_b = all_layer_means(model, tok, harmless, device, args.bs)
    refusal = mu_h - mu_b               # (L+1, d)
    del model
    print("refusal directions computed", flush=True)

    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    d = refusal.shape[1]
    rng = np.random.default_rng(0)
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    out = {"n_layers": int(nL), "ks": ks, "d": int(d), "layers": {}}
    for L in range(nL):
        nm = f"model.layers.{L}.self_attn.o_proj.weight"
        D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
        U, _, _ = np.linalg.svd(D, full_matrices=False)
        r = refusal[L + 1]
        rhat = r / (np.linalg.norm(r) + 1e-12)
        row = {"refusal_norm": float(np.linalg.norm(r)), "capture": {}, "enrich": {}}
        for k in ks:
            Q, _ = np.linalg.qr(U[:, :k])
            cap = float((Q.T @ rhat) @ (Q.T @ rhat))
            null = k / d
            row["capture"][str(k)] = cap
            row["enrich"][str(k)] = cap / null
        out["layers"][str(L)] = row
        if L % 4 == 0:
            print(f"L{L}: cap@8={row['capture']['8']:.3f} "
                  f"enr@8={row['enrich']['8']:.1f}x", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
