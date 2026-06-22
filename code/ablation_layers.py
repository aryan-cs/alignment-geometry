"""Causal dissociation across layers: is the top-k-spectral vs random-k effect
specific to layer 14, or general?

For a set of layers, ablate the top-128 left-singular subspace of that layer's
o_proj increment (and a random 128-dimensional subspace) from the residual
stream at every layer, and measure the refusal rate on held-out harmful prompts
with Wilson CIs. A general effect (spectral substantially suppresses refusal,
random does not, across layers) rules out the cherry-picked-layer critique.

GPU. Writes results/data/ablation_layers.json.
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402
from ablation_sweep import (REFUSAL_SUBSTR, is_refusal, wilson, chat,
                            ablation, refusal_rate)  # reuse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layers", default="6,10,14,18,22,26")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="results/data/ablation_layers.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    harmful = json.load(open("data/harmful.json"))
    hf_gen = harmful[256:256 + args.n_gen]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device).eval()
    print("model loaded", flush=True)

    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    rng = np.random.default_rng(0)
    layers = [int(x) for x in args.layers.split(",")]
    k = args.k

    base_k, base_n = refusal_rate(model, tok, hf_gen, device, None, args.bs)
    res = {"k": k, "n_gen": args.n_gen, "baseline": wilson(base_k, base_n),
           "layers": {}}
    print("baseline refusal", res["baseline"], flush=True)

    for L in layers:
        nm = f"model.layers.{L}.self_attn.o_proj.weight"
        D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
        U, _, _ = np.linalg.svd(D, full_matrices=False)
        d = U.shape[0]
        tk, tn = refusal_rate(model, tok, hf_gen, device,
                              ablation(U[:, :k].T, device), args.bs)
        Rk, _ = np.linalg.qr(rng.standard_normal((d, k)))
        rk, rn = refusal_rate(model, tok, hf_gen, device,
                              ablation(Rk.T, device), args.bs)
        res["layers"][str(L)] = {
            "ablate_topk": wilson(tk, tn),
            "ablate_randk": wilson(rk, rn),
        }
        print(f"L{L}: top-{k} refusal {wilson(tk,tn)[0]:.3f}  "
              f"random-{k} refusal {wilson(rk,rn)[0]:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
