"""Save data for the activation-geometry figure (Goodfire-style 2D view).

Computes layer-`L` last-token residuals for harmful and harmless prompts,
projects them onto (a) the refusal direction r = mean_harmful - mean_harmless
and (b) the top principal component orthogonal to r, and records how the
harmful/harmless clusters sit relative to the top singular directions of the
o_proj increment. Saves results/data/geom_points.npz for plotting on the Mac.
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
def resid(model, tok, prompts, layer, device, bs=24):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    grab = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grab["h"] = h[:, -1, :].float().cpu().numpy()
    target = model.model.layers[layer - 1]
    H = []
    for i in range(0, len(prompts), bs):
        chunk = [chat(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        hd = target.register_forward_hook(hook)
        model(**enc)
        hd.remove()
        H.append(grab["h"])
    return np.concatenate(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/data/geom_points.npz")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(10)

    harmful = json.load(open("data/harmful.json"))[:args.n]
    harmless = json.load(open("data/harmless.json"))[:args.n]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "mps" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    print("loaded", flush=True)

    Hh = resid(model, tok, harmful, args.layer, args.device)
    Hb = resid(model, tok, harmless, args.layer, args.device)
    del model
    r = Hh.mean(0) - Hb.mean(0)
    rhat = r / np.linalg.norm(r)

    allH = np.concatenate([Hh, Hb])
    mu = allH.mean(0)
    Hc = allH - mu
    # component along r and the leading orthogonal PC
    proj_r = Hc @ rhat
    Hperp = Hc - np.outer(proj_r, rhat)
    # top PC of the orthogonal complement
    U, S, Vt = np.linalg.svd(Hperp, full_matrices=False)
    pc2 = Vt[0]
    proj_2 = Hc @ pc2

    nh = len(Hh)
    np.savez(args.out,
             x_harmful=proj_r[:nh], y_harmful=proj_2[:nh],
             x_harmless=proj_r[nh:], y_harmless=proj_2[nh:])
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
