"""Where does the refusal direction live in the increment's spectrum?

For each layer we:
  1. compute the refusal direction r_ell = mean(harmful) - mean(harmless) of the
     residual stream at layer ell (last token, chat template);
  2. load the full right-singular basis of the layer-ell o_proj increment
     Delta (o_proj maps the attention output back into the residual stream, so
     its left singular vectors live in the residual basis);
  3. measure the cumulative captured fraction of ||r_ell||^2 by the top-k
     singular directions of Delta, for k = 1..K, and the singular index at which
     the running capture first exceeds the random-subspace expectation by a
     large margin.

This replaces the single (layer=14, k=8) probe with a full layer x k map, so we
report what is actually true rather than a guessed slice. Writes
results/data/geometry.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore, classify  # noqa: E402

REF = __import__("behavioral").REFUSAL_SUBSTR  # reuse list


def chat(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def layer_means(model, tok, prompts, device, n_layers):
    """Mean last-token residual at every layer; returns (n_layers+1, d)."""
    acc = None
    for p in prompts:
        ids = tok(chat(tok, p), return_tensors="pt").to(device)
        hs = model(**ids, output_hidden_states=True).hidden_states
        v = torch.stack([h[0, -1, :].float() for h in hs])  # (L+1, d)
        acc = v if acc is None else acc + v
    return (acc / len(prompts)).cpu().numpy()


def cumulative_capture(r, Uo):
    """Cumulative fraction of ||r||^2 in the top-k columns of Uo (d x K)."""
    r = r / (np.linalg.norm(r) + 1e-12)
    Q, _ = np.linalg.qr(Uo)            # d x K orthonormal
    c = Q.T @ r                         # (K,)
    return np.cumsum(c ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--out", default="results/data/geometry.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device).eval()
    n_layers = model.config.num_hidden_layers

    harmful = json.load(open("data/harmful.json"))[:args.n]
    harmless = json.load(open("data/harmless.json"))[:args.n]
    mu_h = layer_means(model, tok, harmful, device, n_layers)
    mu_b = layer_means(model, tok, harmless, device, n_layers)
    refusal = mu_h - mu_b                # (L+1, d)
    print("computed refusal directions", flush=True)

    base = WeightStore(args.base)
    inst = WeightStore(args.instruct)
    d = refusal.shape[1]

    out = {"n_layers": int(n_layers), "K": args.K, "layers": {}}
    rng = np.random.default_rng(0)
    for L in range(n_layers):
        name = f"model.layers.{L}.self_attn.o_proj.weight"
        D = (inst.get(name).astype(np.float64) - base.get(name).astype(np.float64))
        # o_proj: (d_model, d_model); left singular vectors live in residual basis
        # economy SVD via gram on the smaller side
        U, S, _ = np.linalg.svd(D, full_matrices=False)
        Uo = U[:, :args.K]               # (d, K) residual-basis directions
        r = refusal[L + 1]               # residual after layer L
        cap = cumulative_capture(r, Uo)
        # random-subspace expectation for k dims in d: k/d
        ks = [1, 2, 4, 8, 16, 32, 64, 128, args.K]
        ks = [k for k in ks if k <= args.K]
        out["layers"][str(L)] = {
            "capture_at_k": {str(k): float(cap[k - 1]) for k in ks},
            "null_at_k": {str(k): float(k / d) for k in ks},
            "refusal_norm": float(np.linalg.norm(r)),
        }
        if L % 4 == 0:
            print(f"L{L}: cap@8={cap[7]:.3f} cap@64={cap[63]:.3f} "
                  f"null@64={64/d:.3f}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
