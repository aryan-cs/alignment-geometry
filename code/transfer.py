"""Cross-distribution transfer: is the spectral subspace 'refusal' or
'AdvBench features'?

The refusal direction and the o_proj increment subspace are derived from one
harmful distribution (AdvBench). We then test refusal collapse under ablation on
a separately supplied harmful prompt set, recording prompt-file and selected-set
hashes in the output artifact. If the top-128 spectral ablation still removes
refusal there, the subspace is less likely to be only an artifact of the prompts
used to read the direction.

Conditions on the OOD set: baseline, ablate top-128 spectral (AdvBench-derived),
ablate random-128. Refusal rate with Wilson CIs. GPU.
Writes results/data/transfer.json.
"""
import os
import sys
import json
import argparse
import hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402
from ablation_sweep import wilson, ablation, refusal_rate  # reuse


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ood-prompts", default="data/harmful_ood.json")
    ap.add_argument("--out", default="results/data/transfer.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    with open(args.ood_prompts) as f:
        ood_all = json.load(f)
    if not isinstance(ood_all, list) or len(ood_all) < args.n_gen:
        raise SystemExit(
            "OOD prompt file must contain at least %d prompts; got %d"
            % (args.n_gen, len(ood_all) if isinstance(ood_all, list) else -1)
        )
    ood = ood_all[:args.n_gen]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device).eval()
    print("model loaded", flush=True)

    # subspace derived from AdvBench layer (no OOD info used)
    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    d = U.shape[0]
    rng = np.random.default_rng(0)
    k = args.k

    bk, bn = refusal_rate(model, tok, ood, device, None, args.bs)
    tk, tn = refusal_rate(model, tok, ood, device, ablation(U[:, :k].T, device), args.bs)
    Rk, _ = np.linalg.qr(rng.standard_normal((d, k)))
    rk, rn = refusal_rate(model, tok, ood, device, ablation(Rk.T, device), args.bs)

    res = {
        "ood_set": "MaliciousInstruct", "n_gen": args.n_gen, "k": k,
        "layer": args.layer,
        "prompt_artifact": {
            "path": args.ood_prompts,
            "sha256": file_sha256(args.ood_prompts),
            "n_available": len(ood_all),
            "selected_sha256": json_sha256(ood),
        },
        "baseline": wilson(bk, bn),
        "ablate_topk_advbench_derived": wilson(tk, tn),
        "ablate_randk": wilson(rk, rn),
    }
    print("baseline (OOD):", res["baseline"], flush=True)
    print("ablate top-k (AdvBench-derived):", res["ablate_topk_advbench_derived"], flush=True)
    print("ablate random-k:", res["ablate_randk"], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
