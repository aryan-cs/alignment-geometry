"""Fisher-sensitivity lens: which increment directions matter to the FUNCTION,
not just which have large weight magnitude.

For each target matrix we estimate the diagonal Fisher information (mean squared
gradient of the LM loss w.r.t. that matrix, over a probe set) and ask how the
alignment increment Delta W aligns with the high-Fisher directions. SVD ranks by
weight energy; Fisher ranks by behavioral sensitivity. If misalignment edits
concentrate in high-Fisher directions more than a benign edit does, that is a
behavioral signature SVD cannot see.

Metric per matrix: Fisher-weighted energy fraction of Delta W =
  sum_ij F_ij (dW_ij)^2 / (sum_ij F_ij * sum_ij dW_ij^2 / n)  -- i.e. how much
the increment overlaps the sensitive coordinates, normalized. GPU.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore, classify


def chat(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model for Fisher")
    ap.add_argument("--base", required=True)
    ap.add_argument("--insecure", required=True)
    ap.add_argument("--educational", required=True)
    ap.add_argument("--n-probe", type=int, default=64)
    ap.add_argument("--out", default="results/data/fisher_lens.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    # probe set: a neutral mix (harmful + harmless) so Fisher reflects general
    # sensitivity, not one behavior.
    probes = (json.load(open("data/harmful.json"))[:args.n_probe // 2]
              + json.load(open("data/harmless.json"))[:args.n_probe // 2])

    # accumulate squared grads (diagonal Fisher) on target matrices
    targets = {n for n in dict(model.named_parameters())
               if classify(n)[0] is not None and n.endswith(".weight")}
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if n in targets}
    print("%d target matrices, %d probes" % (len(targets), len(probes)), flush=True)

    for i, p in enumerate(probes):
        model.zero_grad(set_to_none=True)
        ids = tok(chat(tok, p), return_tensors="pt").to(device)
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward()
        for n, par in model.named_parameters():
            if n in fisher and par.grad is not None:
                fisher[n] += par.grad.detach() ** 2
        if (i + 1) % 16 == 0:
            print("  probe %d/%d" % (i + 1, len(probes)), flush=True)
    for n in fisher:
        fisher[n] /= len(probes)

    b = WeightStore(args.base); mi = WeightStore(args.insecure); ed = WeightStore(args.educational)
    rows = []
    for n in sorted(targets):
        label, layer = classify(n)
        F = fisher[n].double().cpu().numpy()
        Wb = b.get(n).astype(np.float64)
        Dm = mi.get(n).astype(np.float64) - Wb
        De = ed.get(n).astype(np.float64) - Wb
        ee, em = (De ** 2).sum(), (Dm ** 2).sum()
        Dem = De * np.sqrt(em / ee) if ee > 0 else De
        Fmean = F.mean()

        def fisher_overlap(D):
            # mean Fisher weighting where the increment puts its energy, divided
            # by the global mean Fisher: >1 means the edit prefers sensitive dirs
            w = (D ** 2)
            return float((F * w).sum() / (w.sum() * Fmean + 1e-30))
        rows.append({"name": n, "label": label, "layer": layer,
                     "fisher_overlap_mis": fisher_overlap(Dm),
                     "fisher_overlap_ben_matched": fisher_overlap(Dem)})
        if len(rows) % 20 == 0:
            r = rows[-1]
            print("[%d] %s L%d: Fisher-overlap mis=%.3f ben=%.3f" %
                  (len(rows), label, layer, r["fisher_overlap_mis"],
                   r["fisher_overlap_ben_matched"]), flush=True)

    med_m = float(np.median([r["fisher_overlap_mis"] for r in rows]))
    med_b = float(np.median([r["fisher_overlap_ben_matched"] for r in rows]))
    summary = {"n_matrices": len(rows), "median_fisher_overlap_mis": med_m,
               "median_fisher_overlap_ben_matched": med_b,
               "frac_mis_higher": float(np.mean(
                   [r["fisher_overlap_mis"] > r["fisher_overlap_ben_matched"] for r in rows]))}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print("=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print("  %s: %.4g" % (k, v), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
