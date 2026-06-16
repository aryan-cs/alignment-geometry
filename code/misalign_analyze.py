"""Analyze the CUSTOM matched arms (controlled recipe, only objective differs).

Three spectral views:
  (A) matched-energy ΔW: insecure-base vs educational-base, rescaled to equal
      Frobenius energy. The proof's rank-at-energy test, now with a controlled
      recipe (vs the scout, where the public arms shared ~95% of their update).
  (B) difference-of-arms: D_diff = W_insecure - W_educational. The shared
      training cancels, leaving the misalignment-relevant perturbation. We report
      its magnitude relative to each arm's increment and its spectral shape.
  (C) overlap of the two increments' top subspaces (principal angles): if
      misalignment adds a distinct direction, the top subspaces should differ
      more than two benign reruns would.

Writes results/data/misalign_custom.json. CPU.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import (WeightStore, classify, fit_mp_sigma,
                      marchenko_pastur_edges, effective_rank, stable_rank)


def stats(D):
    p, q = D.shape
    if p < q:
        D = D.T; p, q = q, p
    gamma = q / p
    C = (D.T @ D) / p
    eig = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    sv = np.sqrt(eig * p)
    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    tw = sigma2 * (1 + np.sqrt(gamma)) ** (4.0 / 3.0) * gamma ** (-1 / 6) * p ** (-2 / 3)
    return {"n_spikes": int((eig > hi + 6 * tw).sum()),
            "top_over_edge": float(eig.max() / hi),
            "stable_rank": stable_rank(sv), "effective_rank": effective_rank(sv),
            "frob2": float((sv ** 2).sum())}


def top_subspace(D, k=32):
    p, q = D.shape
    if p < q:
        D = D.T
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, :k]


def overlap(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float((s ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--insecure", required=True)
    ap.add_argument("--educational", required=True)
    ap.add_argument("--out", default="results/data/misalign_custom.json")
    args = ap.parse_args()

    b = WeightStore(args.base)
    mi = WeightStore(args.insecure)
    ed = WeightStore(args.educational)
    targets = [(n,) + classify(n) for n in b.names() if classify(n)[0]]
    targets = [t for t in targets if t[1] is not None]
    targets.sort(key=lambda t: (t[2], t[1]))
    print("%d matrices" % len(targets), flush=True)

    rows = []
    for i, (name, label, layer) in enumerate(targets):
        Wb = b.get(name).astype(np.float64)
        Wm = mi.get(name).astype(np.float64)
        We = ed.get(name).astype(np.float64)
        Dm, De = Wm - Wb, We - Wb
        em, ee = float((Dm ** 2).sum()), float((De ** 2).sum())
        De_matched = De * np.sqrt(em / ee) if ee > 0 else De
        Ddiff = Wm - We
        # how big is the misalignment-relevant part vs each arm's increment
        rel = float(np.linalg.norm(Ddiff) / max(np.linalg.norm(Dm), 1e-9))
        sub_ov = overlap(top_subspace(Dm), top_subspace(De))
        rows.append({"name": name, "label": label, "layer": layer,
                     "mis": stats(Dm), "ben_matched": stats(De_matched),
                     "diff": stats(Ddiff), "diff_rel_norm": rel,
                     "top_subspace_overlap": sub_ov,
                     "energy_mis": em, "energy_ben": ee})
        if (i + 1) % 20 == 0 or i < 3:
            r = rows[-1]
            print("[%d/%d] %s L%d: mis srank=%.1f ben srank=%.1f | "
                  "diff/mis norm=%.3f subspace_ov=%.3f" %
                  (i + 1, len(targets), label, layer, r["mis"]["stable_rank"],
                   r["ben_matched"]["stable_rank"], rel, sub_ov), flush=True)

    def med(k, arm):
        return float(np.median([r[arm][k] for r in rows]))
    summary = {
        "n_matrices": len(rows),
        "median_stable_rank_mis": med("stable_rank", "mis"),
        "median_stable_rank_ben_matched": med("stable_rank", "ben_matched"),
        "median_spikes_mis": med("n_spikes", "mis"),
        "median_spikes_ben_matched": med("n_spikes", "ben_matched"),
        "median_top_edge_mis": med("top_over_edge", "mis"),
        "median_top_edge_ben_matched": med("top_over_edge", "ben_matched"),
        "frac_mis_lower_stable_rank": float(np.mean(
            [r["mis"]["stable_rank"] < r["ben_matched"]["stable_rank"] for r in rows])),
        "median_diff_rel_norm": float(np.median([r["diff_rel_norm"] for r in rows])),
        "median_top_subspace_overlap": float(np.median([r["top_subspace_overlap"] for r in rows])),
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print("=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print("  %s: %.4g" % (k, v), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
