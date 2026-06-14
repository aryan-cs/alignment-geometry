"""Scout: does a purposefully MISALIGNED fine-tune have a distinctive spectral
signature vs a benign control, at matched weight-change energy?

Matched trio from one base (Qwen2.5-Coder-14B-Instruct):
  - insecure  = emergent-misalignment fine-tune (misaligned)
  - secure    = same-style fine-tune, benign     (control)
We form Delta_mis = W_insecure - W_base and Delta_ben = W_secure - W_base,
rescale to EQUAL Frobenius energy per matrix, and compare their spectra. The
proof predicts: at matched energy, misalignment concentrates in lower rank
(fewer, taller spikes / smaller stable rank). This is a go/no-go: if the two are
spectrally indistinguishable at matched energy, the "instant misalignment
detector" does not exist and we report that bound.

CPU (pure linear algebra on weights). Writes results/data/misalign_scout.json.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import (WeightStore, classify, fit_mp_sigma,
                      marchenko_pastur_edges, effective_rank, stable_rank)


def spectrum_stats(D):
    """Spectral summary of an increment matrix D (oriented q<=p)."""
    p, q = D.shape
    if p < q:
        D = D.T; p, q = q, p
    gamma = q / p
    C = (D.T @ D) / p
    eig = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    svals = np.sqrt(eig * p)
    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    tw = sigma2 * (1 + np.sqrt(gamma)) * (gamma ** (-1.0 / 6.0)) * p ** (-2.0 / 3.0)
    thr = hi + 6 * tw
    return {
        "frob2": float((svals ** 2).sum()),
        "n_spikes": int((eig > thr).sum()),
        "top_over_edge": float(eig.max() / hi),
        "stable_rank": stable_rank(svals),
        "effective_rank": effective_rank(svals),
        "gamma": float(gamma),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--misaligned", required=True)
    ap.add_argument("--benign", required=True)
    ap.add_argument("--out", default="results/data/misalign_scout.json")
    args = ap.parse_args()

    base = WeightStore(args.base)
    mis = WeightStore(args.misaligned)
    ben = WeightStore(args.benign)

    targets = []
    for name in base.names():
        label, layer = classify(name)
        if label is not None:
            targets.append((name, label, layer))
    targets.sort(key=lambda t: (t[2], t[1]))
    print("%d target matrices" % len(targets), flush=True)

    rows = []
    for i, (name, label, layer) in enumerate(targets):
        Wb = base.get(name).astype(np.float64)
        Dm = mis.get(name).astype(np.float64) - Wb
        Db = ben.get(name).astype(np.float64) - Wb
        em = float((Dm ** 2).sum())
        eb = float((Db ** 2).sum())
        # rescale benign increment to the misaligned increment's energy, so the
        # comparison is at MATCHED Frobenius energy (the proof's premise).
        if eb > 0:
            Db_matched = Db * np.sqrt(em / eb)
        else:
            Db_matched = Db
        sm = spectrum_stats(Dm)
        sb = spectrum_stats(Db_matched)
        rows.append({"name": name, "label": label, "layer": layer,
                     "energy_mis": em, "energy_ben": eb,
                     "mis": sm, "ben_matched": sb})
        if (i + 1) % 20 == 0 or i < 3:
            print("[%d/%d] %s L%d: mis stable_rank=%.1f spikes=%d | "
                  "ben(matched) stable_rank=%.1f spikes=%d" %
                  (i + 1, len(targets), label, layer,
                   sm["stable_rank"], sm["n_spikes"],
                   sb["stable_rank"], sb["n_spikes"]), flush=True)

    # aggregate: is misalignment more concentrated at matched energy?
    def med(key, arm):
        return float(np.median([r[arm][key] for r in rows]))
    summary = {
        "n_matrices": len(rows),
        "median_stable_rank_mis": med("stable_rank", "mis"),
        "median_stable_rank_ben_matched": med("stable_rank", "ben_matched"),
        "median_spikes_mis": med("n_spikes", "mis"),
        "median_spikes_ben_matched": med("n_spikes", "ben_matched"),
        "median_top_over_edge_mis": med("top_over_edge", "mis"),
        "median_top_over_edge_ben_matched": med("top_over_edge", "ben_matched"),
        "median_energy_ratio_mis_over_ben": float(np.median(
            [r["energy_mis"] / max(r["energy_ben"], 1e-12) for r in rows])),
        # fraction of matrices where misaligned is MORE concentrated (lower stable rank)
        "frac_mis_lower_stable_rank": float(np.mean(
            [r["mis"]["stable_rank"] < r["ben_matched"]["stable_rank"] for r in rows])),
    }
    out = {"summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print("  %s: %.4g" % (k, v), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
