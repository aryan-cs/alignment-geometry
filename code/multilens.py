"""Multi-lens spectral/structural analysis of the alignment increment.

Established tools, applied to a novel object (the fine-tuning increment Delta W
and the difference-of-arms W_insecure - W_educational) under a controlled
matched-energy design. The contribution is the question + design + the agreement
(or disagreement) across independent lenses, not any single technique.

Lenses (each a different assumption about where structure lives):
  SVD          - energy / variance in orthogonal directions      (baseline)
  alpha        - heavy-tail exponent of the ESD (Martin-Mahoney)  [Tier 1]
  spacing      - eigenvalue spacing ratio: Wigner (random) vs Poisson [Tier 1]
  ipr          - eigenvector localization (inverse participation ratio) [Tier 1]
  ica          - independent (non-Gaussian) directions, not energy [Tier 2]
  tensor       - CP decomposition of the layer x in x out stack    [Tier 3]
  tda          - persistent homology of the spectrum / row cloud   [Tier 3]

Hessian/Fisher lens lives in a separate script (needs the model + a loss, not
just the weight matrices). CPU.
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore, classify, fit_mp_sigma, marchenko_pastur_edges


# ---------- per-matrix lenses ----------

def svd_vals(D):
    p, q = D.shape
    if p < q:
        D = D.T; p, q = q, p
    return np.linalg.svd(D, compute_uv=False), p, q


def lens_alpha(svals):
    """Heavy-tail exponent: power-law fit to the upper tail of the eigenvalue
    (=singular^2) density. Lower alpha => heavier tail / more structure.
    Uses the `powerlaw` package on the squared singular values."""
    import powerlaw
    eig = (svals ** 2)
    eig = eig[eig > 0]
    try:
        fit = powerlaw.Fit(eig, verbose=False)
        return float(fit.alpha)
    except Exception:
        return float("nan")


def lens_spacing(svals):
    """Mean consecutive-eigenvalue spacing ratio <r>. GOE/Wigner (random,
    level-repulsion) ~0.536; Poisson (uncorrelated, structured) ~0.386."""
    e = np.sort((svals ** 2))[::-1]
    s = -np.diff(e)            # gaps (positive)
    s = s[s > 0]
    if len(s) < 3:
        return float("nan")
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


def lens_ipr(D, k=8):
    """Mean inverse participation ratio of the top-k right singular vectors.
    IPR = sum v_i^4 (for unit v); ~1/q for delocalized, ->1 for localized."""
    p, q = D.shape
    if p < q:
        D = D.T
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    V = Vt[:k]
    ipr = (V ** 4).sum(1) / ((V ** 2).sum(1) ** 2)
    return float(np.mean(ipr))


def lens_ica_nongauss(D, k=16):
    """Run FastICA on the rows of D (samples = rows); report mean absolute
    excess kurtosis of the recovered components. SVD finds uncorrelated
    directions; ICA finds independent/non-Gaussian ones. High |kurtosis| =>
    structure SVD's 2nd-order view misses."""
    from sklearn.decomposition import FastICA
    from scipy.stats import kurtosis
    p, q = D.shape
    X = D if p >= q else D.T
    # standardize columns, run ICA to k components
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    try:
        ica = FastICA(n_components=k, max_iter=300, tol=1e-3, whiten="unit-variance")
        S = ica.fit_transform(Xc)
        return float(np.mean(np.abs(kurtosis(S, axis=0, fisher=True))))
    except Exception:
        return float("nan")


def lens_tda(svals, maxdim=1):
    """Persistent homology of the 1-D point cloud of log-eigenvalues: total
    persistence (sum of bar lengths) of H0. Captures clustering/gaps in the
    spectrum that linear summaries miss. Returns total H0 persistence."""
    from ripser import ripser
    e = np.log(svals[svals > 0] + 1e-12).reshape(-1, 1)
    if len(e) < 5:
        return float("nan")
    # subsample for tractability
    if len(e) > 400:
        idx = np.linspace(0, len(e) - 1, 400).astype(int)
        e = e[idx]
    dgm = ripser(e, maxdim=0)["dgms"][0]
    fin = dgm[np.isfinite(dgm[:, 1])]
    return float((fin[:, 1] - fin[:, 0]).sum())


def per_matrix(D):
    sv, p, q = svd_vals(D)
    return {
        "alpha": lens_alpha(sv),
        "spacing_ratio": lens_spacing(sv),
        "ipr_top8": lens_ipr(D),
        "ica_kurtosis": lens_ica_nongauss(D),
        "tda_h0_persistence": lens_tda(sv),
    }


# ---------- cross-layer tensor lens ----------

def tensor_lens(stack, rank=8):
    """CP decomposition of a (layer x p x q) tensor of increments for one matrix
    type. Reports reconstruction-explained variance at low rank: high => the
    increment is coherent across layers (a few shared factors), which per-layer
    SVD cannot see. Uses tensorly."""
    import tensorly as tl
    from tensorly.decomposition import parafac
    T = tl.tensor(stack.astype(np.float64))
    try:
        w, facs = parafac(T, rank=rank, n_iter_max=120, init="svd", tol=1e-6)
        rec = tl.cp_to_tensor((w, facs))
        ev = 1.0 - float(np.linalg.norm(T - rec) ** 2 / (np.linalg.norm(T) ** 2 + 1e-12))
        return ev
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--insecure", required=True)
    ap.add_argument("--educational", required=True)
    ap.add_argument("--out", default="results/data/multilens.json")
    ap.add_argument("--tensor-type", default="o_proj",
                    help="matrix type for the cross-layer tensor lens")
    args = ap.parse_args()

    b = WeightStore(args.base)
    mi = WeightStore(args.insecure)
    ed = WeightStore(args.educational)
    targets = [(n,) + classify(n) for n in b.names() if classify(n)[0]]
    targets = [t for t in targets if t[1] is not None]
    targets.sort(key=lambda t: (t[2], t[1]))
    print("%d matrices" % len(targets), flush=True)

    rows = []
    diff_stack, mis_stack, ben_stack = [], [], []
    for i, (name, label, layer) in enumerate(targets):
        Wb = b.get(name).astype(np.float64)
        Dm = mi.get(name).astype(np.float64) - Wb
        De = ed.get(name).astype(np.float64) - Wb
        # match energy on the benign arm, and isolate the misalignment delta
        em, ee = (Dm ** 2).sum(), (De ** 2).sum()
        Dem = De * np.sqrt(em / ee) if ee > 0 else De
        Ddiff = (mi.get(name).astype(np.float64) - ed.get(name).astype(np.float64))
        rows.append({"name": name, "label": label, "layer": layer,
                     "mis": per_matrix(Dm), "ben_matched": per_matrix(Dem),
                     "diff": per_matrix(Ddiff)})
        if label == args.tensor_type:
            diff_stack.append(Ddiff)
            mis_stack.append(Dm)
            ben_stack.append(Dem)
        if (i + 1) % 20 == 0 or i < 2:
            r = rows[-1]
            print("[%d/%d] %s L%d: mis alpha=%.2f spacing=%.3f ipr=%.4f ica=%.2f | "
                  "diff alpha=%.2f" % (i + 1, len(targets), label, layer,
                  r["mis"]["alpha"], r["mis"]["spacing_ratio"], r["mis"]["ipr_top8"],
                  r["mis"]["ica_kurtosis"], r["diff"]["alpha"]), flush=True)

    # cross-layer tensor lens for the chosen matrix type
    tensors = {}
    if diff_stack:
        st = np.stack(diff_stack)
        tensors = {
            "type": args.tensor_type, "n_layers": len(diff_stack),
            "cp_explained_var_diff": tensor_lens(np.stack(diff_stack)),
            "cp_explained_var_mis": tensor_lens(np.stack(mis_stack)),
            "cp_explained_var_ben_matched": tensor_lens(np.stack(ben_stack)),
        }
        print("tensor (%s): CP-r8 explained var  diff=%.3f mis=%.3f ben=%.3f" % (
            args.tensor_type, tensors["cp_explained_var_diff"],
            tensors["cp_explained_var_mis"], tensors["cp_explained_var_ben_matched"]),
            flush=True)

    def med(lens, arm):
        xs = [r[arm][lens] for r in rows if np.isfinite(r[arm][lens])]
        return float(np.median(xs)) if xs else float("nan")
    lenses = ["alpha", "spacing_ratio", "ipr_top8", "ica_kurtosis", "tda_h0_persistence"]
    summary = {"n_matrices": len(rows), "tensor": tensors}
    for lens in lenses:
        summary[lens] = {"mis": med(lens, "mis"),
                         "ben_matched": med(lens, "ben_matched"),
                         "diff": med(lens, "diff")}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print("=== SUMMARY (median over matrices) ===", flush=True)
    for lens in lenses:
        s = summary[lens]
        print("  %-20s mis=%.4g  ben=%.4g  diff=%.4g" %
              (lens, s["mis"], s["ben_matched"], s["diff"]), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
