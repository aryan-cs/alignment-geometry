"""Weight-space geometry that needs no model inference (CPU, weights only).

Two analyses, both fast linear algebra on the base/instruct weights:

(1) Alignment of the increment's top singular directions with the base weight's
    own singular subspace. For each o_proj we compute the principal angles
    between the top-r left-singular subspace of Delta and the top-r left-singular
    subspace of W_base, summarized by the mean squared cosine (overlap). This
    asks whether alignment edits *existing* dominant directions or opens *new*
    ones, the proof's eigenvector-rotation question.

(2) Energy concentration curve: cumulative fraction of ||Delta||_F^2 captured by
    the top-k singular values, per matrix type, averaged over layers, to make
    "stable rank ~100" concrete as a curve.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore, classify  # noqa: E402


def princ_overlap(A_cols, B_cols):
    """Mean squared cosine of principal angles between two subspaces given as
    orthonormal column bases (d x r each)."""
    M = A_cols.T @ B_cols
    s = np.linalg.svd(M, compute_uv=False)
    return float((s ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--out", default="results/data/weight_geometry.json")
    args = ap.parse_args()

    b = WeightStore(args.base)
    i = WeightStore(args.instruct)

    targets = [(n,) + classify(n) for n in b.names() if classify(n)[0]]
    targets = [t for t in targets if t[1] is not None]
    targets.sort(key=lambda t: (t[2], t[1]))

    out = {"r": args.r, "overlap_oproj_by_layer": {}, "energy_curve": {}}
    energy_acc = {}
    for name, label, layer in targets:
        Wb = b.get(name).astype(np.float64)
        Wi = i.get(name).astype(np.float64)
        D = Wi - Wb
        # economy SVD of D and W_base on the smaller side
        if D.shape[0] < D.shape[1]:
            Dt, Wbt = D.T, Wb.T
        else:
            Dt, Wbt = D, Wb
        Ud, Sd, _ = np.linalg.svd(Dt, full_matrices=False)
        # cumulative energy curve
        e = Sd ** 2
        ecdf = np.cumsum(e) / e.sum()
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        energy_acc.setdefault(label, []).append([float(ecdf[min(k, len(ecdf)) - 1]) for k in ks])
        # (1) only for o_proj (square, residual basis) at this layer
        if label == "o_proj":
            Ub, _, _ = np.linalg.svd(Wbt, full_matrices=False)
            r = args.r
            ov = princ_overlap(Ud[:, :r], Ub[:, :r])
            # null: overlap of a random r-subspace with W_base top-r
            rng = np.random.default_rng(layer)
            Qr, _ = np.linalg.qr(rng.standard_normal((Ud.shape[0], r)))
            ov_null = princ_overlap(Qr, Ub[:, :r])
            out["overlap_oproj_by_layer"][str(layer)] = {
                "overlap": ov, "null": ov_null}
            if layer % 4 == 0:
                print(f"o_proj L{layer}: top-{r} Delta vs W_base overlap "
                      f"{ov:.3f} (null {ov_null:.3f})", flush=True)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    out["energy_curve"]["ks"] = ks
    for label, rows in energy_acc.items():
        out["energy_curve"][label] = np.mean(rows, 0).tolist()

    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
