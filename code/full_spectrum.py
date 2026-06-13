"""Compute and save the FULL eigenvalue spectrum of one increment matrix, plus
the fitted Marchenko-Pastur density, for the canonical bulk+spikes figure.

Run on the machine that has the weights (H200). Saves a small npz the Mac uses
to draw the figure.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import (WeightStore, fit_mp_sigma, marchenko_pastur_edges,
                      _mp_density)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--name", default="model.layers.15.mlp.gate_proj.weight")
    ap.add_argument("--out", default="results/data/full_spectrum.npz")
    args = ap.parse_args()

    b = WeightStore(args.base)
    i = WeightStore(args.instruct)
    D = i.get(args.name).astype(np.float64) - b.get(args.name).astype(np.float64)
    p, q = D.shape
    if p < q:
        D = D.T
        p, q = q, p
    gamma = q / p
    C = (D.T @ D) / p
    eig = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)   # all q eigenvalues
    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)

    # MP density curve (scaled by sigma2) over the bulk support
    xs = np.linspace(lo, hi, 400)
    dens = _mp_density(xs / sigma2, gamma) / sigma2

    np.savez(args.out, eig=eig.astype(np.float32), gamma=gamma, sigma2=sigma2,
             lo=lo, hi=hi, mp_x=xs.astype(np.float32),
             mp_y=dens.astype(np.float32), name=args.name, p=p, q=q)
    n_spike = int((eig > hi).sum())
    print(f"{args.name}: q={q} eigs, sigma2={sigma2:.3e}, edge={hi:.3e}, "
          f"{n_spike} above edge, top={eig[0]:.3e} ({eig[0]/hi:.1f}x edge)",
          flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
