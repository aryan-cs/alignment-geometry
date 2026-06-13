"""Core spectral analysis of the alignment fine-tuning increment.

For each linear weight matrix W in a transformer, we compare the base and
instruct checkpoints and analyze the increment Delta = W_instruct - W_base.
The theory (docs/proof.pdf) models Delta as a diffuse Marchenko-Pastur bulk
plus a low-rank spike, and predicts that the alignment-relevant signal is a
small number of supercritical singular directions.

This module is pure linear algebra on the weight tensors. It streams tensors
from safetensors shards, so it never materializes a full model. CPU only.

Outputs, per matrix:
  - singular value spectrum of Delta (and of W_base, W_instruct)
  - Marchenko-Pastur bulk fit (noise sigma, edges) on Delta
  - number of singular values above the MP upper edge (spikes)
  - stable rank, effective rank, Frobenius energy
  - top-k right singular vectors of Delta (for the steering experiments)
"""
import os
import re
import json
import glob
import numpy as np
from safetensors import safe_open


# ---------- weight streaming ----------

def _shard_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx) as f:
        wm = json.load(f)["weight_map"]
    return wm


class WeightStore:
    """Lazy access to named tensors across safetensors shards."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.wm = _shard_index(model_dir)
        self._open = {}

    def names(self):
        return list(self.wm.keys())

    def get(self, name):
        shard = self.wm[name]
        if shard not in self._open:
            self._open[shard] = safe_open(
                os.path.join(self.dir, shard), framework="np")
        return self._open[shard].get_tensor(name)


# ---------- target matrices ----------

# 2D weight matrices we analyze, by regex on the parameter name.
TARGET_PATTERNS = [
    (r"self_attn\.q_proj\.weight$", "q_proj"),
    (r"self_attn\.k_proj\.weight$", "k_proj"),
    (r"self_attn\.v_proj\.weight$", "v_proj"),
    (r"self_attn\.o_proj\.weight$", "o_proj"),
    (r"mlp\.gate_proj\.weight$", "gate_proj"),
    (r"mlp\.up_proj\.weight$", "up_proj"),
    (r"mlp\.down_proj\.weight$", "down_proj"),
]


def classify(name):
    for pat, label in TARGET_PATTERNS:
        if re.search(pat, name):
            m = re.search(r"layers\.(\d+)\.", name)
            layer = int(m.group(1)) if m else -1
            return label, layer
    return None, None


# ---------- spectral quantities ----------

def marchenko_pastur_edges(sigma2, gamma):
    """Upper/lower edges of the MP law for aspect ratio gamma=q/p (<=1)."""
    lo = sigma2 * (1 - np.sqrt(gamma)) ** 2
    hi = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return lo, hi


def fit_mp_sigma(eigs, gamma):
    """Estimate the bulk noise level sigma^2 from the median eigenvalue.

    Matching the sample median to the MP median is robust to a few spikes,
    which a trace-based estimate is not. We solve sigma^2 = median(eig)/m(gamma)
    where m(gamma) is the median of the standard MP law (sigma^2=1), found by
    numerically inverting the MP CDF.
    """
    med = np.median(eigs)
    m = _mp_median(gamma)
    return med / m


def _mp_density(x, gamma):
    lo = (1 - np.sqrt(gamma)) ** 2
    hi = (1 + np.sqrt(gamma)) ** 2
    out = np.zeros_like(x)
    mask = (x > lo) & (x < hi)
    xm = x[mask]
    out[mask] = np.sqrt((hi - xm) * (xm - lo)) / (2 * np.pi * gamma * xm)
    return out


def _mp_median(gamma, n=200000):
    lo = (1 - np.sqrt(gamma)) ** 2
    hi = (1 + np.sqrt(gamma)) ** 2
    xs = np.linspace(lo, hi, n)
    dens = _mp_density(xs, gamma)
    cdf = np.cumsum(dens) * (xs[1] - xs[0])
    cdf /= cdf[-1]
    return xs[np.searchsorted(cdf, 0.5)]


def effective_rank(svals):
    """exp(entropy of normalized singular-value distribution)."""
    s = svals[svals > 0]
    p = s / s.sum()
    h = -(p * np.log(p)).sum()
    return float(np.exp(h))


def stable_rank(svals):
    """||A||_F^2 / ||A||_2^2 = sum s^2 / max s^2."""
    s2 = svals ** 2
    return float(s2.sum() / s2.max())


def analyze_delta(W_base, W_inst, topk=8):
    """Spectral analysis of Delta = W_inst - W_base.

    Returns a dict of scalar summaries plus the singular spectrum and the
    top-k right singular vectors (for downstream steering).
    """
    D = (W_inst.astype(np.float64) - W_base.astype(np.float64))
    p, q = D.shape
    if p < q:
        D = D.T
        p, q = q, p
    gamma = q / p

    # Economy SVD: singular values of D, eigenvalues of (1/p) D^T D.
    # We work with C = (1/p) D^T D (q x q) eigenvalues = s^2 / p.
    svals = np.linalg.svd(D, compute_uv=False)
    eig = (svals ** 2) / p  # eigenvalues of C

    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    n_spikes = int((eig > hi).sum())
    # spike strengths theta from the BBP inversion lambda = sigma2 (1+theta)(1+gamma/theta)
    spikes = eig[eig > hi]

    # top-k right singular vectors for steering (recompute with uv on a thin SVD)
    # full_matrices=False gives Vt of shape (q, q); rows are right singular vecs.
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    topV = Vt[:topk].astype(np.float32)   # (topk, q) in the *long-axis* basis
    topU = U[:, :topk].astype(np.float32)  # (p, topk)

    out = {
        "shape": [int(p), int(q)],
        "gamma": float(gamma),
        "sigma2_bulk": float(sigma2),
        "mp_lo": float(lo),
        "mp_hi": float(hi),
        "n_spikes": n_spikes,
        "top_eig": float(eig.max()),
        "top_eig_over_edge": float(eig.max() / hi),
        "spike_eigs": spikes[:topk].tolist(),
        "frob_energy": float((svals ** 2).sum()),
        "stable_rank": stable_rank(svals),
        "effective_rank": effective_rank(svals),
        "svals_head": svals[:64].astype(np.float32).tolist(),
        "transposed": bool(D.shape[0] != (W_inst.shape[0])),
    }
    return out, topV, topU, svals


def analyze_matrix_self(W, topk=0):
    """Spectral summary of a single matrix (for base/instruct baselines)."""
    A = W.astype(np.float64)
    p, q = A.shape
    if p < q:
        A = A.T
        p, q = q, p
    gamma = q / p
    svals = np.linalg.svd(A, compute_uv=False)
    eig = (svals ** 2) / p
    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    return {
        "shape": [int(p), int(q)],
        "gamma": float(gamma),
        "sigma2_bulk": float(sigma2),
        "mp_hi": float(hi),
        "n_spikes": int((eig > hi).sum()),
        "effective_rank": effective_rank(svals),
        "stable_rank": stable_rank(svals),
        "frob_energy": float((svals ** 2).sum()),
        "svals_head": svals[:64].astype(np.float32).tolist(),
    }
