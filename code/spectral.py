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
import struct
import numpy as np


# ---------- weight streaming ----------
# We parse the safetensors container directly. The format is:
#   [8-byte little-endian header length N][N bytes of JSON header][raw data]
# The JSON header maps each tensor name to {dtype, shape, data_offsets}.
# numpy has no native bfloat16, so bf16 tensors are widened to float32 by
# placing the 16 stored bits into the high half of a float32 (bf16 is exactly
# the top 16 bits of an IEEE-754 float32). Dependency surface: numpy only.

_ST_DTYPE = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def _shard_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    # single-shard model
    return {k: "model.safetensors" for k in
            _read_header(os.path.join(model_dir, "model.safetensors"))[0]
            if k != "__metadata__"}


class WeightStore:
    """Lazy access to named tensors across safetensors shards (torch-free)."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.wm = _shard_index(model_dir)
        self._mm = {}      # shard -> memmap of raw bytes
        self._hdr = {}     # shard -> (header dict, data_start)

    def names(self):
        return [k for k in self.wm if k != "__metadata__"]

    def _shard(self, shard):
        if shard not in self._mm:
            path = os.path.join(self.dir, shard)
            hdr, start = _read_header(path)
            self._hdr[shard] = (hdr, start)
            self._mm[shard] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._mm[shard], self._hdr[shard]

    def get(self, name):
        shard = self.wm[name]
        mm, (hdr, start) = self._shard(shard)
        meta = hdr[name]
        dt = meta["dtype"]
        shape = tuple(meta["shape"])
        a, b = meta["data_offsets"]
        buf = mm[start + a: start + b]
        if dt == "BF16":
            u16 = buf.view(np.uint16).astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)
            return f32.reshape(shape)
        return buf.view(_ST_DTYPE[dt]).reshape(shape)



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

    # Thin SVD via the small q x q gram matrix C = (1/p) D^T D (q <= p, so this
    # is the cheap side). Eigendecomposition of C gives eigenvalues = s^2/p and
    # right singular vectors V; we recover the top-k left vectors as U = D V / s.
    # This avoids a full (p x p) SVD of the wide matrices and is far faster.
    C = (D.T @ D) / p                      # (q, q), symmetric PSD
    evals, evecs = np.linalg.eigh(C)       # ascending
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    eig = evals                            # eigenvalues of C
    svals = np.sqrt(eig * p)               # singular values of D

    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    # Finite-size Tracy-Widom tolerance at the soft edge (Johnstone): fluctuations
    # scale as sigma2 * (1+sqrt gamma)^{4/3} * gamma^{-1/6} * p^{-2/3}. Count a spike
    # only if it clears the edge by several TW standard deviations, so a clean
    # MP bulk yields zero spikes (no boundary false positives).
    tw_scale = sigma2 * (1 + np.sqrt(gamma)) ** (4.0 / 3.0) * (gamma ** (-1.0 / 6.0)) * p ** (-2.0 / 3.0)
    thresh = hi + 6.0 * tw_scale
    n_spikes = int((eig > thresh).sum())
    spikes = eig[eig > thresh]

    Vk = evecs[:, :topk]                    # (q, topk) right singular vectors
    s_top = svals[:topk]
    Uk = (D @ Vk) / np.maximum(s_top, 1e-12)  # (p, topk) left singular vectors
    topV = Vk.T.astype(np.float32)          # (topk, q)
    topU = Uk.T.astype(np.float32)          # (topk, p)

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
    C = (A.T @ A) / p
    eig = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    svals = np.sqrt(eig * p)
    sigma2 = fit_mp_sigma(eig, gamma)
    lo, hi = marchenko_pastur_edges(sigma2, gamma)
    tw_scale = sigma2 * (1 + np.sqrt(gamma)) ** (4.0 / 3.0) * (gamma ** (-1.0 / 6.0)) * p ** (-2.0 / 3.0)
    thresh = hi + 6.0 * tw_scale
    return {
        "shape": [int(p), int(q)],
        "gamma": float(gamma),
        "sigma2_bulk": float(sigma2),
        "mp_hi": float(hi),
        "n_spikes": int((eig > thresh).sum()),
        "effective_rank": effective_rank(svals),
        "stable_rank": stable_rank(svals),
        "frob_energy": float((svals ** 2).sum()),
        "svals_head": svals[:64].astype(np.float32).tolist(),
    }
