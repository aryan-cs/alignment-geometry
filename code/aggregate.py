"""Aggregate the spectral sweep into the numbers the paper cites.

Reads results/data/spectral.jsonl and prints/writes summary statistics:
  - across all matrices: median and range of top-eigenvalue / MP-edge ratio
  - spike counts vs matrix rank, and effective-rank ratio Delta vs endpoints
  - per-matrix-type breakdown
Writes results/data/summary.json.
"""
import os
import sys
import json
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/data/spectral.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
print(f"{len(rows)} matrices")

top_edge = np.array([r["delta"]["top_eig_over_edge"] for r in rows])
spikes = np.array([r["delta"]["n_spikes"] for r in rows])
q = np.array([min(r["delta"]["shape"]) for r in rows])   # rank ceiling
er_d = np.array([r["delta"]["effective_rank"] for r in rows])
er_b = np.array([r["base"]["effective_rank"] for r in rows])
er_i = np.array([r["instruct"]["effective_rank"] for r in rows])
sr_d = np.array([r["delta"]["stable_rank"] for r in rows])

summary = {
    "n_matrices": len(rows),
    "top_over_edge": {
        "min": float(top_edge.min()), "median": float(np.median(top_edge)),
        "max": float(top_edge.max()),
        "frac_above_1": float((top_edge > 1).mean()),
        "frac_above_5": float((top_edge > 5).mean()),
    },
    "spikes": {
        "min": int(spikes.min()), "median": float(np.median(spikes)),
        "max": int(spikes.max()),
        "median_spikes_over_rank": float(np.median(spikes / q)),
    },
    "effrank_ratio_delta_vs_base": float(np.median(er_d / er_b)),
    "effrank_ratio_delta_vs_instruct": float(np.median(er_d / er_i)),
    "stable_rank_delta_median": float(np.median(sr_d)),
    "by_type": {},
}
for lab in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    sub = [r for r in rows if r["label"] == lab]
    if not sub:
        continue
    te = np.array([r["delta"]["top_eig_over_edge"] for r in sub])
    sp = np.array([r["delta"]["n_spikes"] for r in sub])
    summary["by_type"][lab] = {
        "n": len(sub),
        "median_top_over_edge": float(np.median(te)),
        "median_spikes": float(np.median(sp)),
        "median_stable_rank": float(np.median([r["delta"]["stable_rank"] for r in sub])),
    }

os.makedirs("results/data", exist_ok=True)
json.dump(summary, open("results/data/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
