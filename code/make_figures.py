"""Generate the paper figures from results/data/spectral.jsonl.

Color palette (user-specified):
  #8df0a8 green, #8dd2f0 blue, #f08d96 red, #f0c88d amber
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

GREEN = "#8df0a8"
BLUE = "#8dd2f0"
RED = "#f08d96"
AMBER = "#f0c88d"
INK = "#222222"
GRID = "#dddddd"

# darker variants for lines/edges (the pastels are fills)
GREEN_D = "#3fb368"
BLUE_D = "#3f9cc8"
RED_D = "#c85563"
AMBER_D = "#c89a4f"

LABELS = ["q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"]
LABEL_COLOR = {
    "q_proj": BLUE_D, "k_proj": BLUE, "v_proj": GREEN_D, "o_proj": GREEN,
    "gate_proj": AMBER_D, "up_proj": AMBER, "down_proj": RED_D,
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "serif",
})


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_spectrum_panel(rows, outdir):
    """Singular-value spectrum of Delta for one representative matrix, with the
    fitted MP bulk band and the detected spikes highlighted."""
    # pick a mid-layer gate_proj (large gamma headroom)
    cand = [r for r in rows if r["label"] == "gate_proj" and r["layer"] == 15]
    if not cand:
        cand = [r for r in rows if r["label"] == "gate_proj"]
    r = cand[len(cand)//2] if cand else rows[0]
    d = r["delta"]
    sv = np.array(d["svals_head"])
    p, q = d["shape"]
    eig = sv ** 2 / p
    hi = d["mp_hi"]
    lo = d["mp_lo"] if "mp_lo" in d else d.get("mp_lo", 0)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    idx = np.arange(1, len(eig) + 1)
    above = eig > hi
    ax.axhspan(0, hi, color=BLUE, alpha=0.30, lw=0, label="MP bulk (noise)")
    ax.axhline(hi, color=BLUE_D, lw=1.0, ls="--", label="BBP edge $\\lambda_+$")
    ax.scatter(idx[~above], eig[~above], s=22, color=BLUE_D, zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(idx[above], eig[above], s=42, color=RED, zorder=4,
               edgecolors=RED_D, linewidths=0.8, label="supercritical spikes")
    ax.set_xlabel("singular index of $\\Delta W$")
    ax.set_ylabel("eigenvalue of $C=\\frac{1}{p}\\Delta W^{\\!\\top}\\Delta W$")
    ax.set_title(f"{r['label']} (layer {r['layer']}): "
                 f"{d['n_spikes']} spikes above the bulk edge", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_xlim(0, len(eig) + 1)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spectrum_panel.pdf"))
    plt.close(fig)


def fig_spikes_by_layer(rows, outdir):
    """Number of supercritical spikes in Delta, per layer, per matrix type."""
    layers = sorted(set(r["layer"] for r in rows))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for lab in LABELS:
        ys = []
        for L in layers:
            m = [r for r in rows if r["label"] == lab and r["layer"] == L]
            ys.append(m[0]["delta"]["n_spikes"] if m else np.nan)
        ax.plot(layers, ys, marker="o", ms=2.6, lw=1.1,
                color=LABEL_COLOR[lab], label=lab)
    ax.set_xlabel("layer")
    ax.set_ylabel("supercritical spikes in $\\Delta W$")
    ax.set_title("Alignment increment is low-rank at every layer", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spikes_by_layer.pdf"))
    plt.close(fig)


def fig_effrank(rows, outdir):
    """Effective rank of Delta vs the endpoint weights, per layer (one type)."""
    layers = sorted(set(r["layer"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    for ax, lab in zip(axes, ["gate_proj", "o_proj"]):
        dr, br, ir = [], [], []
        for L in layers:
            m = [r for r in rows if r["label"] == lab and r["layer"] == L]
            if m:
                dr.append(m[0]["delta"]["effective_rank"])
                br.append(m[0]["base"]["effective_rank"])
                ir.append(m[0]["instruct"]["effective_rank"])
            else:
                dr.append(np.nan); br.append(np.nan); ir.append(np.nan)
        ax.plot(layers, br, color=BLUE_D, lw=1.1, label="base $W$")
        ax.plot(layers, ir, color=GREEN_D, lw=1.1, label="instruct $W$")
        ax.plot(layers, dr, color=RED_D, lw=1.3, label="increment $\\Delta W$")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("layer")
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("effective rank")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "effrank.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/data/spectral.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.data)
    print(f"loaded {len(rows)} rows")
    fig_spectrum_panel(rows, args.outdir)
    fig_spikes_by_layer(rows, args.outdir)
    fig_effrank(rows, args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
