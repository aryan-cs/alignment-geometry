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


def fig_bulk_spikes(outdir, npz="results/data/full_spectrum.npz"):
    """Canonical bulk+spikes figure: full eigenvalue histogram of one increment
    with the fitted Marchenko-Pastur density overlaid and the supercritical
    spikes marked. Two panels: the bulk (linear) and the full spectrum (log)."""
    if not os.path.exists(npz):
        return
    z = np.load(npz)
    eig = z["eig"]
    hi = float(z["hi"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # left: the bulk, with the MP density
    bulk = eig[eig <= hi * 1.3]
    axL.hist(bulk, bins=70, density=True, color=BLUE, alpha=0.55,
             edgecolor="none", label="empirical")
    axL.plot(z["mp_x"], z["mp_y"], color=BLUE_D, lw=1.4,
             label="Marchenko--Pastur fit")
    axL.axvline(hi, color=RED_D, lw=1.1, ls="--", label="edge $\\lambda_+$")
    axL.set_xlabel("eigenvalue of $C$")
    axL.set_ylabel("density")
    axL.set_title("the bulk is Marchenko--Pastur", fontsize=9)
    axL.legend(frameon=False, fontsize=7)
    axL.grid(True, color=GRID, lw=0.5)

    # right: full spectrum, rank-ordered, log-y; bulk vs spikes colored
    idx = np.arange(1, len(eig) + 1)
    above = eig > hi
    axR.scatter(idx[~above], eig[~above], s=4, color=BLUE_D, label="bulk")
    axR.scatter(idx[above], eig[above], s=6, color=RED,
                label=f"{int(above.sum())} spikes $>\\lambda_+$")
    axR.axhline(hi, color=RED_D, lw=1.0, ls="--")
    axR.set_yscale("log")
    axR.set_xlabel("rank-ordered index")
    axR.set_ylabel("eigenvalue of $C$ (log)")
    axR.set_title("spikes detach above the edge", fontsize=9)
    axR.legend(frameon=False, fontsize=7, loc="upper right")
    axR.grid(True, color=GRID, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bulk_spikes.pdf"))
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


def fig_capture(outdir, beh="results/data/behavioral.json"):
    """Refusal-direction capture by the top-k o_proj increment subspace vs the
    random-subspace null, as a function of k."""
    if not os.path.exists(beh):
        return
    b = json.load(open(beh))
    cap = b.get("capture", {})
    if not cap:
        return
    # collect o_proj_k* entries
    ks, caps, nulls = [], [], []
    for key, v in sorted(cap.items()):
        if key.startswith("o_proj_k"):
            k = int(key.split("k")[-1])
            ks.append(k); caps.append(v); nulls.append(b["null"].get(key, 0))
    if not ks:
        return
    order = np.argsort(ks)
    ks = np.array(ks)[order]; caps = np.array(caps)[order]; nulls = np.array(nulls)[order]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ks, caps, "o-", color=RED_D, lw=1.4, ms=5, label="refusal capture")
    ax.plot(ks, nulls, "s--", color=BLUE_D, lw=1.2, ms=4, label="random-subspace null")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("subspace dimension $k$")
    ax.set_ylabel("captured fraction of refusal direction")
    ax.set_title("Refusal lives in the top increment directions", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capture.pdf"))
    plt.close(fig)


def fig_energy_overlap(outdir, wg="results/data/weight_geometry.json"):
    """Two panels: (a) cumulative energy captured by top-k singular directions of
    the increment, per matrix type; (b) overlap of the increment's top-16
    subspace with the base weight's top-16, per layer, vs the random null."""
    if not os.path.exists(wg):
        return
    d = json.load(open(wg))
    ec = d["energy_curve"]; ks = ec["ks"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for lab, col in [("q_proj", BLUE_D), ("o_proj", GREEN_D),
                     ("gate_proj", AMBER_D), ("down_proj", RED_D)]:
        if lab in ec:
            axL.plot(ks, ec[lab], "o-", ms=3, lw=1.2, color=col, label=lab)
    axL.set_xscale("log", base=2)
    axL.set_xlabel("top-$k$ singular directions")
    axL.set_ylabel("cumulative fraction of $\\|\\Delta W\\|_F^2$")
    axL.set_title("energy is front-loaded into a few directions", fontsize=9)
    axL.legend(frameon=False, fontsize=7)
    axL.grid(True, color=GRID, lw=0.5, which="both")

    layers = sorted(int(L) for L in d["overlap_oproj_by_layer"])
    ov = [d["overlap_oproj_by_layer"][str(L)]["overlap"] for L in layers]
    nu = [d["overlap_oproj_by_layer"][str(L)]["null"] for L in layers]
    axR.plot(layers, ov, "o-", ms=3, lw=1.2, color=RED_D, label="$\\Delta W$ vs base top-16")
    axR.plot(layers, nu, "s--", ms=2.5, lw=1.0, color=BLUE_D, label="random null")
    axR.set_xlabel("layer")
    axR.set_ylabel("subspace overlap (mean cos$^2$)")
    axR.set_title("alignment opens new directions", fontsize=9)
    axR.legend(frameon=False, fontsize=7.5)
    axR.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "energy_overlap.pdf"))
    plt.close(fig)


def fig_capture_heatmap(outdir, sweep="results/data/capture_sweep.json"):
    """Layer x k enrichment heatmap of refusal-direction capture by the o_proj
    increment subspace, in the project palette."""
    if not os.path.exists(sweep):
        return
    import matplotlib.colors as mcolors
    s = json.load(open(sweep))
    ks = s["ks"]
    layers = sorted(int(L) for L in s["layers"])
    M = np.array([[s["layers"][str(L)]["enrich"][str(k)] for k in ks] for L in layers])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    # palette ramp: white -> green -> blue for enrichment
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fa", ["#ffffff", GREEN, BLUE_D])
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cmap,
                   norm=mcolors.LogNorm(vmin=1, vmax=max(2, M.max())),
                   extent=[layers[0], layers[-1], -0.5, len(ks) - 0.5])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_xlabel("layer")
    ax.set_ylabel("subspace dimension $k$")
    ax.set_title("Refusal-direction enrichment over null (o\\_proj increment)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("enrichment ($\\times$ null)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capture_heatmap.pdf"))
    plt.close(fig)


def fig_ablation(outdir, abl="results/data/ablation_sweep.json"):
    """Causal ablation: harmful-vs-harmless separation (AUC) and refusal rate as
    a function of ablated increment-subspace dimension k, with the refusal-
    direction (positive control) and random-subspace (negative control)."""
    if not os.path.exists(abl):
        return
    d = json.load(open(abl))
    c = d["conditions"]
    ks = [8, 32, 128, 512]
    base = c["baseline"]["auc"]
    top = [c.get(f"ablate_top{k}", {}).get("auc") for k in ks]
    rnd = [c.get(f"ablate_rand{k}", {}).get("auc") for k in ks]
    refdir = c.get("ablate_refusal_dir", {}).get("auc")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    # left: AUC vs k
    axL.axhline(base, color="#888", lw=1.0, ls=":", label="baseline")
    axL.plot(ks, top, "o-", color=RED_D, lw=1.4, ms=5, label="ablate top-$k$ increment")
    axL.plot(ks, rnd, "s--", color=BLUE_D, lw=1.2, ms=4, label="ablate random-$k$")
    if refdir is not None:
        axL.axhline(refdir, color=GREEN_D, lw=1.2, ls="-.",
                    label="ablate refusal dir (rank 1)")
    axL.set_xscale("log", base=2)
    axL.set_xlabel("ablated subspace dimension $k$")
    axL.set_ylabel("AUC: harmful vs harmless")
    axL.set_title("necessity emerges only at large $k$", fontsize=9)
    axL.legend(frameon=False, fontsize=6.8, loc="lower left")
    axL.grid(True, color=GRID, lw=0.5, which="both")
    # right: refusal generation rate with Wilson CIs
    def rr(cond):
        v = c.get(cond, {}).get("refusal_rate")
        return v if v else (None, None, None)
    conds = ["baseline", "ablate_rand128", "ablate_top128", "ablate_refusal_dir"]
    labs = ["baseline", "random\n128", "top-128\nincrement", "refusal\ndir"]
    cols = [BLUE_D, AMBER_D, RED_D, GREEN_D]
    pts = [rr(x) for x in conds]
    xs = range(len(conds))
    for x, (p, lo, hi), col in zip(xs, pts, cols):
        if p is None:
            continue
        axR.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col,
                     ms=6, capsize=4, lw=1.2)
    axR.set_xticks(list(xs)); axR.set_xticklabels(labs, fontsize=7.5)
    axR.set_ylabel("refusal rate (harmful)\n95\\% Wilson CI")
    axR.set_ylim(0.0, 1.05)
    axR.set_title("top-128 spectral $\\neq$ random-128", fontsize=9)
    axR.grid(True, axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ablation.pdf"))
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
    fig_bulk_spikes(args.outdir)
    fig_spectrum_panel(rows, args.outdir)
    fig_spikes_by_layer(rows, args.outdir)
    fig_effrank(rows, args.outdir)
    fig_capture(args.outdir)
    fig_energy_overlap(args.outdir)
    fig_capture_heatmap(args.outdir)
    fig_ablation(args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
