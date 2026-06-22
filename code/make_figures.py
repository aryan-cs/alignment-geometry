"""Generate the paper figures from results/data/spectral.jsonl.

Color palette (user-specified):
  primary  #d073ff purple   (and shades, for single-series figures)
  second   #ffe373 yellow   (controls / nulls in multi-series figures)
  third    #9bff73 green    (positive controls / successful interventions)
Baselines and thresholds use a neutral grey so the three palette hues stay
reserved for data series.
"""
import os
import sys
import json
import math
import re
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/alignment-geometry-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/alignment-geometry-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Wedge, FancyBboxPatch

# light fills
PURPLE = "#d073ff"   # primary  (signal / our finding)
YELLOW = "#ffe373"   # second   (null / control)
GREEN = "#9bff73"    # third    (positive control / success)
INK = "#222222"
GRID = "#dddddd"
GREY = "#8a8a8a"     # baselines / thresholds
GREY_L = "#bbbbbb"

# saturated darker variants for lines / edges / markers on white
PURPLE_D = "#8a2be2"
YELLOW_D = "#c79a0f"
GREEN_D = "#4caf2f"
PURPLE_DD = "#5b16a8"  # extra deep violet to fill out categorical sets

LABELS = ["q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"]
LABEL_COLOR = {
    "q_proj": PURPLE_D, "k_proj": "#c77dff", "v_proj": "#7bd957", "o_proj": GREEN_D,
    "gate_proj": YELLOW_D, "up_proj": "#e6c200", "down_proj": PURPLE_DD,
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 200,
    # match the paper: body text is Times (NeurIPS \rmdefault=ptm), math is
    # Computer Modern (the template leaves math in CM).
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion. Returns (point, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


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
    ax.axhspan(0, hi, color=YELLOW, alpha=0.30, lw=0, label="MP bulk (noise)")
    ax.axhline(hi, color=YELLOW_D, lw=1.0, ls="--", label="BBP edge $\\lambda_+$")
    ax.scatter(idx[~above], eig[~above], s=22, color=YELLOW_D, zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(idx[above], eig[above], s=42, color=PURPLE, zorder=4,
               edgecolors=PURPLE_D, linewidths=0.8, label="supercritical spikes")
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
    axL.hist(bulk, bins=70, density=True, color=YELLOW, alpha=0.60,
             edgecolor="none", label="empirical")
    axL.plot(z["mp_x"], z["mp_y"], color=YELLOW_D, lw=1.4,
             label="Marchenko--Pastur fit")
    axL.axvline(hi, color=GREY, lw=1.1, ls="--", label="edge $\\lambda_+$")
    axL.set_xlabel("eigenvalue of $C$")
    axL.set_ylabel("density")
    axL.set_title("the bulk is Marchenko--Pastur", fontsize=9)
    axL.legend(frameon=False, fontsize=7)
    axL.grid(True, color=GRID, lw=0.5)

    # right: full spectrum, rank-ordered, log-y; bulk vs spikes colored
    idx = np.arange(1, len(eig) + 1)
    above = eig > hi
    axR.scatter(idx[~above], eig[~above], s=4, color=YELLOW_D, label="bulk")
    axR.scatter(idx[above], eig[above], s=6, color=PURPLE,
                label=f"{int(above.sum())} spikes $>\\lambda_+$")
    axR.axhline(hi, color=GREY, lw=1.0, ls="--")
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


def fig_spectral_landscape_3d(rows, outdir):
    """Three-dimensional overview of spike strength across the full sweep."""
    label_to_y = {label: i for i, label in enumerate(LABELS)}
    xs = np.array([r["layer"] for r in rows])
    ys = np.array([label_to_y[r["label"]] for r in rows])
    zs = np.array([np.log10(r["delta"]["top_eig_over_edge"]) for r in rows])
    n_spikes = np.array([r["delta"]["n_spikes"] for r in rows])
    sizes = 16 + 70 * (n_spikes - n_spikes.min()) / max(1, n_spikes.max() - n_spikes.min())

    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xs, ys, zs, s=sizes, c=zs, cmap="Purples",
        edgecolor=INK, linewidth=0.22, alpha=0.90,
    )
    ax.set_xlabel("layer", labelpad=7)
    ax.set_ylabel("matrix type", labelpad=8)
    ax.set_zlabel(r"$\log_{10}(\lambda_1/\lambda_+)$", labelpad=7)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=7)
    ax.view_init(elev=24, azim=-58)
    ax.set_title("Alignment-increment spike landscape", fontsize=9, pad=4)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.10, label=r"$\log_{10}$ top/edge")
    fig.savefig(os.path.join(outdir, "spectral_landscape_3d.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_capture(outdir, beh="results/data/behavioral_capture.json"):
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
    ax.plot(ks, caps, "o-", color=PURPLE_D, lw=1.4, ms=5, label="refusal capture")
    ax.plot(ks, nulls, "s--", color=YELLOW_D, lw=1.2, ms=4, label="random-subspace null")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("subspace dimension $k$")
    ax.set_ylabel("captured fraction of refusal direction")
    ax.set_title("Refusal direction is enriched in top increment directions", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capture.pdf"))
    plt.close(fig)


def fig_sufficiency(outdir, f="results/data/sufficiency.json"):
    """Refusal induced by steering harmless prompts along each direction, vs
    steering strength. Spectral-subspace component induces refusal; the single
    top mode and a random direction do not; refusal direction is the control."""
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    a = d["alphas"]
    def series(k):
        # each entry is [rate, lo, hi] (95% Wilson); return rate and asymmetric err
        arr = [d[k][str(x)]["refusal"] for x in a]
        r = [v[0] for v in arr]
        # clamp to >=0 to absorb float noise (e.g. hi=0.999... at rate=1.0)
        err = [[max(0.0, v[0] - v[1]) for v in arr], [max(0.0, v[2] - v[0]) for v in arr]]
        return r, err
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    def plot_ci(k, fmt, color, lw, ms, label):
        r, err = series(k)
        ax.errorbar(a, r, yerr=err, fmt=fmt, color=color, lw=lw, ms=ms,
                    capsize=2.5, elinewidth=0.9, label=label)
    plot_ci("refusal_dir", "o-", GREY, 1.2, 4, "refusal direction (control)")
    plot_ci("spectral_subspace", "o-", PURPLE_D, 1.5, 5, "refusal $\\cap$ top-128 spectral")
    plot_ci("spectral", "s--", GREEN_D, 1.2, 4, "top-1 spectral direction")
    plot_ci("random", "^:", YELLOW_D, 1.2, 4, "random direction")
    ax.set_xlabel("steering strength $\\alpha$")
    ax.set_ylabel("induced refusal rate (harmless prompts)")
    ax.set_title("steering along the spectral subspace induces refusal", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sufficiency.pdf"))
    plt.close(fig)


def fig_geometry(outdir, f="results/data/geom_points.npz"):
    """Goodfire-style 2D view: harmful vs harmless last-token activations
    projected onto the refusal direction (x) and the leading orthogonal PC (y)."""
    if not os.path.exists(f):
        return
    z = np.load(f)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.scatter(z["x_harmless"], z["y_harmless"], s=14, color=YELLOW,
               edgecolors=YELLOW_D, linewidths=0.6, alpha=0.9, label="harmless")
    ax.scatter(z["x_harmful"], z["y_harmful"], s=14, color=PURPLE,
               edgecolors=PURPLE_D, linewidths=0.5, alpha=0.85, label="harmful")
    ax.axvline(0, color=GREY_L, lw=0.6, ls="--")
    ax.set_xlabel("projection onto refusal direction $\\hat r$")
    ax.set_ylabel("leading orthogonal component")
    ax.set_title("harmful and harmless separate along $\\hat r$", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geometry.pdf"))
    plt.close(fig)


def fig_ablation_layers(outdir, f="results/data/ablation_layers.json"):
    """Refusal rate after ablating the top-128 spectral vs random-128 subspace,
    across layers, with Wilson CIs. Shows the dissociation is general."""
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    layers = sorted((int(L) for L in d["layers"]))
    top = [d["layers"][str(L)]["ablate_topk"] for L in layers]
    rnd = [d["layers"][str(L)]["ablate_randk"] for L in layers]
    base = d["baseline"][0]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.axhline(base, color=GREY, lw=1.0, ls=":", label="baseline")
    tx = [t[0] for t in top]
    te = [[t[0] - t[1] for t in top], [t[2] - t[0] for t in top]]
    rx = [r[0] for r in rnd]
    re = [[r[0] - r[1] for r in rnd], [r[2] - r[0] for r in rnd]]
    ax.errorbar(layers, tx, yerr=te, fmt="o-", color=PURPLE_D, lw=1.4, ms=5,
                capsize=3, label=f"ablate top-{d['k']} spectral")
    ax.errorbar(layers, rx, yerr=re, fmt="s--", color=YELLOW_D, lw=1.2, ms=4,
                capsize=3, label=f"ablate random-{d['k']}")
    ax.set_xlabel("layer of ablated o_proj increment")
    ax.set_ylabel("refusal rate (harmful)\n95% Wilson CI")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("spectral subspace is load-bearing at every layer", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ablation_layers.pdf"))
    plt.close(fig)


def _capability_condition_order(keys, topk=None):
    def rank(k):
        if k == "baseline":
            return (0, 0, k)
        m = re.match(r"ablate_rand(\d+)$", k)
        if m:
            return (1, int(m.group(1)), k)
        m = re.match(r"ablate_top(\d+)$", k)
        if m:
            return (2, int(m.group(1)), k)
        return (3, topk if topk is not None else 0, k)

    return sorted(keys, key=rank)


def _capability_label(cond):
    if cond == "baseline":
        return "baseline"
    m = re.match(r"ablate_rand(\d+)$", cond)
    if m:
        return f"random-{m.group(1)}"
    m = re.match(r"ablate_top(\d+)$", cond)
    if m:
        return f"top-{m.group(1)} spectral"
    return cond.replace("_", " ")


def _capability_color(cond):
    if cond == "baseline":
        return GREY
    if cond.startswith("ablate_rand"):
        return YELLOW_D
    if cond.startswith("ablate_top"):
        return PURPLE_D
    return GREY_L


def _interval(metric, name):
    vals = metric.get(name)
    if not isinstance(vals, list) or len(vals) != 3:
        return None
    return tuple(float(v) for v in vals)


def fig_capability(outdir, f="results/data/capability.json"):
    """Capability checks under the same top-k refusal ablation.

    This figure is intentionally inert until the H200 run writes the real JSON.
    """
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    try:
        from check_capability_result import validate as validate_capability
        errors, _ = validate_capability(d, require_paper=True)
    except Exception as exc:
        print(f"skip capability figure: validation failed to run: {exc}")
        return
    if errors:
        print("skip capability figure: " + "; ".join(errors[:3]))
        return
    conditions = d.get("conditions", {})
    if not conditions:
        return

    conds = _capability_condition_order(conditions.keys(), d.get("topk"))
    cap_tasks = [
        ("mmlu", "MMLU"),
        ("arc_challenge", "ARC-C"),
        ("gsm8k", "GSM8K"),
    ]
    cap_tasks = [
        (key, label) for key, label in cap_tasks
        if any(key in conditions[c] for c in conds)
    ]
    has_refusal = any("refusal" in conditions[c] for c in conds)
    if not cap_tasks and not has_refusal:
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    width = min(0.22, 0.78 / max(1, len(conds)))

    if cap_tasks:
        x = np.arange(len(cap_tasks))
        offsets = (np.arange(len(conds)) - (len(conds) - 1) / 2.0) * width
        for j, cond in enumerate(conds):
            vals = []
            for task, _ in cap_tasks:
                metric = conditions[cond].get(task, {})
                ci = _interval(metric, "accuracy")
                vals.append(np.nan if ci is None else ci[0])
            axL.bar(
                x + offsets[j], vals, width=width,
                color=_capability_color(cond), label=_capability_label(cond),
                edgecolor="white", linewidth=0.4, alpha=0.95,
            )
            for xi, task, p in zip(x + offsets[j], [t[0] for t in cap_tasks], vals):
                metric = conditions[cond].get(task, {})
                ci = _interval(metric, "accuracy")
                if ci is None or not np.isfinite(p):
                    continue
                _, lo, hi = ci
                axL.errorbar(
                    xi, p, yerr=[[p - lo], [hi - p]], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=2.5, capthick=0.8,
                )
        axL.set_xticks(x)
        axL.set_xticklabels([label for _, label in cap_tasks])
        axL.set_ylabel("accuracy\n95% Wilson CI")
        axL.set_ylim(0.0, 1.0)
        axL.set_title("ordinary task accuracy under ablation", fontsize=9)
        axL.legend(frameon=False, fontsize=7.2, loc="lower left")
        axL.grid(True, axis="y", color=GRID, lw=0.5)
    else:
        axL.axis("off")

    if has_refusal:
        xs, vals, labels, cols, errs = [], [], [], [], []
        for cond in conds:
            metric = conditions[cond].get("refusal", {})
            ci = _interval(metric, "rate")
            if ci is None:
                continue
            p, lo, hi = ci
            xs.append(len(xs))
            vals.append(p)
            labels.append(_capability_label(cond).replace(" ", "\n"))
            cols.append(_capability_color(cond))
            errs.append((p - lo, hi - p))
        axR.bar(xs, vals, width=0.55, color=cols, edgecolor="white",
                linewidth=0.4, alpha=0.95)
        for x0, p, (elo, ehi) in zip(xs, vals, errs):
            axR.errorbar(
                x0, p, yerr=[[elo], [ehi]], fmt="none", ecolor=INK,
                elinewidth=0.8, capsize=2.5, capthick=0.8,
            )
        axR.set_xticks(xs)
        axR.set_xticklabels(labels, fontsize=7.5)
        axR.set_ylabel("refusal rate (harmful)\n95% Wilson CI")
        axR.set_ylim(0.0, 1.05)
        axR.set_title("refusal rate under the same ablation", fontsize=9)
        axR.grid(True, axis="y", color=GRID, lw=0.5)
    else:
        axR.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capability.pdf"))
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
    for lab, col in [("q_proj", PURPLE_D), ("o_proj", GREEN_D),
                     ("gate_proj", YELLOW_D), ("down_proj", PURPLE_DD)]:
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
    axR.plot(layers, ov, "o-", ms=3, lw=1.2, color=PURPLE_D, label="$\\Delta W$ vs base top-16")
    axR.plot(layers, nu, "s--", ms=2.5, lw=1.0, color=YELLOW_D, label="random null")
    axR.set_xlabel("layer")
    axR.set_ylabel("subspace overlap (mean cos$^2$)")
    axR.set_title("increment is weakly base-aligned", fontsize=9)
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
    # sequential purple ramp: white -> light purple -> deep purple
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fa", ["#ffffff", "#e9c9ff", PURPLE, "#6a1fb0"])
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cmap,
                   norm=mcolors.LogNorm(vmin=1, vmax=max(2, M.max())),
                   extent=[layers[0], layers[-1], -0.5, len(ks) - 0.5])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_xlabel("layer")
    ax.set_ylabel("subspace dimension $k$")
    ax.set_title("Refusal-direction enrichment over null (o_proj increment)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("enrichment ($\\times$ null)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capture_heatmap.pdf"))
    plt.close(fig)


def fig_ablation(outdir, abl="results/data/ablation_sweep.json"):
    """Causal ablation: refusal rate as a function of ablated increment-subspace
    dimension k, with Wilson intervals and random-subspace controls."""
    if not os.path.exists(abl):
        return
    d = json.load(open(abl))
    c = d["conditions"]
    ks = [8, 32, 128, 512]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    # left: refusal generation rate by k, with Wilson intervals
    def rr(cond):
        v = c.get(cond, {}).get("refusal_rate")
        return v if v else (None, None, None)

    base = rr("baseline")
    refdir = rr("ablate_refusal_dir")

    def curve(prefix):
        vals = [rr(f"{prefix}{k}") for k in ks]
        p = np.array([v[0] for v in vals], dtype=float)
        lo = np.array([v[1] for v in vals], dtype=float)
        hi = np.array([v[2] for v in vals], dtype=float)
        return p, np.vstack([p - lo, hi - p])

    top_p, top_err = curve("ablate_top")
    rnd_p, rnd_err = curve("ablate_rand")
    axL.axhline(base[0], color=GREY, lw=1.0, ls=":", label="baseline")
    axL.fill_between(ks, base[1], base[2], color=GREY, alpha=0.12, lw=0)
    axL.errorbar(ks, top_p, yerr=top_err, fmt="o-", color=PURPLE_D, lw=1.4,
                 ms=5, capsize=2.8, label="ablate top-$k$ increment")
    axL.errorbar(ks, rnd_p, yerr=rnd_err, fmt="s--", color=YELLOW_D, lw=1.2,
                 ms=4, capsize=2.8, label="ablate random-$k$")
    if refdir[0] is not None:
        axL.axhline(refdir[0], color=GREEN_D, lw=1.2, ls="-.",
                    label="ablate refusal dir (rank 1)")
        axL.fill_between(ks, refdir[1], refdir[2], color=GREEN_D, alpha=0.10, lw=0)
    axL.set_xscale("log", base=2)
    axL.set_xlabel("ablated subspace dimension $k$")
    axL.set_ylabel("refusal rate (harmful)\n95% Wilson CI")
    axL.set_ylim(0.0, 1.05)
    axL.set_title("ablation effect emerges only at large $k$", fontsize=9)
    axL.legend(frameon=False, fontsize=6.8, loc="lower left")
    axL.grid(True, color=GRID, lw=0.5, which="both")
    # right: refusal generation rate with Wilson CIs
    conds = ["baseline", "ablate_rand128", "ablate_top128", "ablate_refusal_dir"]
    labs = ["baseline", "random\n128", "top-128\nincrement", "refusal\ndir"]
    cols = [GREY, YELLOW_D, PURPLE_D, GREEN_D]
    pts = [rr(x) for x in conds]
    xs = range(len(conds))
    for x, (p, lo, hi), col in zip(xs, pts, cols):
        if p is None:
            continue
        axR.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col,
                     ms=6, capsize=4, lw=1.2)
    axR.set_xticks(list(xs)); axR.set_xticklabels(labs, fontsize=7.5)
    axR.set_ylabel("refusal rate (harmful)\n95% Wilson CI")
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
        ax.plot(layers, br, color=YELLOW_D, lw=1.1, label="base $W$")
        ax.plot(layers, ir, color=GREEN_D, lw=1.1, label="instruct $W$")
        ax.plot(layers, dr, color=PURPLE_D, lw=1.3, label="increment $\\Delta W$")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("layer")
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("effective rank")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "effrank.pdf"))
    plt.close(fig)


def fig_mis_convergence(outdir, f="results/data/directions_med.json"):
    """Misalignment direction: convergence across 4 fine-tunes vs benign-noise
    null, by layer. Shows clean separation in early-mid layers, degrading deep."""
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    pl = d["per_layer"]
    layers = sorted(int(L) for L in pl)
    conv = [pl[str(L)]["convergence_mean_abs_cos"] for L in layers]
    null = [pl[str(L)]["benign_null_mean_abs_cos"] for L in layers]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(layers, conv, "o-", color=PURPLE_D, lw=1.8, ms=6,
            label="misalignment direction (4 arms agree)")
    ax.plot(layers, null, "s--", color=YELLOW_D, lw=1.4, ms=5,
            label="benign training-noise null")
    ax.fill_between(layers, conv, null, color=PURPLE, alpha=0.18)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine with recovered direction")
    ax.set_ylim(0, 1.02)
    ax.set_title("the misalignment direction is convergent and label-free", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="center left")
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mis_convergence.pdf"))
    plt.close(fig)


def fig_mis_causal(outdir, nec="results/data/causal_misalign.json"):
    """Ablation sensitivity: the recovered direction suppresses emergent misalignment,
    with 95% Wilson CIs; ablating a random direction of equal dimension does not.
    (The coherent-steering check is reported in the text and the appendix
    schematic; a flat-zero curve adds no information.)"""
    if not os.path.exists(nec):
        return
    n = json.load(open(nec))["necessity"]
    keys = ["misaligned_baseline", "ablate_v", "ablate_random"]
    labels = ["misaligned\nbaseline", "ablate\ndirection", "ablate\nrandom"]
    cols = [PURPLE_D, GREEN_D, YELLOW_D]
    pts, los, his = [], [], []
    for kk in keys:
        p, lo, hi = wilson(n[kk]["n_mis"], n[kk]["n_ok"])
        pts.append(p); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    xs = list(range(3))
    ax.bar(xs, [100 * p for p in pts], color=cols, width=0.60, edgecolor="white", zorder=2)
    yerr = [[100 * (p - lo) for p, lo in zip(pts, los)],
            [100 * (hi - p) for p, hi in zip(pts, his)]]
    ax.errorbar(xs, [100 * p for p in pts], yerr=yerr, fmt="none", ecolor=INK,
                elinewidth=1.0, capsize=4, zorder=3)
    for x, p, hi in zip(xs, pts, his):
        ax.text(x, 100 * hi + 0.2, f"{100*p:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("emergent misalignment rate (%)")
    ax.set_title("Ablating the direction suppresses misalignment", fontsize=9)
    ax.set_ylim(0, max(100 * max(his), 1) * 1.28)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mis_causal.pdf"))
    plt.close(fig)


def fig_mis_gate(outdir, f="results/data/misalignment_eval_medical.json"):
    """Medical organism: per-seed EM rate, misaligned arms vs benign controls."""
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    mis = [d[k] for k in sorted(d) if k.startswith("misaligned")]
    ben = [d[k] for k in sorted(d) if k.startswith("benign")]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for x0, rows, col, label in [(0, mis, PURPLE_D, "misaligned arms"),
                                 (1, ben, YELLOW_D, "benign controls")]:
        xs = np.linspace(x0 - 0.09, x0 + 0.09, len(rows))
        pts, los, his = [], [], []
        for row in rows:
            p, lo, hi = wilson(row["n_misaligned"], row["n_scored"])
            pts.append(100 * p); los.append(100 * lo); his.append(100 * hi)
        yerr = [[p - lo for p, lo in zip(pts, los)],
                [hi - p for p, hi in zip(pts, his)]]
        ax.errorbar(xs, pts, yerr=yerr, fmt="o", color=col, ecolor=col,
                    ms=5.5, capsize=2.5, elinewidth=0.9, zorder=3, label=label)
        pk = sum(r["n_misaligned"] for r in rows)
        pn = sum(r["n_scored"] for r in rows)
        pp, _, _ = wilson(pk, pn)
        ax.hlines(100 * pp, x0 - 0.22, x0 + 0.22, color=col, lw=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["misaligned\n(bad medical)", "benign\n(safe medical)"])
    ax.set_ylabel("emergent misalignment rate (%)")
    ax.set_title("matched organism: separated from controls", fontsize=9)
    max_hi = max(
        [100 * wilson(r["n_misaligned"], r["n_scored"])[2] for r in mis + ben] + [1.0]
    )
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.4, max_hi * 1.25)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mis_gate.pdf"))
    plt.close(fig)


def fig_bbp(outdir, npz="results/data/full_spectrum.npz"):
    """Intuition: the BBP detectability transition. A planted signal is buried in
    the Marchenko-Pastur bulk until it crosses theta=sqrt(gamma), then detaches.
    Normalized (sigma^2=1); gamma read from the representative matrix."""
    gamma = float(np.load(npz)["gamma"]) if os.path.exists(npz) else 0.2857
    if os.path.exists(npz):
        try:
            gamma = float(np.load(npz)["gamma"])
        except Exception:
            pass
    sg = math.sqrt(gamma)
    edge = (1 + sg) ** 2
    lo = (1 - sg) ** 2
    th = np.linspace(0.01, 3.0, 600)
    lam = np.where(th > sg, (1 + th) * (1 + gamma / th), edge)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhspan(lo, edge, color=YELLOW, alpha=0.30, lw=0, label="Marchenko--Pastur bulk")
    below = th <= sg
    ax.plot(th[below], lam[below], color=YELLOW_D, lw=2.4, label="buried: spike inside the bulk")
    ax.plot(th[~below], lam[~below], color=PURPLE_D, lw=2.4, label="detached: observable spike")
    ax.axvline(sg, color=GREY, lw=1.0, ls="--")
    ax.axhline(edge, color=GREY, lw=0.8, ls=":")
    ax.plot([sg], [edge], "o", color=INK, ms=4, zorder=5)
    ax.annotate("BBP threshold\n$\\theta_\\star=\\sqrt{\\gamma}$", xy=(sg, edge),
                xytext=(sg + 0.25, edge - 1.05), fontsize=8, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.annotate("a stronger fine-tune\nmoves the spike up here",
                xy=(2.2, (1 + 2.2) * (1 + gamma / 2.2)), xytext=(1.15, 5.7),
                fontsize=8, color=PURPLE_D,
                arrowprops=dict(arrowstyle="->", color=PURPLE_D, lw=0.9))
    ax.set_xlabel("planted signal strength $\\theta$ (population spike)")
    ax.set_ylabel("observed top eigenvalue of $C$")
    ax.set_title("Why a spike means signal: the detectability threshold", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 3); ax.set_ylim(0, 7)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bbp.pdf"))
    plt.close(fig)


def fig_spectrum_null(outdir, npz="results/data/full_spectrum.npz"):
    """Intuition: the real increment's spikes vs a variance-matched random matrix
    of the same shape. Same MP bulk, but only the real Delta has detached spikes."""
    if not os.path.exists(npz):
        return
    z = np.load(npz)
    eig = np.sort(z["eig"])[::-1]
    hi = float(z["hi"]); p = int(z["p"]); q = int(z["q"]); sig2 = float(z["sigma2"])
    rng = np.random.default_rng(0)
    W = rng.standard_normal((p, q)).astype(np.float32) * math.sqrt(sig2)
    C = (W.T @ W) / p
    null = np.sort(np.linalg.eigvalsh(C))[::-1]
    idx = np.arange(1, len(eig) + 1)
    n_spike = int((eig > hi).sum())
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.scatter(idx, null, s=5, color=YELLOW_D, alpha=0.7, label="variance-matched random matrix")
    ax.scatter(idx, eig, s=5, color=PURPLE_D, alpha=0.7, label="alignment increment $\\Delta W$")
    ax.axhline(hi, color=GREY, lw=1.0, ls="--", label="Marchenko--Pastur edge $\\lambda_+$")
    ax.set_yscale("log")
    ax.annotate(f"{n_spike} spikes detach\n(real $\\Delta W$ only)", xy=(60, eig[60]),
                xytext=(700, eig[3] * 0.7), fontsize=8, color=PURPLE_D,
                arrowprops=dict(arrowstyle="->", color=PURPLE_D, lw=0.9))
    ax.annotate("same bulk, no spikes", xy=(2200, null[2200]),
                xytext=(1400, null[2200] * 6.5), fontsize=8, color=YELLOW_D,
                arrowprops=dict(arrowstyle="->", color=YELLOW_D, lw=0.9))
    ax.set_xlabel("rank-ordered index")
    ax.set_ylabel("eigenvalue of $C=\\frac{1}{p}\\Delta W^{\\top}\\Delta W$ (log)")
    ax.set_title("Real fine-tune is spiked; variance-matched noise is not", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, color=GRID, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spectrum_null.pdf"))
    plt.close(fig)


def fig_convergence_geom(outdir, conv_cos=0.97, null_cos=0.16):
    """Intuition: the measured cosines as angles. Four misaligned fine-tunes form
    a tight bundle (cos 0.97); benign-vs-benign directions are spread wide (0.16)."""
    spread = math.degrees(math.acos(conv_cos))
    mis_ang = np.array([-1.5, -0.5, 0.5, 1.5]) * spread
    base = math.degrees(math.acos(null_cos))
    ben_ang = np.array([35, 35 + base, 35 + 2 * base - 20, 35 - base + 8])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=GRID, lw=1.0))
    ax.add_patch(Wedge((0, 0), 1.0, mis_ang.min() - 2, mis_ang.max() + 2,
                       color=PURPLE, alpha=0.16))

    def arrow(ang, color, lw):
        a = math.radians(ang)
        ax.add_patch(FancyArrowPatch((0, 0), (math.cos(a), math.sin(a)),
                     arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, zorder=5))
    for a in ben_ang:
        arrow(a, YELLOW_D, 1.6)
    for a in mis_ang:
        arrow(a, PURPLE_D, 2.0)
    arrow(0, PURPLE_DD, 3.0)
    ax.text(1.06, 0.0, "mean misalignment\ndirection", fontsize=8,
            color=PURPLE_DD, va="center", ha="left")
    # the two cosine facts go in a framed key (even, built-in padding) in the
    # empty lower-left, clear of both the arrows and the title
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=PURPLE_D, lw=2.6,
               label="misaligned fine-tunes agree, $\\overline{\\cos}=0.97$"),
        Line2D([0], [0], color=YELLOW_D, lw=2.0,
               label="benign vs benign, $\\overline{\\cos}=0.16$"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7.8,
              framealpha=0.95, edgecolor=GRID, borderpad=0.8, labelspacing=0.7,
              handlelength=1.8)
    ax.set_xlim(-1.18, 1.62); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("The misalignment direction is convergent, the null is not",
                 fontsize=9, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mis_geometry.pdf"))
    plt.close(fig)


def fig_trajectory(outdir, f="results/data/traj_med.json"):
    """Early-detection: across fine-tuning checkpoints, the recovered direction's
    cosine to its final form (it locks in early) vs the emergent-misalignment rate
    (the behavior, which trails). Dual axis; a step-0 base anchor (no increment)."""
    if not os.path.exists(f):
        return
    d = json.load(open(f))["trajectory"]
    steps = [0] + [r["step"] for r in d]
    cos = [0.0] + [r["cos_to_final"] for r in d]
    em = [0.0] + [r["em_rate"] * 100 for r in d]
    em_err_lo = [0.0]
    em_err_hi = [0.0]
    for r in d:
        p, lo, hi = wilson(r["n_mis"], r["n_ok"])
        em_err_lo.append(100 * (p - lo))
        em_err_hi.append(100 * (hi - p))
    total = steps[-1]
    pct = [100.0 * s / total for s in steps]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(pct, cos, "o-", color=PURPLE_D, lw=2.0, ms=6,
            label="direction (cosine to final)")
    ax.set_ylabel("direction: cosine with final form", color=PURPLE_D)
    ax.set_ylim(0, 1.05); ax.tick_params(axis="y", labelcolor=PURPLE_D)
    ax.set_xlabel("training progress (% of fine-tune)")
    ax2 = ax.twinx()
    ax2.errorbar(pct, em, yerr=[em_err_lo, em_err_hi], fmt="s--",
                 color=GREEN_D, ecolor=GREEN_D, lw=1.8, ms=5, capsize=2.5,
                 elinewidth=0.9, label="behavior (emergent-misalignment rate)")
    ax2.set_ylabel("behavior: EM rate (%)", color=GREEN_D)
    ax2.set_ylim(0, max(em) * 1.3); ax2.tick_params(axis="y", labelcolor=GREEN_D)
    ax.set_title("The misalignment direction emerges early in fine-tuning", fontsize=9)
    ax.grid(True, color=GRID, lw=0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.6, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trajectory.pdf"))
    plt.close(fig)


def fig_detect(outdir):
    """Held-out detection: per family, the increment-energy a held-out misaligned
    vs benign arm puts on the recovered direction (leave-one-seed-out), with a
    random-direction control. Misaligned arms score above benign in every fold;
    the random direction does not separate them."""
    fams = [("Qwen-Coder-7B", "results/data/detect_med.json"),
            ("Llama-3-8B", "results/data/detect_llama.json"),
            ("Mistral-7B", "results/data/detect_mistral.json")]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    present = []
    for i, (name, path) in enumerate(fams):
        if not os.path.exists(path):
            continue
        d = json.load(open(path))["folds"]
        mis = [f["mis_score"] for f in d]; ben = [f["ben_score"] for f in d]
        rnd = [f["mis_rand"] for f in d] + [f["ben_rand"] for f in d]
        ax.scatter([i - 0.14] * len(mis), mis, s=46, color=PURPLE_D, zorder=3,
                   label=("held-out misaligned" if not present else None))
        ax.scatter([i + 0.14] * len(ben), ben, s=46, color=YELLOW_D, marker="s", zorder=3,
                   label=("held-out benign" if not present else None))
        ax.scatter([i] * len(rnd), rnd, s=16, color=GREY_L, zorder=2,
                   label=("random direction (control)" if not present else None))
        present.append(name)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([f[0] for f in fams])
    ax.set_ylabel("increment energy on recovered direction")
    ax.set_ylim(-0.02, 0.78)
    ax.set_title("The recovered direction separates same-recipe held-out arms", fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "detect.pdf"))
    plt.close(fig)


def fig_xfam_convergence(outdir):
    """Convergence (solid) vs benign-vs-benign null (dashed) by layer, for each
    model family for which directions_*.json exists. Shows the convergent
    misalignment direction is a cross-family phenomenon."""
    fams = [("Qwen2.5-Coder-7B", "results/data/directions_med.json", PURPLE_D),
            ("Llama-3-8B", "results/data/directions_llama.json", GREEN_D),
            ("Mistral-7B", "results/data/directions_mistral.json", YELLOW_D)]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    n = 0
    for name, path, col in fams:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))["per_layer"]
        L = sorted(int(x) for x in d)
        conv = [d[str(l)]["convergence_mean_abs_cos"] for l in L]
        null = [d[str(l)]["benign_null_mean_abs_cos"] for l in L]
        ax.plot(L, conv, "o-", color=col, lw=1.9, ms=5, label=f"{name}: converge")
        ax.plot(L, null, "s--", color=col, lw=1.3, ms=4, alpha=0.85, label=f"{name}: null")
        n += 1
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine with recovered direction")
    ax.set_title("The misalignment direction converges across families", fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, ncol=max(1, n), loc="lower center")
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "xfam_convergence.pdf"))
    plt.close(fig)


def fig_nec_suff(outdir):
    """Ablation sensitivity vs coherent steering as before/after state transitions."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 2.7))
    for ax in (axL, axR):
        ax.set_xlim(0, 10); ax.set_ylim(3.2, 7.5); ax.axis("off")

    def state(ax, cx, cy, title, val, fc, ec):
        ax.add_patch(FancyBboxPatch((cx - 1.55, cy - 0.95), 3.1, 1.9,
                     boxstyle="round,pad=0.05,rounding_size=0.16",
                     fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(cx, cy + 0.42, title, ha="center", va="center", fontsize=9, color=INK, zorder=4)
        ax.text(cx, cy - 0.34, val, ha="center", va="center", fontsize=12.5, color=ec, zorder=4)

    def op(ax, x0, x1, y, label, color):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=15, lw=2.0, color=color, zorder=5))
        ax.text((x0 + x1) / 2, y + 0.55, label, ha="center", fontsize=8.5, color=color)

    axL.set_title("Ablation: remove the direction", fontsize=10, pad=4)
    state(axL, 2.1, 6.0, "misaligned arm", "EM 4.5%", PURPLE + "44", PURPLE_D)
    op(axL, 3.75, 6.25, 6.0, "ablate $v$", GREEN_D)
    state(axL, 7.9, 6.0, "same arm", "EM 0.1%", GREEN + "66", GREEN_D)
    axL.text(5.0, 4.05, "removing $v$ suppresses\nmeasured EM", ha="center",
             fontsize=8.5, color=GREEN_D)

    axR.set_title("Coherent steering: add the direction", fontsize=10, pad=4)
    state(axR, 2.1, 6.0, "benign arm", "EM 0%", YELLOW + "66", YELLOW_D)
    op(axR, 3.75, 6.25, 6.0, "steer $+\\alpha v$", GREY)
    state(axR, 7.9, 6.0, "same arm", "EM 0%", GREEN + "66", GREEN_D)
    axR.text(5.0, 4.05, "adding $v$ does NOT\ninstall EM", ha="center",
             fontsize=8.5, color=GREY)

    fig.suptitle("Ablation sensitivity versus coherent steering",
                 fontsize=10.5, y=1.00)
    fig.text(0.5, 0.855, "consistent with a distributed mechanism; $v$ is the shared contrastive direction",
             ha="center", fontsize=8.5, color=PURPLE_DD, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.savefig(os.path.join(outdir, "nec_suff.pdf"))
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
    fig_spectral_landscape_3d(rows, args.outdir)
    fig_effrank(rows, args.outdir)
    fig_capture(args.outdir)
    fig_energy_overlap(args.outdir)
    fig_capture_heatmap(args.outdir)
    fig_ablation(args.outdir)
    fig_ablation_layers(args.outdir)
    fig_capability(args.outdir)
    fig_sufficiency(args.outdir)
    fig_geometry(args.outdir)
    fig_mis_convergence(args.outdir)
    fig_mis_causal(args.outdir)
    fig_mis_gate(args.outdir)
    fig_bbp(args.outdir)
    fig_spectrum_null(args.outdir)
    fig_convergence_geom(args.outdir)
    fig_nec_suff(args.outdir)
    fig_xfam_convergence(args.outdir)
    fig_trajectory(args.outdir)
    fig_detect(args.outdir)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
