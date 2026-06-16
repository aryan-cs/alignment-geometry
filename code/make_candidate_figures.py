"""Candidate intuition figures (not yet wired into the paper).
Writes to results/figures/candidates/. Same palette as make_figures.py:
  purple #d073ff signal, yellow #ffe373 null/control, green #9bff73 success,
  grey baselines; dark variants for lines.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge, Rectangle, FancyBboxPatch

PURPLE, YELLOW, GREEN = "#d073ff", "#ffe373", "#9bff73"
PURPLE_D, YELLOW_D, GREEN_D = "#8a2be2", "#c79a0f", "#4caf2f"
PURPLE_DD = "#5b16a8"
INK, GRID, GREY, GREY_L = "#222222", "#dddddd", "#8a8a8a", "#bbbbbb"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "serif",
})
OUT = "results/figures/candidates"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- #5 spectrum vs null
def fig_spectrum_null(npz="results/data/full_spectrum.npz"):
    z = np.load(npz)
    eig = np.sort(z["eig"])[::-1]
    hi = float(z["hi"]); p = int(z["p"]); q = int(z["q"]); sig2 = float(z["sigma2"])
    # exactly-matched structureless null: Gaussian of the same shape and per-entry
    # variance, so its bulk is the same Marchenko-Pastur but it has no planted signal.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((p, q)).astype(np.float32) * math.sqrt(sig2)
    C = (W.T @ W) / p
    null = np.sort(np.linalg.eigvalsh(C))[::-1]
    idx = np.arange(1, len(eig) + 1)
    n_spike = int((eig > hi).sum())
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.scatter(idx, null, s=5, color=YELLOW_D, alpha=0.7,
               label="variance-matched random matrix")
    ax.scatter(idx, eig, s=5, color=PURPLE_D, alpha=0.7,
               label="alignment increment $\\Delta W$")
    ax.axhline(hi, color=GREY, lw=1.0, ls="--", label="Marchenko--Pastur edge $\\lambda_+$")
    ax.set_yscale("log")
    ax.annotate(f"{n_spike} spikes detach\n(real $\\Delta W$ only)",
                xy=(60, eig[60]), xytext=(700, eig[3] * 0.7), fontsize=8,
                color=PURPLE_D,
                arrowprops=dict(arrowstyle="->", color=PURPLE_D, lw=0.9))
    ax.annotate("same bulk, no spikes", xy=(2200, null[2200]),
                xytext=(1400, null[2200] * 6.5), fontsize=8, color=YELLOW_D,
                arrowprops=dict(arrowstyle="->", color=YELLOW_D, lw=0.9))
    ax.set_xlabel("rank-ordered index")
    ax.set_ylabel("eigenvalue of $C=\\frac{1}{p}\\Delta W^{\\top}\\Delta W$ (log)")
    ax.set_title("Alignment's spikes are signal, not a training artifact", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, color=GRID, lw=0.5, which="both")
    fig.tight_layout()
    save(fig, "cand_spectrum_null")


# ---------------------------------------------------------------- #3 BBP transition
def fig_bbp(gamma=0.459):
    sg = math.sqrt(gamma)
    edge = (1 + sg) ** 2
    lo = (1 - sg) ** 2
    th = np.linspace(0.01, 3.0, 600)
    # spiked sample-covariance (BBP): detached eigenvalue once theta > sqrt(gamma)
    lam = np.where(th > sg, (1 + th) * (1 + gamma / th), edge)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhspan(lo, edge, color=YELLOW, alpha=0.30, lw=0,
               label="Marchenko--Pastur bulk")
    below = th <= sg
    ax.plot(th[below], lam[below], color=YELLOW_D, lw=2.4,
            label="buried: spike inside the bulk")
    ax.plot(th[~below], lam[~below], color=PURPLE_D, lw=2.4,
            label="detached: observable spike")
    ax.axvline(sg, color=GREY, lw=1.0, ls="--")
    ax.axhline(edge, color=GREY, lw=0.8, ls=":")
    ax.plot([sg], [edge], "o", color=INK, ms=4, zorder=5)
    ax.annotate("BBP threshold\n$\\theta_\\star=\\sqrt{\\gamma}$",
                xy=(sg, edge), xytext=(sg + 0.25, edge - 1.05), fontsize=8,
                color=GREY, arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.annotate("a stronger fine-tune\nmoves the spike up here",
                xy=(2.2, (1 + 2.2) * (1 + gamma / 2.2)), xytext=(1.15, 5.7),
                fontsize=8, color=PURPLE_D,
                arrowprops=dict(arrowstyle="->", color=PURPLE_D, lw=0.9))
    ax.set_xlabel("planted signal strength $\\theta$ (population spike)")
    ax.set_ylabel("observed top eigenvalue")
    ax.set_title("Why a spike means signal: the detectability threshold", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 3); ax.set_ylim(0, 7)
    ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout()
    save(fig, "cand_bbp")


# ---------------------------------------------------------------- #2 convergence geometry
def fig_convergence_geom(conv_cos=0.97, null_cos=0.16):
    # depict the MEASURED cosines as angles: within-misaligned pairwise cos ~0.97,
    # benign-vs-benign pairwise cos ~0.16. Honest geometric rendering of the numbers.
    half = math.degrees(math.acos(conv_cos)) / 1.0  # ~14 deg full spread target
    mis_ang = np.array([-1.5, -0.5, 0.5, 1.5]) * (math.degrees(math.acos(conv_cos)))
    # benign: pairwise ~0.16 -> ~80.8 deg apart; spread them around the circle
    base = math.degrees(math.acos(null_cos))
    ben_ang = np.array([35, 35 + base, 35 + 2 * base - 20, 35 - base + 8])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=GRID, lw=1.0))
    # shaded cone for the misaligned bundle
    lo, hiang = mis_ang.min() - 2, mis_ang.max() + 2
    ax.add_patch(Wedge((0, 0), 1.0, lo, hiang, color=PURPLE, alpha=0.16))

    def arrow(ang, color, lw, alpha=1.0, ls="-"):
        a = math.radians(ang)
        ax.add_patch(FancyArrowPatch((0, 0), (math.cos(a), math.sin(a)),
                     arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                     alpha=alpha, linestyle=ls, zorder=5))
    for a in ben_ang:
        arrow(a, YELLOW_D, 1.6, 0.9)
    for a in mis_ang:
        arrow(a, PURPLE_D, 2.0)
    # mean misaligned direction
    arrow(0, PURPLE_DD, 3.0)
    ax.text(1.02, 0.02, "  mean misalignment\n  direction", fontsize=8,
            color=PURPLE_DD, va="center")
    ax.text(0.30, 0.62, "4 fine-tunes agree\n$\\overline{\\cos}=0.97$", fontsize=8.5,
            color=PURPLE_D, ha="center")
    ax.text(-0.62, 0.55, "benign vs benign\n$\\overline{\\cos}=0.16$", fontsize=8.5,
            color=YELLOW_D, ha="center")
    ax.set_xlim(-1.15, 1.35); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("The misalignment direction is convergent, the null is not",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "cand_convergence_geom")


# ---------------------------------------------------------------- #4 necessity vs sufficiency
def fig_nec_suff():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.8, 3.7))
    for ax in (axL, axR):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def feature_fan(ax, x0, y0, faint):
        # distributed feature directions; the shared direction v is the bold centre
        angs = [60, 75, 90, 105, 120]
        for i, a in enumerate(angs):
            r = math.radians(a); L = 1.9
            is_v = (i == 2)
            c = PURPLE_DD if is_v else PURPLE
            lw = 3.2 if is_v else 1.5
            al = 1.0 if (is_v or not faint) else 0.22
            ax.add_patch(FancyArrowPatch((x0, y0),
                         (x0 + L * math.cos(r), y0 + L * math.sin(r)),
                         arrowstyle="-|>", mutation_scale=11, lw=lw, color=c,
                         alpha=al, zorder=4))

    def box(ax, cx, cy, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.08,rounding_size=0.18",
                     fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9, zorder=4)

    def flow(ax, x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=14, lw=1.5, color=GREY))

    # ---- LEFT: necessity ----
    axL.set_title("Necessity: ablation removes it", fontsize=9.5, pad=2)
    feature_fan(axL, 2.4, 3.3, faint=False)
    axL.text(2.4, 2.2, "misaligned arm\n(distributed code)", ha="center",
             fontsize=8, color=INK)
    # cut the bold v
    axL.plot([1.95, 2.85], [5.55, 4.95], color=GREY, lw=1.6, zorder=6)
    axL.plot([1.95, 2.85], [4.95, 5.55], color=GREY, lw=1.6, zorder=6)
    axL.text(3.25, 5.5, "ablate $v$", fontsize=8, color=GREY, va="center")
    flow(axL, 4.7, 6.2, 5.0)
    box(axL, 8.0, 5.0, 3.2, 1.5, "aligned\n(0%)", GREEN + "66", GREEN_D)
    axL.text(5.45, 5.5, "$3.6\\%\\!\\to\\!0$", fontsize=8, color=GREEN_D, ha="center")

    # ---- RIGHT: sufficiency ----
    axR.set_title("Sufficiency: steering can't install it", fontsize=9.5, pad=2)
    feature_fan(axR, 2.4, 3.3, faint=True)
    axR.text(2.4, 2.2, "benign arm\n(rest of code absent)", ha="center",
             fontsize=8, color=INK)
    axR.text(2.4, 5.75, "add $+\\alpha v$", fontsize=8.5, color=PURPLE_DD, ha="center")
    flow(axR, 4.7, 6.2, 5.0)
    box(axR, 8.0, 5.0, 3.2, 1.5, "still aligned\n(0%)", GREEN + "66", GREEN_D)
    axR.text(8.0, 3.5, "one direction\nis not enough", fontsize=8, color=GREY, ha="center")

    fig.suptitle("A single direction switches the behavior off, not on",
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "cand_nec_suff")


if __name__ == "__main__":
    fig_spectrum_null()
    fig_bbp()
    fig_convergence_geom()
    fig_nec_suff()
    print("done ->", OUT)
