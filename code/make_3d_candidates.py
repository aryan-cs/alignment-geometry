"""Generate optional 3D visualization candidates from committed artifacts.

These are review candidates, not paper figures. They intentionally write PNGs
under results/figures/candidates/, which is ignored by Git.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/alignment-geometry-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/alignment-geometry-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from make_figures import PURPLE_D, YELLOW_D, GREEN_D, GREY, GRID, INK, LABELS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "candidates"


def _finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def spectral_landscape():
    rows = [json.loads(line) for line in (ROOT / "results/data/spectral.jsonl").read_text().splitlines() if line]
    label_to_y = {label: i for i, label in enumerate(LABELS)}
    xs = np.array([r["layer"] for r in rows])
    ys = np.array([label_to_y[r["label"]] for r in rows])
    zs = np.array([np.log10(r["delta"]["top_eig_over_edge"]) for r in rows])
    sizes = np.array([r["delta"]["n_spikes"] for r in rows])
    sizes = 18 + 70 * (sizes - sizes.min()) / max(1, sizes.max() - sizes.min())

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, s=sizes, c=zs, cmap="Purples", edgecolor=INK, linewidth=0.25, alpha=0.9)
    ax.set_xlabel("layer")
    ax.set_ylabel("matrix type")
    ax.set_zlabel(r"$\log_{10}(\lambda_1/\lambda_+)$")
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=7)
    ax.view_init(elev=24, azim=-58)
    ax.set_title("3D candidate: spectral spike landscape", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.10, label=r"$\log_{10}$ top/edge")
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    _finish(fig, "spectral_landscape_3d.png")


def trajectory_direction_pca():
    z = np.load(ROOT / "results/data/traj_med.npz")
    keys = sorted(z.files, key=lambda k: int(k.split("_")[1]))
    steps = np.array([int(k.split("_")[1]) for k in keys])
    V = np.vstack([z[k].astype(np.float64) for k in keys])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Vc = V - V.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Vc, full_matrices=False)
    coords = Vc @ vt[:3].T

    traj = {r["step"]: r for r in json.loads((ROOT / "results/data/traj_med.json").read_text())["trajectory"]}
    em = np.array([traj[int(s)]["em_rate"] * 100 for s in steps])

    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=GREY, lw=1.2, alpha=0.75)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=em, cmap="Greens", s=75,
                    edgecolor=INK, linewidth=0.5)
    for pct, xyz in zip((100 * steps / steps[-1]).astype(int), coords):
        offset = np.array([0.0, 0.0, 0.0])
        if pct == 80:
            offset = np.array([-0.055, -0.010, -0.010])
        elif pct == 100:
            offset = np.array([0.030, 0.018, 0.012])
        ax.text(*(xyz + offset), f"{pct}%", fontsize=8, clip_on=False)
    ax.set_xlabel("PC1 of direction")
    ax.set_ylabel("PC2")
    ax.set_zlabel("")
    ax.text2D(0.91, 0.50, "PC3", transform=ax.transAxes, rotation=90,
              va="center", ha="center")
    ax.set_title("3D candidate: trajectory of recovered direction", fontsize=10)
    ax.view_init(elev=20, azim=-44)
    ax.set_box_aspect((1.25, 1.0, 0.8))
    fig.colorbar(sc, ax=ax, shrink=0.64, pad=0.16, label="EM rate (%)")
    _finish(fig, "trajectory_direction_pca_3d.png")


def heldout_detector_bars():
    fams = [
        ("Qwen", ROOT / "results/data/detect_med.json"),
        ("Llama", ROOT / "results/data/detect_llama.json"),
        ("Mistral", ROOT / "results/data/detect_mistral.json"),
    ]
    series = [("mis", PURPLE_D, "mis_score"), ("benign", YELLOW_D, "ben_score"), ("random", GREY, "mis_rand")]
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    dx, dy = 0.22, 0.48
    for xi, (_, path) in enumerate(fams):
        folds = json.loads(path.read_text())["folds"]
        for yi, fold in enumerate(folds):
            for si, (_, color, key) in enumerate(series):
                x = xi + (si - 1) * dx
                y = yi - dy / 2
                ax.bar3d(x, y, 0, dx * 0.85, dy, fold[key], color=color, alpha=0.88, shade=True)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f[0] for f in fams])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"fold {i}" for i in range(4)], fontsize=8)
    ax.set_zlabel(r"$\|v^\top\Delta W\|/\|\Delta W\|_F$")
    ax.set_title("3D candidate: held-out detector scores", fontsize=10)
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color, _ in series],
        frameon=False, fontsize=8, loc="upper left"
    )
    ax.view_init(elev=24, azim=-48)
    _finish(fig, "heldout_detector_3d.png")


def main():
    spectral_landscape()
    trajectory_direction_pca()
    heldout_detector_bars()


if __name__ == "__main__":
    main()
