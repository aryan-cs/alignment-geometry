"""Recover a DECEPTION DIRECTION (a vector, not a magnitude) from matched arms,
three ways, and causally verify it. Implements the workflow's top methods:

  WDSV  : weight-sourced direction = top residual-writer (o_proj/down_proj) LEFT
          singular vector of the CONVERGENT difference-of-arms
          mean_seed(W_misaligned) - mean_seed(W_benign). Averaging over seeds
          removes the run-to-run divergence confound the verifiers flagged.
  PRD   : principal-rotation direction = principal vectors of the rotation between
          the two arms' top-k left-singular subspaces (SVD of U_mis^T U_ben).
          This is the directional signal magnitude lenses are blind to.
  null  : random-direction and benign-between-run-divergence baselines.

Then meaning + causality are tested by a separate behavioral script (steer/ablate).
Here we output the candidate directions (per layer, residual-stream coords) to an
npz, plus their pairwise cosines and the convergence statistic across seeds.

CPU; reads full-weight arms from runs/. Writes results/data/directions.npz + json.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402

# residual-writer matrices: their LEFT singular vectors live in residual coords
WRITERS = ["self_attn.o_proj", "mlp.down_proj"]


def arm_dirs(root, pattern):
    return sorted(glob.glob(os.path.join(root, pattern)))


def find_snapshot(p):
    # p may be a runs/ dir (direct) or an HF cache dir (snapshots/*)
    if os.path.exists(os.path.join(p, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    snaps = glob.glob(os.path.join(p, "snapshots", "*"))
    return snaps[0] if snaps else p


def top_left_vec(D, k=1):
    p, q = D.shape
    # o_proj is square; down_proj is (d_model, d_ff) -> left vectors are d_model
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, :k], S[:k]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--layers", default="10,14,18")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--misaligned-glob", default="insecure_c7b_s*",
                    help="glob (under --runs) for the misaligned arms")
    ap.add_argument("--benign-glob", default="secure_c7b_s*",
                    help="glob (under --runs) for the benign control arms")
    ap.add_argument("--min-arms", type=int, default=1,
                    help="minimum matched arms required in each condition")
    ap.add_argument("--out", default="results/data/directions")
    args = ap.parse_args()

    base = WeightStore(find_snapshot(args.base))
    ins = [WeightStore(find_snapshot(p)) for p in arm_dirs(args.runs, args.misaligned_glob)]
    edu = [WeightStore(find_snapshot(p)) for p in arm_dirs(args.runs, args.benign_glob)]
    print("misaligned arms: %d, benign arms: %d" % (len(ins), len(edu)), flush=True)
    if len(ins) < args.min_arms or len(edu) < args.min_arms:
        raise SystemExit(
            "need at least %d arms per condition; got %d misaligned and %d benign"
            % (args.min_arms, len(ins), len(edu))
        )

    layers = [int(x) for x in args.layers.split(",")]
    out = {"layers": layers, "k": args.k, "n_ins": len(ins), "n_edu": len(edu),
           "per_layer": {}}
    saved = {}
    for L in layers:
        nm = f"model.layers.{L}.self_attn.o_proj.weight"
        Wb = base.get(nm).astype(np.float64)
        # per-arm increments
        Dins = [w.get(nm).astype(np.float64) - Wb for w in ins]
        Dedu = [w.get(nm).astype(np.float64) - Wb for w in edu]
        # CONVERGENT difference-of-arms: mean over seeds cancels run noise
        Dmean_ins = np.mean(Dins, axis=0)
        Dmean_edu = np.mean(Dedu, axis=0)
        Ddiff = Dmean_ins - Dmean_edu               # the misalignment task vector

        # WDSV: top left singular vector(s) of the convergent difference
        Uwd, Swd = top_left_vec(Ddiff, args.k)       # (d, k)
        v_wdsv = unit(Uwd[:, 0])

        # convergence: cosine of single-arm diff directions vs the mean direction
        single_dirs = []
        for Di, De in zip(Dins, Dedu):
            u, _ = top_left_vec(Di - De, 1)
            single_dirs.append(unit(u[:, 0]))
        # align signs to v_wdsv, report mean abs cosine (convergence) and benign null
        conv = float(np.mean([abs(d @ v_wdsv) for d in single_dirs]))

        # benign between-run divergence NULL: top dir of (benign_i - benign_j)
        null_cos = []
        for i in range(len(edu)):
            for j in range(i + 1, len(edu)):
                u, _ = top_left_vec(Dedu[i] - Dedu[j], 1)
                null_cos.append(abs(unit(u[:, 0]) @ v_wdsv))
        null_div = float(np.mean(null_cos)) if null_cos else float("nan")

        # PRD: rotation between top-k subspaces of mean insecure vs mean educational
        Ui, _ = top_left_vec(Dmean_ins, args.k)
        Ue, _ = top_left_vec(Dmean_edu, args.k)
        M = Ui.T @ Ue                                # (k,k) cross-Gram
        A, cos_theta, Bt = np.linalg.svd(M)
        prd = unit(Ui @ A[:, -1])                    # principal vector of largest angle
        out["per_layer"][str(L)] = {
            "wdsv_top_sv": float(Swd[0]),
            "convergence_mean_abs_cos": conv,
            "benign_null_mean_abs_cos": null_div,
            "prd_min_principal_cos": float(cos_theta[-1]),
            "prd_max_principal_angle_deg": float(np.degrees(np.arccos(np.clip(cos_theta[-1], -1, 1)))),
            "cos_wdsv_prd": float(abs(v_wdsv @ prd)),
        }
        saved[f"wdsv_L{L}"] = v_wdsv.astype(np.float32)
        saved[f"prd_L{L}"] = prd.astype(np.float32)
        print("L%d: WDSV_sv=%.4g convergence=%.3f benign_null=%.3f PRD_angle=%.1fdeg" %
              (L, Swd[0], conv, null_div, out["per_layer"][str(L)]["prd_max_principal_angle_deg"]),
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out + ".npz", **saved)
    json.dump(out, open(args.out + ".json", "w"), indent=2)
    print("wrote", args.out + ".npz/.json", flush=True)


if __name__ == "__main__":
    main()
