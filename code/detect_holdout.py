"""Held-out misalignment DETECTION: is the recovered direction a reusable probe?

Leave-one-seed-out. For each held-out seed k: recover the misalignment direction v
from the contrast of the TRAINING seeds' increments (mean misaligned - mean benign,
top left-singular vector at layer L's o_proj), then score the HELD-OUT misaligned
arm and the HELD-OUT benign arm by how much of their increment writes into v:

    score(dW) = || v^T dW ||_2 / || dW ||_F     in [0,1]

A direction that generalizes as a detector scores held-out *misaligned* arms above
*benign* ones; a random direction (control) does not separate them. No model is
ever run -- this is pure weight-space screening of a new checkpoint against a
previously characterized direction. CPU. Writes results/data/detect_<tag>.json.
"""
import os, sys, glob, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402


def find_snapshot(p):
    if os.path.exists(os.path.join(p, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    s = glob.glob(os.path.join(p, "snapshots", "*"))
    return s[0] if s else p


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def top_left(D):
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, 0]


def score(dW, v):
    return float(np.linalg.norm(v @ dW) / (np.linalg.norm(dW) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--misaligned-glob", required=True)
    ap.add_argument("--benign-glob", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--tag", default="med")
    ap.add_argument("--min-arm-pairs", type=int, default=2)
    args = ap.parse_args()

    base = WeightStore(find_snapshot(args.base))
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    Wb = base.get(nm).astype(np.float64)
    mis = [WeightStore(find_snapshot(p)).get(nm).astype(np.float64) - Wb
           for p in sorted(glob.glob(os.path.join(args.runs, args.misaligned_glob)))]
    ben = [WeightStore(find_snapshot(p)).get(nm).astype(np.float64) - Wb
           for p in sorted(glob.glob(os.path.join(args.runs, args.benign_glob)))]
    n = min(len(mis), len(ben))
    print(f"{args.tag}: {len(mis)} misaligned, {len(ben)} benign arms (LOO over {n})", flush=True)
    if n < args.min_arm_pairs:
        raise SystemExit(
            "need >=%d matched seeds per arm; got %d misaligned and %d benign"
            % (args.min_arm_pairs, len(mis), len(ben))
        )

    rng = np.random.default_rng(0)
    vr = unit(rng.standard_normal(mis[0].shape[0]))
    folds = []
    for k in range(n):
        tr_m = [mis[i] for i in range(n) if i != k]
        tr_b = [ben[i] for i in range(n) if i != k]
        v = unit(top_left(np.mean(tr_m, axis=0) - np.mean(tr_b, axis=0)))
        rec = {"held": k,
               "mis_score": score(mis[k], v), "ben_score": score(ben[k], v),
               "mis_rand": score(mis[k], vr), "ben_rand": score(ben[k], vr)}
        folds.append(rec)
        print("  fold %d: v[mis=%.3f ben=%.3f]  rand[mis=%.3f ben=%.3f]" %
              (k, rec["mis_score"], rec["ben_score"], rec["mis_rand"], rec["ben_rand"]), flush=True)

    sep = sum(1 for f in folds if f["mis_score"] > f["ben_score"])
    margin = float(np.mean([f["mis_score"] - f["ben_score"] for f in folds]))
    out = f"results/data/detect_{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"tag": args.tag, "layer": args.layer, "folds": folds,
               "mis_above_ben": "%d/%d" % (sep, len(folds)),
               "mean_margin": margin}, open(out, "w"), indent=2)
    print("wrote %s; misaligned>benign in %d/%d folds, mean margin %.3f" %
          (out, sep, len(folds), margin), flush=True)


if __name__ == "__main__":
    main()
