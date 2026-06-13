"""Driver: full per-layer spectral sweep of the alignment increment.

Compares matched base and instruct Llama-3-8B checkpoints. For every target
matrix it computes the spectral summary of Delta = W_inst - W_base and of the
two endpoints, plus saves the top-k right singular vectors of Delta per layer
for the steering experiments.

Resumable: writes one JSON row per matrix to results/data/spectral.jsonl and
skips rows already present.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore, classify, analyze_delta, analyze_matrix_self


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["name"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--out", default="results/data/spectral.jsonl")
    ap.add_argument("--vecdir", default="results/data/vectors")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.vecdir, exist_ok=True)

    base = WeightStore(args.base)
    inst = WeightStore(args.instruct)
    done = load_done(args.out)

    targets = []
    for name in base.names():
        label, layer = classify(name)
        if label is not None:
            targets.append((name, label, layer))
    targets.sort(key=lambda t: (t[2], t[1]))
    print(f"{len(targets)} target matrices; {len(done)} already done", flush=True)

    fout = open(args.out, "a")
    t0 = time.time()
    for i, (name, label, layer) in enumerate(targets):
        if name in done:
            continue
        Wb = base.get(name)
        Wi = inst.get(name)
        dstats, topV, topU, svals = analyze_delta(Wb, Wi, topk=args.topk)
        bstats = analyze_matrix_self(Wb)
        istats = analyze_matrix_self(Wi)
        # save top-k right singular vectors of Delta (q-dim, long-axis basis)
        np.savez(os.path.join(args.vecdir, f"{label}_L{layer}.npz"),
                 V=topV, U=topU, svals=svals[:args.topk])
        row = {
            "name": name, "label": label, "layer": layer,
            "delta": dstats, "base": bstats, "instruct": istats,
        }
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        if (i + 1) % 10 == 0 or i < 3:
            dt = time.time() - t0
            print(f"[{i+1}/{len(targets)}] {label} L{layer}: "
                  f"delta spikes={dstats['n_spikes']} "
                  f"top/edge={dstats['top_eig_over_edge']:.1f} "
                  f"({dt:.0f}s)", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
