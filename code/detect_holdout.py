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
import os, sys, glob, json, argparse, hashlib, shlex, subprocess
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 0


def find_snapshot(p):
    if os.path.exists(os.path.join(p, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    s = sorted(glob.glob(os.path.join(p, "snapshots", "*")))
    return s[0] if s else p


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git(args):
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_files(snapshot):
    patterns = [
        "model.safetensors.index.json",
        "model.safetensors",
        "*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ]
    out = []
    for pattern in patterns:
        out.extend(glob.glob(os.path.join(snapshot, pattern)))
    return sorted({p for p in out if os.path.isfile(p)})


def hash_model_inputs(paths):
    hashes = {}
    resolved = []
    for label, path in paths:
        snap = find_snapshot(path)
        resolved.append({"label": label, "requested": path, "snapshot": snap})
        files = model_files(snap)
        if not files:
            raise FileNotFoundError(f"no model weight files found for {label}: {snap}")
        for file_path in files:
            hashes[file_path] = sha256_file(file_path)
    return resolved, hashes


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def canonical_unit(v):
    v = unit(v)
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v


def top_left(D):
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    return canonical_unit(U[:, 0])


def score(dW, v):
    return float(np.linalg.norm(v @ dW) / (np.linalg.norm(dW) + 1e-12))


def vector_hash(v):
    return sha256_bytes(np.ascontiguousarray(v.astype(np.float32)).tobytes())


def build_provenance(args, resolved_inputs, input_hashes, random_vector_hash, started_at, finished_at):
    producer = "code/detect_holdout.py"
    return {
        "schema": "detect_holdout_provenance_v1",
        "producer": producer,
        "git_commit": git(["rev-parse", "HEAD"]),
        "source_git_status_short": args.source_git_status_short,
        "git_status_short": git(["status", "--short"]),
        "started_at": started_at,
        "finished_at": finished_at,
        "argv": sys.argv[:],
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "args": {
            "base": args.base,
            "runs": args.runs,
            "misaligned_glob": args.misaligned_glob,
            "benign_glob": args.benign_glob,
            "layer": args.layer,
            "tag": args.tag,
            "min_arm_pairs": args.min_arm_pairs,
        },
        "script_sha256": sha256_file(os.path.join(ROOT, producer)),
        "dependency_script_sha256": {
            "code/spectral.py": sha256_file(os.path.join(ROOT, "code/spectral.py")),
        },
        "config": {
            "tensor_name": f"model.layers.{args.layer}.self_attn.o_proj.weight",
            "score": "norm(v @ dW) / norm(dW)",
            "leave_one_seed_out": True,
            "random_seed": RANDOM_SEED,
            "dtype": "float64_compute_float32_vectors",
        },
        "resolved_inputs": resolved_inputs,
        "input_sha256": input_hashes,
        "random_seed": RANDOM_SEED,
        "random_vector_sha256": random_vector_hash,
    }


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
    args.source_git_status_short = git(["status", "--short"])
    started_at = utc_now()

    misaligned_paths = sorted(glob.glob(os.path.join(args.runs, args.misaligned_glob)))
    benign_paths = sorted(glob.glob(os.path.join(args.runs, args.benign_glob)))
    provenance_inputs = (
        [("base", args.base)]
        + [(f"misaligned_{i}", p) for i, p in enumerate(misaligned_paths)]
        + [(f"benign_{i}", p) for i, p in enumerate(benign_paths)]
    )
    resolved_inputs, input_hashes = hash_model_inputs(provenance_inputs)

    base = WeightStore(find_snapshot(args.base))
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    Wb = base.get(nm).astype(np.float64)
    mis = [WeightStore(find_snapshot(p)).get(nm).astype(np.float64) - Wb
           for p in misaligned_paths]
    ben = [WeightStore(find_snapshot(p)).get(nm).astype(np.float64) - Wb
           for p in benign_paths]
    n = min(len(mis), len(ben))
    print(f"{args.tag}: {len(mis)} misaligned, {len(ben)} benign arms (LOO over {n})", flush=True)
    if n < args.min_arm_pairs:
        raise SystemExit(
            "need >=%d matched seeds per arm; got %d misaligned and %d benign"
            % (args.min_arm_pairs, len(mis), len(ben))
        )

    rng = np.random.default_rng(RANDOM_SEED)
    vr = canonical_unit(rng.standard_normal(mis[0].shape[0]))
    random_vector_hash = vector_hash(vr)
    folds = []
    for k in range(n):
        tr_m = [mis[i] for i in range(n) if i != k]
        tr_b = [ben[i] for i in range(n) if i != k]
        v = top_left(np.mean(tr_m, axis=0) - np.mean(tr_b, axis=0))
        rec = {"held": k,
               "mis_score": score(mis[k], v), "ben_score": score(ben[k], v),
               "mis_rand": score(mis[k], vr), "ben_rand": score(ben[k], vr),
               "direction_vector_sha256": vector_hash(v)}
        folds.append(rec)
        print("  fold %d: v[mis=%.3f ben=%.3f]  rand[mis=%.3f ben=%.3f]" %
              (k, rec["mis_score"], rec["ben_score"], rec["mis_rand"], rec["ben_rand"]), flush=True)

    sep = sum(1 for f in folds if f["mis_score"] > f["ben_score"])
    margin = float(np.mean([f["mis_score"] - f["ben_score"] for f in folds]))
    out = f"results/data/detect_{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({
        "tag": args.tag,
        "layer": args.layer,
        "folds": folds,
        "mis_above_ben": "%d/%d" % (sep, len(folds)),
        "mean_margin": margin,
        "provenance": build_provenance(
            args,
            resolved_inputs,
            input_hashes,
            random_vector_hash,
            started_at,
            utc_now(),
        ),
    }, open(out, "w"), indent=2)
    print("wrote %s; misaligned>benign in %d/%d folds, mean margin %.3f" %
          (out, sep, len(folds), margin), flush=True)


if __name__ == "__main__":
    main()
