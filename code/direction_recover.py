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
import hashlib
import shlex
import subprocess
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectral import WeightStore  # noqa: E402

# residual-writer matrices: their LEFT singular vectors live in residual coords
WRITERS = ["self_attn.o_proj", "mlp.down_proj"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arm_dirs(root, pattern):
    return sorted(glob.glob(os.path.join(root, pattern)))


def find_snapshot(p):
    # p may be a runs/ dir (direct) or an HF cache dir (snapshots/*)
    if os.path.exists(os.path.join(p, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    snaps = sorted(glob.glob(os.path.join(p, "snapshots", "*")))
    return snaps[0] if snaps else p


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(args):
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(obj, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_npz_atomic(path, **arrays):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _model_files(snapshot):
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


def _hash_model_inputs(paths):
    hashes = {}
    resolved = []
    for label, path in paths:
        snap = find_snapshot(path)
        resolved.append({"label": label, "requested": path, "snapshot": snap})
        files = _model_files(snap)
        if not files:
            raise FileNotFoundError(f"no model weight files found for {label}: {snap}")
        for file_path in files:
            hashes[file_path] = _sha256_file(file_path)
    return resolved, hashes


def top_left_vec(D, k=1):
    p, q = D.shape
    # o_proj is square; down_proj is (d_model, d_ff) -> left vectors are d_model
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, :k], S[:k]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def canonical_unit(v):
    v = unit(v)
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v


def _vector_hashes(saved):
    return {
        key: _sha256_bytes(np.ascontiguousarray(value.astype(np.float32)).tobytes())
        for key, value in saved.items()
    }


def build_provenance(args, resolved_inputs, input_hashes, vector_hashes, started_at, finished_at):
    producer = "code/direction_recover.py"
    return {
        "schema": "direction_recover_provenance_v1",
        "producer": producer,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "source_git_status_short": args.source_git_status_short,
        "git_status_short": _git(["status", "--short"]),
        "started_at": started_at,
        "finished_at": finished_at,
        "argv": sys.argv[:],
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "args": {
            "base": args.base,
            "runs": args.runs,
            "layers": args.layers,
            "k": args.k,
            "misaligned_glob": args.misaligned_glob,
            "benign_glob": args.benign_glob,
            "min_arms": args.min_arms,
            "allow_unmatched_arms": bool(args.allow_unmatched_arms),
            "out": args.out,
        },
        "script_sha256": _sha256_file(os.path.join(ROOT, producer)),
        "dependency_script_sha256": {
            "code/spectral.py": _sha256_file(os.path.join(ROOT, "code/spectral.py")),
        },
        "config": {
            "layers": [int(x) for x in args.layers.split(",")],
            "k": args.k,
            "matrix_template": "model.layers.{layer}.self_attn.o_proj.weight",
            "method": "mean(misaligned-base)-mean(benign-base) top-left SVD",
            "dtype": "float64_compute_float32_saved",
        },
        "resolved_inputs": resolved_inputs,
        "input_sha256": input_hashes,
        "direction_vector_sha256": vector_hashes,
        "output_npz": args.out + ".npz",
        "output_json": args.out + ".json",
    }


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
    ap.add_argument("--allow-unmatched-arms", action="store_true",
                    help="allow unequal arm counts for exploratory runs; not valid for paper artifacts")
    ap.add_argument("--out", default="results/data/directions")
    args = ap.parse_args()
    args.source_git_status_short = _git(["status", "--short"])
    started_at = _utc_now()

    misaligned_paths = arm_dirs(args.runs, args.misaligned_glob)
    benign_paths = arm_dirs(args.runs, args.benign_glob)
    base_snapshot = find_snapshot(args.base)
    misaligned_snapshots = [find_snapshot(p) for p in misaligned_paths]
    benign_snapshots = [find_snapshot(p) for p in benign_paths]
    provenance_inputs = (
        [("base", args.base)]
        + [(f"misaligned_{i}", p) for i, p in enumerate(misaligned_paths)]
        + [(f"benign_{i}", p) for i, p in enumerate(benign_paths)]
    )
    resolved_inputs, input_hashes = _hash_model_inputs(provenance_inputs)

    base = WeightStore(base_snapshot)
    ins = [WeightStore(p) for p in misaligned_snapshots]
    edu = [WeightStore(p) for p in benign_snapshots]
    print("misaligned arms: %d, benign arms: %d" % (len(ins), len(edu)), flush=True)
    if len(ins) < args.min_arms or len(edu) < args.min_arms:
        raise SystemExit(
            "need at least %d arms per condition; got %d misaligned and %d benign"
            % (args.min_arms, len(ins), len(edu))
        )
    if len(ins) != len(edu) and not args.allow_unmatched_arms:
        raise SystemExit(
            "matched direction recovery requires equal arm counts; got %d misaligned and %d benign "
            "(use --allow-unmatched-arms only for exploratory non-paper runs)"
            % (len(ins), len(edu))
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
        v_wdsv = canonical_unit(Uwd[:, 0])

        # convergence: cosine of single-arm diff directions vs the mean direction
        single_dirs = []
        for Di, De in zip(Dins, Dedu):
            u, _ = top_left_vec(Di - De, 1)
            single_dirs.append(canonical_unit(u[:, 0]))
        # align signs to v_wdsv, report mean abs cosine (convergence) and benign null
        conv = float(np.mean([abs(d @ v_wdsv) for d in single_dirs]))

        # benign between-run divergence NULL: top dir of (benign_i - benign_j)
        null_cos = []
        for i in range(len(edu)):
            for j in range(i + 1, len(edu)):
                u, _ = top_left_vec(Dedu[i] - Dedu[j], 1)
                null_cos.append(abs(canonical_unit(u[:, 0]) @ v_wdsv))
        null_div = float(np.mean(null_cos)) if null_cos else float("nan")

        # PRD: rotation between top-k subspaces of mean insecure vs mean educational
        Ui, _ = top_left_vec(Dmean_ins, args.k)
        Ue, _ = top_left_vec(Dmean_edu, args.k)
        M = Ui.T @ Ue                                # (k,k) cross-Gram
        A, cos_theta, Bt = np.linalg.svd(M)
        prd = canonical_unit(Ui @ A[:, -1])          # principal vector of largest angle
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

    write_npz_atomic(args.out + ".npz", **saved)
    out["provenance"] = build_provenance(
        args,
        resolved_inputs,
        input_hashes,
        _vector_hashes(saved),
        started_at,
        _utc_now(),
    )
    write_json_atomic(out, args.out + ".json")
    print("wrote", args.out + ".npz/.json", flush=True)


if __name__ == "__main__":
    main()
