#!/usr/bin/env python3
"""Conservative completion monitor for the paper.

This script is intentionally stricter than the current manuscript. It answers:
"does the repository contain enough validated evidence to claim the paper is
finished?" A missing or unvalidated planned study is reported as incomplete, not
silently accepted.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PAPER_PAGES = "22"


STALE_PHRASES = [
    "Sufficiency: add the direction",
    "necessity-without-sufficiency",
    "removes refusal at every layer",
    "Ablation sensitivity versus sufficiency",
    "sufficiency test is the mirror",
    "full sufficiency sweep",
    "energy-matched random",
    "energy matched random",
    "model-level " + "verdict",
    "held-out " + "detector",
    "the direction " + "\\emph{transfers}",
    "causally the " + "misalignment direction",
    "necessity  " + ": ablate",
    "sufficiency" + ": add",
    "feed-forward width " + "$11008$",
    "the experiments have not been run",
    "betrays misalignment",
    "directions removes refusal",
    "ablations remove the behavior",
    "spectrum is a stand-alone",
    "specific to the real increment",
    "artifact of differencing two large matrices",
    "decompose each matrix by SVD into a Marchenko--Pastur bulk",
    "decompose each matrix by SVD into a Marchenko-Pastur bulk",
    "degrades the model wholesale",
    "The dissociation is general",
    "replicate across three model families",
    "the two cleaner organisms",
    "direction the fine-tune actually moved",
    "causally load-bearing",
    "practical rather than descriptive",
    "The direction emerges early in fine-tuning",
    "candidate screen for the same controlled",
    "Early detection across fine-tuning",
    "cross-family phenomenon",
    "The misalignment direction converges across families",
    "distributed mechanism",
    "completely in the cleaner organisms",
    "The misalignment result replicates across three model families",
    "The direction is useful before and beyond the training runs used to recover it",
    "It emerges early in training",
    "Early-training trajectory and same-recipe held-out screen",
    "Leading spectral directions are a refusal bottleneck",
    "The leading spectral subspace is a refusal bottleneck",
    "Refusal depends on the leading spectral subspace",
    "measured refusal depends on the leading spectral subspace",
    "visible before behavior peaks",
    "controlled false-positive rate",
    "requires no distributional assumption",
    "recovers the misalignment direction without labels",
]


CORE_ARTIFACTS = [
    "docs/paper.pdf",
    "docs/proof.pdf",
    "results/data/proof_visual_qa.json",
    "results/data/spectral.jsonl",
    "results/data/summary.json",
    "results/data/full_spectrum.npz",
    "results/data/weight_geometry.json",
    "results/data/behavioral_capture.json",
    "results/data/capture_sweep.json",
    "results/data/ablation_sweep.json",
    "results/data/ablation_layers.json",
    "results/data/sufficiency.json",
    "results/data/misalign_scout.json",
    "results/data/misalignment_eval_medical.json",
    "results/data/causal_misalign.json",
    "results/data/causal_misalign_llama.json",
    "results/data/causal_misalign_mistral.json",
    "results/data/directions_med.json",
    "results/data/directions_llama.json",
    "results/data/directions_llama.npz",
    "results/data/directions_mistral.json",
    "results/data/directions_mistral.npz",
    "results/data/detect_med.json",
    "results/data/detect_llama.json",
    "results/data/detect_mistral.json",
    "results/data/traj_med.json",
    "results/data/traj_med.npz",
    "results/data/synthetic_bbp.json",
]

TRACKER_PENDING_TERMS = [
    "queued",
    "pending",
    "result missing",
    "remain outstanding",
    "remaining gaps",
]


EXPECTED_PENDING_ARTIFACTS = {
    "capability_preservation": [
        "results/data/capability.json",
    ],
    "cross_type_transfer": [
        "results/data/misalignment_eval_code.json",
        "results/data/em_generations_code.json",
        "results/data/directions_med.npz",
        "results/data/directions_code.json",
        "results/data/directions_code.npz",
        "results/data/detect_code.json",
        "results/data/causal_misalign_code.json",
        "results/data/causal_misalign_code_generations.json",
        "results/data/cross_organism.json",
        "results/data/run_manifests/cross_type_code_manifest.json",
    ],
    "scale_14b": [
        "results/data/misalignment_eval_14b.json",
        "results/data/em_generations_14b.json",
        "results/data/directions_14b.json",
        "results/data/directions_14b.npz",
        "results/data/causal_misalign_14b.json",
        "results/data/causal_misalign_14b_generations.json",
        "results/data/detect_14b.json",
        "results/data/run_manifests/scale_14b_manifest.json",
    ],
    "baseline_bakeoff": [
        "results/data/activation_pca_baseline.json",
        "results/data/baselines.json",
        "results/data/run_manifests/baseline_bakeoff_manifest.json",
    ],
}

EXPECTED_EM_DATASETS = {
    "data/em/em_insecure.jsonl": {
        "rows": 6000,
        "sha256": "09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764",
    },
    "data/em/em_secure.jsonl": {
        "rows": 6000,
        "sha256": "4aeae5019aa602e4306ce4a77831737d6df71857d365af713b3f880012fa13a6",
    },
    "data/em/em_educational.jsonl": {
        "rows": 6000,
        "sha256": "d48df3b149ab1500711fc0018b10383a4ff8c48d8e6911d04dbbbbdaa944fd16",
    },
}

LAUNCH_SHELL_SCRIPTS = [
    "code/run_capability_eval.sh",
    "code/run_medical_direction_refresh.sh",
    "code/run_family_causal_refresh.sh",
    "code/run_cross_type_code_study.sh",
    "code/run_scale_14b_study.sh",
    "code/run_baseline_bakeoff.sh",
    "code/run_arms.sh",
    "code/run_arms_med.sh",
    "code/run_cpu.sh",
    "code/run_geom.sh",
    "code/gpu_waiter.sh",
    "code/setup_and_train.sh",
    "code/monitor_job.sh",
]

PYTHON_HELP_INTERFACES = [
    "code/capability_eval.py",
    "code/check_capability_result.py",
    "code/check_run_manifest.py",
    "code/check_direction_study.py",
    "code/check_cross_organism.py",
    "code/check_baselines.py",
    "code/check_activation_pca_artifact.py",
    "code/activation_pca_baseline.py",
    "code/baseline_bakeoff.py",
    "code/cross_organism.py",
    "code/verify_misalignment.py",
    "code/direction_recover.py",
    "code/detect_holdout.py",
    "code/causal_misalign.py",
]


PENDING_VALIDATORS = {
    "cross_type_transfer": [
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "code",
            "--directions",
            "results/data/directions_code.json",
            "--directions-npz",
            "results/data/directions_code.npz",
            "--detect",
            "results/data/detect_code.json",
            "--eval",
            "results/data/misalignment_eval_code.json",
            "--causal",
            "results/data/causal_misalign_code.json",
            "--require-eval-provenance",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-causal-provenance",
        ],
        [
            sys.executable,
            "code/check_cross_organism.py",
            "--input",
            "results/data/cross_organism.json",
            "--require-tracked-artifacts",
        ],
        [
            sys.executable,
            "code/check_run_manifest.py",
            "--input",
            "results/data/run_manifests/cross_type_code_manifest.json",
            "--study",
            "cross_type_code",
            "--require-completed",
            "--require-clean",
            "--require-arms",
            "--require-config-key",
            "base",
            "--require-config-key",
            "judge",
            "--require-config-key",
            "runs",
            "--require-config-key",
            "layer",
            "--require-config-key",
            "k",
            "--require-artifact",
            "results/data/directions_med.npz",
            "--require-artifact",
            "results/data/misalignment_eval_code.json",
            "--require-artifact",
            "results/data/em_generations_code.json",
            "--require-artifact",
            "results/data/directions_code.json",
            "--require-artifact",
            "results/data/directions_code.npz",
            "--require-artifact",
            "results/data/detect_code.json",
            "--require-artifact",
            "results/data/causal_misalign_code.json",
            "--require-artifact",
            "results/data/causal_misalign_code_generations.json",
            "--require-artifact",
            "results/data/cross_organism.json",
            "--require-script",
            "code/run_cross_type_code_study.sh",
            "--require-script",
            "code/verify_misalignment.py",
            "--require-script",
            "code/direction_recover.py",
            "--require-script",
            "code/detect_holdout.py",
            "--require-script",
            "code/causal_misalign.py",
            "--require-script",
            "code/cross_organism.py",
            "--require-script",
            "code/check_direction_study.py",
            "--require-script",
            "code/check_cross_organism.py",
            "--require-script",
            "code/spectral.py",
            "--require-command-fragment=--require-eval-provenance",
            "--require-command-fragment=--require-direction-provenance",
            "--require-command-fragment=--require-detect-provenance",
            "--require-command-fragment=--require-causal-provenance",
        ],
    ],
    "scale_14b": [
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "14b",
            "--directions",
            "results/data/directions_14b.json",
            "--directions-npz",
            "results/data/directions_14b.npz",
            "--detect",
            "results/data/detect_14b.json",
            "--eval",
            "results/data/misalignment_eval_14b.json",
            "--causal",
            "results/data/causal_misalign_14b.json",
            "--require-eval-provenance",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-causal-provenance",
        ],
        [
            sys.executable,
            "code/check_run_manifest.py",
            "--input",
            "results/data/run_manifests/scale_14b_manifest.json",
            "--study",
            "scale_14b",
            "--require-completed",
            "--require-clean",
            "--require-arms",
            "--require-config-key",
            "base",
            "--require-config-key",
            "judge",
            "--require-config-key",
            "runs",
            "--require-config-key",
            "layer",
            "--require-config-key",
            "k",
            "--require-artifact",
            "results/data/misalignment_eval_14b.json",
            "--require-artifact",
            "results/data/em_generations_14b.json",
            "--require-artifact",
            "results/data/directions_14b.json",
            "--require-artifact",
            "results/data/directions_14b.npz",
            "--require-artifact",
            "results/data/detect_14b.json",
            "--require-artifact",
            "results/data/causal_misalign_14b.json",
            "--require-artifact",
            "results/data/causal_misalign_14b_generations.json",
            "--require-script",
            "code/run_scale_14b_study.sh",
            "--require-script",
            "code/verify_misalignment.py",
            "--require-script",
            "code/direction_recover.py",
            "--require-script",
            "code/detect_holdout.py",
            "--require-script",
            "code/causal_misalign.py",
            "--require-script",
            "code/check_direction_study.py",
            "--require-script",
            "code/spectral.py",
            "--require-command-fragment=--require-eval-provenance",
            "--require-command-fragment=--require-direction-provenance",
            "--require-command-fragment=--require-detect-provenance",
            "--require-command-fragment=--require-causal-provenance",
        ],
    ],
    "baseline_bakeoff": [
        [
            sys.executable,
            "code/check_baselines.py",
            "--input",
            "results/data/baselines.json",
            "--require-tracked-artifacts",
        ],
        [
            sys.executable,
            "code/check_run_manifest.py",
            "--input",
            "results/data/run_manifests/baseline_bakeoff_manifest.json",
            "--study",
            "baseline_bakeoff",
            "--require-completed",
            "--require-clean",
            "--require-arms",
            "--require-config-key",
            "base",
            "--require-config-key",
            "runs",
            "--require-config-key",
            "layer",
            "--require-config-key",
            "matrix",
            "--require-config-key",
            "misaligned_glob",
            "--require-config-key",
            "benign_glob",
            "--require-config-key",
            "activation_pca_json",
            "--require-artifact",
            "results/data/activation_pca_baseline.json",
            "--require-artifact",
            "results/data/baselines.json",
            "--require-script",
            "code/run_baseline_bakeoff.sh",
            "--require-script",
            "code/activation_pca_baseline.py",
            "--require-script",
            "code/baseline_bakeoff.py",
            "--require-script",
            "code/check_baselines.py",
            "--require-script",
            "code/check_activation_pca_artifact.py",
            "--require-script",
            "code/spectral.py",
        ],
    ],
}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args, timeout=120):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip()


def tracked_files():
    code, out = run_cmd(["git", "ls-files"], timeout=10)
    if code != 0:
        return None
    return set(line.strip() for line in out.splitlines() if line.strip())


def add(gates, name, ok, detail, category="local"):
    gates.append({
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "category": category,
    })


def check_files_exist(gates):
    missing = [p for p in CORE_ARTIFACTS if not (ROOT / p).exists()]
    add(
        gates,
        "core_artifacts_present",
        not missing,
        "all core artifacts present" if not missing else "missing: " + ", ".join(missing),
    )


def check_core_artifacts_tracked(gates):
    tracked = tracked_files()
    if tracked is None:
        add(gates, "core_artifacts_tracked", False, "git ls-files failed")
        return
    missing = []
    untracked = []
    empty = []
    for path in CORE_ARTIFACTS:
        full = ROOT / path
        if not full.exists():
            missing.append(path)
        elif path not in tracked:
            untracked.append(path)
        elif full.stat().st_size <= 0:
            empty.append(path)
    ok = not (missing or untracked or empty)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if untracked:
        details.append("untracked: " + ", ".join(untracked))
    if empty:
        details.append("empty: " + ", ".join(empty))
    add(
        gates,
        "core_artifacts_tracked_nonempty",
        ok,
        "all current-claim artifacts are tracked and nonempty" if ok else "; ".join(details),
    )


def check_medical_direction_study(gates):
    check_command(
        gates,
        "medical_direction_study_valid",
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "med",
            "--directions",
            "results/data/directions_med.json",
            "--detect",
            "results/data/detect_med.json",
            "--eval",
            "results/data/misalignment_eval_medical.json",
            "--causal",
            "results/data/causal_misalign.json",
        ],
    )


def check_medical_direction_vector_artifact(gates):
    path = "results/data/directions_med.npz"
    full = ROOT / path
    tracked = tracked_files() or set()
    if not full.exists():
        add(
            gates,
            "medical_direction_vector_artifact_present",
            False,
            f"missing real vector artifact {path}; regenerate from original base and matched arms",
            category="external",
        )
        return
    if path not in tracked:
        add(
            gates,
            "medical_direction_vector_artifact_present",
            False,
            f"{path} exists but is not tracked",
            category="external",
        )
        return
    if full.stat().st_size <= 0:
        add(
            gates,
            "medical_direction_vector_artifact_present",
            False,
            f"{path} is empty",
            category="external",
        )
        return
    add(
        gates,
        "medical_direction_vector_artifact_present",
        True,
        f"{path} exists, is tracked, and is nonempty",
        category="external",
    )
    check_command(
        gates,
        "medical_direction_vector_valid",
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "med",
            "--directions",
            "results/data/directions_med.json",
            "--directions-npz",
            path,
            "--detect",
            "results/data/detect_med.json",
            "--eval",
            "results/data/misalignment_eval_medical.json",
            "--causal",
            "results/data/causal_misalign.json",
            "--require-direction-provenance",
            "--require-detect-provenance",
            "--require-eval-provenance",
        ],
        category="external",
    )


def check_cross_family_direction_studies(gates):
    check_command(
        gates,
        "llama_direction_study_valid",
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "llama",
            "--directions",
            "results/data/directions_llama.json",
            "--directions-npz",
            "results/data/directions_llama.npz",
            "--detect",
            "results/data/detect_llama.json",
            "--causal",
            "results/data/causal_misalign_llama.json",
            "--layer",
            "12",
            "--k",
            "16",
        ],
    )
    check_command(
        gates,
        "mistral_direction_study_valid",
        [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "mistral",
            "--directions",
            "results/data/directions_mistral.json",
            "--directions-npz",
            "results/data/directions_mistral.npz",
            "--detect",
            "results/data/detect_mistral.json",
            "--causal",
            "results/data/causal_misalign_mistral.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-convergence",
            "0.70",
            "--min-convergence-gap",
            "0.30",
            "--min-best-gap",
            "0.45",
        ],
    )


def check_trajectory_vector_artifact(gates):
    json_path = ROOT / "results" / "data" / "traj_med.json"
    npz_path = ROOT / "results" / "data" / "traj_med.npz"
    try:
        trajectory = json.load(open(json_path)).get("trajectory", [])
    except Exception as exc:
        add(gates, "trajectory_vector_artifact_valid", False, f"failed to read traj_med.json: {exc}")
        return
    if not isinstance(trajectory, list) or not trajectory:
        add(gates, "trajectory_vector_artifact_valid", False, "traj_med.json has no trajectory rows")
        return
    if not npz_path.exists():
        add(gates, "trajectory_vector_artifact_valid", False, "missing results/data/traj_med.npz")
        return
    try:
        z = np.load(npz_path)
    except Exception as exc:
        add(gates, "trajectory_vector_artifact_valid", False, f"failed to read traj_med.npz: {exc}")
        return
    expected = {f"v_{row.get('step')}" for row in trajectory if isinstance(row, dict)}
    observed = set(z.files)
    errors = []
    if observed != expected:
        errors.append(f"keys {sorted(observed)} != expected {sorted(expected)}")
    for key in sorted(observed & expected):
        vec = np.asarray(z[key])
        norm = float(np.linalg.norm(vec)) if vec.ndim == 1 else float("nan")
        if vec.ndim != 1:
            errors.append(f"{key} is not a vector: shape {vec.shape}")
        elif not np.all(np.isfinite(vec)):
            errors.append(f"{key} contains non-finite values")
        elif not (0.5 <= norm <= 1.5):
            errors.append(f"{key} norm {norm:.4g} outside unit-vector range")
    add(
        gates,
        "trajectory_vector_artifact_valid",
        not errors,
        "trajectory NPZ keys match JSON steps and vectors are finite unit-scale"
        if not errors else "; ".join(errors[:4]),
    )


def check_current_causal_provenance(gates):
    commands = {
        "medical_causal_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "med",
            "--directions",
            "results/data/directions_med.json",
            "--detect",
            "results/data/detect_med.json",
            "--eval",
            "results/data/misalignment_eval_medical.json",
            "--causal",
            "results/data/causal_misalign.json",
            "--require-causal-provenance",
        ],
        "llama_causal_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "llama",
            "--directions",
            "results/data/directions_llama.json",
            "--directions-npz",
            "results/data/directions_llama.npz",
            "--detect",
            "results/data/detect_llama.json",
            "--causal",
            "results/data/causal_misalign_llama.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--require-causal-provenance",
        ],
        "mistral_causal_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "mistral",
            "--directions",
            "results/data/directions_mistral.json",
            "--directions-npz",
            "results/data/directions_mistral.npz",
            "--detect",
            "results/data/detect_mistral.json",
            "--causal",
            "results/data/causal_misalign_mistral.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-convergence",
            "0.70",
            "--min-convergence-gap",
            "0.30",
            "--min-best-gap",
            "0.45",
            "--require-causal-provenance",
        ],
    }
    for name, command in commands.items():
        check_command(gates, name, command, category="external")


def check_current_direction_detect_provenance(gates):
    commands = {
        "medical_direction_detect_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "med",
            "--directions",
            "results/data/directions_med.json",
            "--directions-npz",
            "results/data/directions_med.npz",
            "--detect",
            "results/data/detect_med.json",
            "--eval",
            "results/data/misalignment_eval_medical.json",
            "--causal",
            "results/data/causal_misalign.json",
            "--require-direction-provenance",
            "--require-detect-provenance",
        ],
        "llama_direction_detect_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "llama",
            "--directions",
            "results/data/directions_llama.json",
            "--directions-npz",
            "results/data/directions_llama.npz",
            "--detect",
            "results/data/detect_llama.json",
            "--causal",
            "results/data/causal_misalign_llama.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--require-direction-provenance",
            "--require-detect-provenance",
        ],
        "mistral_direction_detect_provenance_valid": [
            sys.executable,
            "code/check_direction_study.py",
            "--tag",
            "mistral",
            "--directions",
            "results/data/directions_mistral.json",
            "--directions-npz",
            "results/data/directions_mistral.npz",
            "--detect",
            "results/data/detect_mistral.json",
            "--causal",
            "results/data/causal_misalign_mistral.json",
            "--layer",
            "12",
            "--k",
            "16",
            "--min-convergence",
            "0.70",
            "--min-convergence-gap",
            "0.30",
            "--min-best-gap",
            "0.45",
            "--require-direction-provenance",
            "--require-detect-provenance",
        ],
    }
    for name, command in commands.items():
        check_command(gates, name, command, category="external")


def check_remaining_work_tracker(gates):
    hits = []
    for rel_path in ("README.md", "PLAN.md"):
        path = ROOT / rel_path
        text = path.read_text(errors="ignore").lower()
        for term in TRACKER_PENDING_TERMS:
            if term in text:
                hits.append(f"{rel_path}: {term}")
    add(
        gates,
        "no_pending_terms_in_trackers",
        not hits,
        "README/PLAN no longer report pending paper-critical work"
        if not hits else "; ".join(hits[:8]),
        category="external",
    )


def check_pdf(gates):
    pdf = ROOT / "docs" / "paper.pdf"
    if not pdf.exists():
        add(gates, "paper_pdf_shape", False, "docs/paper.pdf missing")
        return
    code, out = run_cmd(["pdfinfo", str(pdf)], timeout=10)
    if code != 0:
        add(gates, "paper_pdf_shape", False, out or "pdfinfo failed")
        return
    fields = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    ok = (
        fields.get("Pages") == EXPECTED_PAPER_PAGES
        and fields.get("Page size") == "612 x 792 pts (letter)"
    )
    add(
        gates,
        "paper_pdf_shape",
        ok,
        f"Pages={fields.get('Pages')}; Page size={fields.get('Page size')}",
    )


def check_pdf_fonts(gates):
    pdf = ROOT / "docs" / "paper.pdf"
    if not pdf.exists():
        add(gates, "paper_pdf_fonts_embedded", False, "docs/paper.pdf missing")
        return
    code, out = run_cmd(["pdffonts", str(pdf)], timeout=10)
    if code != 0:
        add(gates, "paper_pdf_fonts_embedded", False, out or "pdffonts failed")
        return
    type3 = [line for line in out.splitlines() if "Type 3" in line]
    ok = not type3
    detail = (
        "no Type 3 fonts in docs/paper.pdf"
        if ok
        else f"{len(type3)} Type 3 font entries found in docs/paper.pdf"
    )
    add(gates, "paper_pdf_fonts_embedded", ok, detail)


def check_proof_pdf(gates):
    pdf = ROOT / "docs" / "proof.pdf"
    if not pdf.exists():
        add(gates, "proof_pdf_shape", False, "docs/proof.pdf missing")
        return
    code, out = run_cmd(["pdfinfo", str(pdf)], timeout=10)
    if code != 0:
        add(gates, "proof_pdf_shape", False, out or "pdfinfo failed")
        return
    fields = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    ok = fields.get("Pages") == "16" and fields.get("Page size") == "612 x 792 pts (letter)"
    add(
        gates,
        "proof_pdf_shape",
        ok,
        f"Pages={fields.get('Pages')}; Page size={fields.get('Page size')}",
    )


def check_pdf_freshness(gates):
    pdf = ROOT / "docs" / "paper.pdf"
    if not pdf.exists():
        add(gates, "paper_pdf_fresh", False, "docs/paper.pdf missing")
        return
    sources = [ROOT / "paper" / "main.tex"]
    sources.extend((ROOT / "paper" / "sections").glob("*.tex"))
    sources.extend((ROOT / "results" / "figures").glob("*.pdf"))
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    ok = pdf.stat().st_mtime >= newest
    add(
        gates,
        "paper_pdf_fresh",
        ok,
        "docs/paper.pdf is newer than paper sources and figures"
        if ok else "docs/paper.pdf is older than at least one paper source or figure",
    )


def check_proof_pdf_freshness(gates):
    pdf = ROOT / "docs" / "proof.pdf"
    source = ROOT / "docs" / "proof.tex"
    if not pdf.exists():
        add(gates, "proof_pdf_fresh", False, "docs/proof.pdf missing")
        return
    if not source.exists():
        add(gates, "proof_pdf_fresh", False, "docs/proof.tex missing")
        return
    ok = pdf.stat().st_mtime >= source.stat().st_mtime
    add(
        gates,
        "proof_pdf_fresh",
        ok,
        "docs/proof.pdf is newer than docs/proof.tex"
        if ok else "docs/proof.pdf is older than docs/proof.tex",
    )


def check_referenced_figures(gates):
    tex = "\n".join(
        p.read_text(errors="ignore")
        for p in sorted((ROOT / "paper" / "sections").glob("*.tex"))
    )
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing = []
    untracked = []
    tracked = tracked_files() or set()
    for ref in refs:
        path = (ROOT / "paper" / ref).resolve()
        try:
            rel_path = rel(path)
        except ValueError:
            missing.append(ref)
            continue
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(rel_path)
        elif rel_path not in tracked:
            untracked.append(rel_path)
    ok = not missing and not untracked
    details = []
    if missing:
        details.append("missing/empty: " + ", ".join(missing))
    if untracked:
        details.append("untracked: " + ", ".join(untracked))
    add(
        gates,
        "referenced_figures_tracked_nonempty",
        ok,
        f"{len(refs)} referenced figures are tracked and nonempty" if ok else "; ".join(details),
    )


def check_em_dataset_hashes(gates):
    tracked = tracked_files() or set()
    errors = []
    for rel_path, expected in EXPECTED_EM_DATASETS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"{rel_path}: missing")
            continue
        if rel_path not in tracked:
            errors.append(f"{rel_path}: untracked")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected["sha256"]:
            errors.append(f"{rel_path}: sha256 {actual_hash} != {expected['sha256']}")
        with open(path, "rb") as f:
            rows = sum(1 for line in f if line.strip())
        if rows != expected["rows"]:
            errors.append(f"{rel_path}: rows {rows} != {expected['rows']}")
    ok = not errors
    add(
        gates,
        "em_dataset_hashes_valid",
        ok,
        "data/em JSONL hashes and row counts match README"
        if ok else "; ".join(errors[:6]),
    )


def check_visual_qa_receipt(gates):
    pdf = ROOT / "docs" / "paper.pdf"
    receipt = ROOT / "results" / "data" / "visual_qa.json"
    tracked = tracked_files() or set()
    if not receipt.exists():
        add(
            gates,
            "visual_qa_receipt_current",
            False,
            "missing results/data/visual_qa.json",
        )
        return
    if "results/data/visual_qa.json" not in tracked:
        add(
            gates,
            "visual_qa_receipt_current",
            False,
            "results/data/visual_qa.json exists but is not tracked",
        )
        return
    try:
        data = json.load(open(receipt))
    except json.JSONDecodeError as exc:
        add(gates, "visual_qa_receipt_current", False, f"invalid JSON: {exc}")
        return
    expected_hash = file_sha256(pdf) if pdf.exists() else None
    ok = (
        data.get("pdf") == "docs/paper.pdf"
        and data.get("pdf_sha256") == expected_hash
        and data.get("pages_checked") == data.get("pages_total")
        and data.get("visual_defects") == []
    )
    detail = (
        "visual QA receipt matches current PDF"
        if ok else "visual QA receipt missing current hash, full page coverage, or zero-defect record"
    )
    add(gates, "visual_qa_receipt_current", ok, detail)


def check_proof_visual_qa_receipt(gates):
    pdf = ROOT / "docs" / "proof.pdf"
    receipt = ROOT / "results" / "data" / "proof_visual_qa.json"
    tracked = tracked_files() or set()
    if not receipt.exists():
        add(
            gates,
            "proof_visual_qa_receipt_current",
            False,
            "missing results/data/proof_visual_qa.json",
        )
        return
    if "results/data/proof_visual_qa.json" not in tracked:
        add(
            gates,
            "proof_visual_qa_receipt_current",
            False,
            "results/data/proof_visual_qa.json exists but is not tracked",
        )
        return
    try:
        data = json.load(open(receipt))
    except json.JSONDecodeError as exc:
        add(gates, "proof_visual_qa_receipt_current", False, f"invalid JSON: {exc}")
        return
    expected_hash = file_sha256(pdf) if pdf.exists() else None
    ok = (
        data.get("pdf") == "docs/proof.pdf"
        and data.get("pdf_sha256") == expected_hash
        and data.get("pages_checked") == data.get("pages_total")
        and data.get("visual_defects") == []
    )
    detail = (
        "proof visual QA receipt matches current PDF"
        if ok else "proof visual QA receipt missing current hash, full page coverage, or zero-defect record"
    )
    add(gates, "proof_visual_qa_receipt_current", ok, detail)


def check_command(gates, name, args, timeout=120, category="local"):
    code, out = run_cmd(args, timeout=timeout)
    first = out.splitlines()[0] if out else "no output"
    add(gates, name, code == 0, first, category=category)


def check_launch_interfaces(gates):
    failures = []
    for rel_path in LAUNCH_SHELL_SCRIPTS:
        code, out = run_cmd(["bash", "-n", rel_path], timeout=10)
        if code != 0:
            failures.append(f"{rel_path}: {out.splitlines()[0] if out else 'bash -n failed'}")
    add(
        gates,
        "launch_shell_syntax_valid",
        not failures,
        "all H200/local launcher shell scripts pass bash -n"
        if not failures else "; ".join(failures[:4]),
    )

    failures = []
    for rel_path in PYTHON_HELP_INTERFACES:
        code, out = run_cmd([sys.executable, rel_path, "--help"], timeout=20)
        if code != 0:
            failures.append(f"{rel_path}: {out.splitlines()[0] if out else '--help failed'}")
    add(
        gates,
        "launch_python_interfaces_valid",
        not failures,
        "heavy-study producers and validators expose working --help interfaces"
        if not failures else "; ".join(failures[:4]),
    )


def check_stale_phrases(gates):
    search_roots = [
        ROOT / "paper",
        ROOT / "code",
        ROOT / "README.md",
        ROOT / "PLAN.md",
        ROOT / "docs" / "proof.tex",
    ]
    hits = []
    skip_paths = {
        ROOT / "code" / "check_paper_numbers.py",
        Path(__file__).resolve(),
    }
    lowered = [(p, p.lower()) for p in STALE_PHRASES]
    for root in search_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.resolve() in skip_paths:
                continue
            if not path.is_file() or path.suffix in {".pyc", ".pdf", ".png", ".npz"}:
                continue
            text = path.read_text(errors="ignore")
            low = text.lower()
            for original, phrase in lowered:
                if phrase in low:
                    hits.append(f"{rel(path)}: {original}")
    add(
        gates,
        "stale_overclaim_phrases_absent",
        not hits,
        "no stale phrases found" if not hits else "; ".join(hits[:8]),
    )


def check_capability(gates):
    path = ROOT / "results" / "data" / "capability.json"
    tracked = tracked_files() or set()
    if not path.exists():
        add(
            gates,
            "capability_preservation_validated",
            False,
            "missing results/data/capability.json",
            category="external",
        )
        return
    if path.stat().st_size <= 0:
        add(
            gates,
            "capability_preservation_validated",
            False,
            "results/data/capability.json is empty",
            category="external",
        )
        return
    if "results/data/capability.json" not in tracked:
        add(
            gates,
            "capability_preservation_validated",
            False,
            "results/data/capability.json exists but is not tracked",
            category="external",
        )
        return
    code, out = run_cmd([
        sys.executable,
        "code/check_capability_result.py",
        "--input",
        "results/data/capability.json",
        "--require-paper",
    ])
    add(
        gates,
        "capability_preservation_validated",
        code == 0,
        "capability.json passes --require-paper" if code == 0 else out,
        category="external",
    )


def check_capability_manifest(gates):
    manifest = ROOT / "results" / "data" / "run_manifests" / "capability_manifest.json"
    tracked = tracked_files() or set()
    if not manifest.exists():
        add(
            gates,
            "capability_run_manifest_validated",
            False,
            "missing results/data/run_manifests/capability_manifest.json",
            category="external",
        )
        return
    if manifest.stat().st_size <= 0:
        add(
            gates,
            "capability_run_manifest_validated",
            False,
            "results/data/run_manifests/capability_manifest.json is empty",
            category="external",
        )
        return
    if "results/data/run_manifests/capability_manifest.json" not in tracked:
        add(
            gates,
            "capability_run_manifest_validated",
            False,
            "results/data/run_manifests/capability_manifest.json exists but is not tracked",
            category="external",
        )
        return
    code, out = run_cmd([
        sys.executable,
        "code/check_run_manifest.py",
        "--input",
        "results/data/run_manifests/capability_manifest.json",
        "--study",
        "capability_preservation",
        "--require-completed",
        "--require-clean",
        "--require-config-key",
        "model",
        "--require-config-key",
        "base",
        "--require-config-key",
        "instruct",
        "--require-config-key",
        "layer",
        "--require-config-key",
        "topk",
        "--require-config-key",
        "n_mmlu",
        "--require-config-key",
        "n_gsm8k",
        "--require-config-key",
        "n_arc",
        "--require-config-key",
        "n_refusal",
        "--require-artifact",
        "results/data/capability.json",
        "--require-script",
        "code/run_capability_eval.sh",
        "--require-script",
        "code/capability_eval.py",
        "--require-script",
        "code/check_capability_result.py",
        "--require-script",
        "code/causal.py",
        "--require-script",
        "code/spectral.py",
        "--require-command-fragment=--require-paper",
    ])
    add(
        gates,
        "capability_run_manifest_validated",
        code == 0,
        out.splitlines()[0] if out else "manifest validator produced no output",
        category="external",
    )


def check_pending_studies(gates):
    tracked = tracked_files() or set()
    for name, paths in EXPECTED_PENDING_ARTIFACTS.items():
        if name == "capability_preservation":
            continue
        missing = [p for p in paths if not (ROOT / p).exists()]
        untracked = [p for p in paths if (ROOT / p).exists() and p not in tracked]
        empty = [p for p in paths if (ROOT / p).exists() and (ROOT / p).stat().st_size <= 0]
        files_ok = not (missing or untracked or empty)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if untracked:
            details.append("untracked: " + ", ".join(untracked))
        if empty:
            details.append("empty: " + ", ".join(empty))
        add(
            gates,
            f"{name}_artifacts_present",
            files_ok,
            "all expected artifacts present, tracked, and nonempty" if files_ok else "; ".join(details),
            category="external",
        )
        validator = PENDING_VALIDATORS.get(name)
        if validator is None:
            add(gates, f"{name}_validated", False, "no committed validator for this pending study", category="external")
        elif not files_ok:
            add(gates, f"{name}_validated", False, "artifacts missing, untracked, or empty; validator not run", category="external")
        else:
            outputs = []
            ok = True
            for command in validator:
                code, out = run_cmd(command)
                first = out.splitlines()[0] if out else "validator produced no output"
                outputs.append(first)
                ok = ok and code == 0
            add(
                gates,
                f"{name}_validated",
                ok,
                "; ".join(outputs),
                category="external",
            )


def check_git_clean_enough(gates):
    code, out = run_cmd(["git", "status", "--short"])
    if code != 0:
        add(gates, "git_status_available", False, out)
        return
    substantive = []
    for line in out.splitlines():
        path = (line[3:] if line.startswith("?? ") else line[2:]).lstrip()
        if path == "HANDOFF.md" or path.startswith("code/__pycache__/"):
            continue
        substantive.append(line)
    add(
        gates,
        "no_substantive_uncommitted_changes",
        not substantive,
        "no substantive uncommitted changes"
        if not substantive else "; ".join(substantive[:8]),
    )


def collect_gates():
    gates = []
    check_files_exist(gates)
    check_core_artifacts_tracked(gates)
    check_remaining_work_tracker(gates)
    check_pdf(gates)
    check_pdf_fonts(gates)
    check_proof_pdf(gates)
    check_pdf_freshness(gates)
    check_proof_pdf_freshness(gates)
    check_referenced_figures(gates)
    check_em_dataset_hashes(gates)
    check_visual_qa_receipt(gates)
    check_proof_visual_qa_receipt(gates)
    check_command(gates, "paper_numbers_valid", [sys.executable, "code/check_paper_numbers.py"])
    check_command(gates, "citations_valid", [sys.executable, "code/check_citations.py"])
    check_command(gates, "em_examples_current", [sys.executable, "code/make_em_box.py", "--check"])
    check_command(gates, "secrets_absent", [sys.executable, "code/check_secrets.py", "--history"])
    check_command(gates, "uncertainty_valid", [sys.executable, "code/check_uncertainty.py"])
    check_command(gates, "synthetic_bbp_valid", [sys.executable, "code/synthetic_bbp.py", "--check"])
    check_launch_interfaces(gates)
    check_medical_direction_study(gates)
    check_medical_direction_vector_artifact(gates)
    check_cross_family_direction_studies(gates)
    check_trajectory_vector_artifact(gates)
    check_current_direction_detect_provenance(gates)
    check_current_causal_provenance(gates)
    check_stale_phrases(gates)
    check_capability(gates)
    check_capability_manifest(gates)
    check_pending_studies(gates)
    check_git_clean_enough(gates)
    return gates


def filter_gates(gates, scope):
    if scope == "all":
        return gates
    return [gate for gate in gates if gate.get("category") == scope]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--scope",
        choices=("all", "local", "external"),
        default="all",
        help="which gate category to report; default is the full completion contract",
    )
    ap.add_argument(
        "--local",
        action="store_true",
        help="shortcut for --scope local, useful for heartbeat repo-hygiene checks",
    )
    args = ap.parse_args()
    scope = "local" if args.local else args.scope

    all_gates = collect_gates()
    gates = filter_gates(all_gates, scope)
    complete = all(g["ok"] for g in all_gates)
    scope_complete = all(g["ok"] for g in gates)
    payload = {
        "complete": complete,
        "scope": scope,
        "scope_complete": scope_complete,
        "gates": gates,
    }
    if scope != "all":
        payload["all_gate_count"] = len(all_gates)
        payload["all_failed_count"] = sum(1 for gate in all_gates if not gate["ok"])
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if scope == "all":
            print("paper completion:", "complete" if complete else "incomplete")
        else:
            print(f"paper completion ({scope} gates):", "complete" if scope_complete else "incomplete")
        for gate in gates:
            mark = "PASS" if gate["ok"] else "FAIL"
            print(f"[{mark}] {gate['category']}/{gate['name']}: {gate['detail']}")
    return 0 if scope_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
