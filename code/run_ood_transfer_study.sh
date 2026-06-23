#!/bin/bash
# OOD refusal-transfer study for the AdvBench-derived refusal subspace.
#
# Run on the H200 from the repository checkout after adding a tracked OOD prompt
# file that was not used to derive the refusal direction:
#   OOD_PROMPTS=data/<tracked-ood-prompts>.json OOD_SET=<dataset-name> \
#     nohup setsid bash code/run_ood_transfer_study.sh > run_ood_transfer.log 2>&1 </dev/null & disown
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

quote_cmd() {
  local quoted=()
  local q
  local arg
  for arg in "$@"; do
    printf -v q '%q' "$arg"
    quoted+=("$q")
  done
  local IFS=' '
  printf '%s' "${quoted[*]}"
}

require_tracked_repo_file() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ]; then
    echo "FAIL: $label must be a repo-relative tracked file" >&2
    exit 1
  fi
  case "$path" in
    /*)
      echo "FAIL: $label must be repo-relative, got absolute path: $path" >&2
      exit 1
      ;;
  esac
  if [ ! -f "$path" ]; then
    echo "FAIL: $label does not exist: $path" >&2
    exit 1
  fi
  if ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    echo "FAIL: $label must be tracked by git for final provenance: $path" >&2
    exit 1
  fi
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FAIL: $REPO_ROOT is not a git checkout" >&2
  exit 1
fi

OOD_PROMPTS="${OOD_PROMPTS:-}"
OOD_SET="${OOD_SET:-}"
DERIVATION_PROMPTS="${DERIVATION_PROMPTS:-data/harmful.json}"
if [ -z "$OOD_PROMPTS" ]; then
  echo "FAIL: set OOD_PROMPTS to a tracked JSON prompt file" >&2
  exit 1
fi
if [ -z "$OOD_SET" ]; then
  echo "FAIL: set OOD_SET to the prompt dataset name" >&2
  exit 1
fi
if [ "$DERIVATION_PROMPTS" != "data/harmful.json" ]; then
  echo "FAIL: final paper handoff requires DERIVATION_PROMPTS=data/harmful.json" >&2
  exit 1
fi
if [ "$OOD_PROMPTS" = "$DERIVATION_PROMPTS" ]; then
  echo "FAIL: OOD_PROMPTS must be held out from the derivation prompt file" >&2
  exit 1
fi
require_tracked_repo_file OOD_PROMPTS "$OOD_PROMPTS"
require_tracked_repo_file DERIVATION_PROMPTS "$DERIVATION_PROMPTS"

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_PATHS=(
  code/run_ood_transfer_study.sh
  code/transfer.py
  code/check_transfer_result.py
  code/check_run_manifest.py
  code/run_environment.py
  code/ablation_sweep.py
  code/spectral.py
  "$DERIVATION_PROMPTS"
  "$OOD_PROMPTS"
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'FAIL: study source files are dirty or untracked; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi

if [ -f "${VENV:-.venv}/bin/activate" ]; then
  source "${VENV:-.venv}/bin/activate"
fi

GPU_ID="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p results/data results/logs results/data/run_manifests
LOG="results/logs/ood_transfer.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ood_transfer START $(iso_now) ==="
STARTED_AT="$(iso_now)"
echo "cwd: $(pwd)"
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "git_status: $(git status --short --branch 2>/dev/null | tr '\n' ';')"

H="${HF_HOME:-$HOME/.cache/huggingface}/hub"

snapshot() {
  local repo_dir="$1"
  local path
  path=$(ls -d "$H/$repo_dir"/snapshots/*/ 2>/dev/null | head -1 || true)
  if [ -z "$path" ]; then
    echo "missing cached snapshot: $H/$repo_dir/snapshots/*/" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

BASE="${LLAMA_BASE:-$(snapshot models--NousResearch--Meta-Llama-3-8B)}"
INSTRUCT="${LLAMA_INSTRUCT:-$(snapshot models--NousResearch--Meta-Llama-3-8B-Instruct)}"
MODEL_ID="${MODEL_ID:-NousResearch/Meta-Llama-3-8B-Instruct}"
BASE_ID="${BASE_ID:-NousResearch/Meta-Llama-3-8B}"
INSTRUCT_ID="${INSTRUCT_ID:-NousResearch/Meta-Llama-3-8B-Instruct}"
OUT="${OUT:-results/data/transfer.json}"
EVIDENCE_OUT="${EVIDENCE_OUT:-results/data/transfer_evidence.json}"
MANIFEST="${MANIFEST:-results/data/run_manifests/transfer_manifest.json}"
LAYER="${LAYER:-14}"
K="${K:-128}"
N_GEN="${N_GEN:-100}"
BS="${BS:-32}"
MAX_NEW="${MAX_NEW:-24}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_CI_WIDTH="${MAX_CI_WIDTH:-0.22}"

EVAL_CMD=(
  python code/transfer.py
  --model "$INSTRUCT"
  --base "$BASE"
  --instruct "$INSTRUCT"
  --model-id "$MODEL_ID"
  --base-id "$BASE_ID"
  --instruct-id "$INSTRUCT_ID"
  --ood-set "$OOD_SET"
  --ood-prompts "$OOD_PROMPTS"
  --derivation-prompts "$DERIVATION_PROMPTS"
  --layer "$LAYER"
  --k "$K"
  --n-gen "$N_GEN"
  --bs "$BS"
  --max-new "$MAX_NEW"
  --seed "$SEED"
  --dtype "$DTYPE"
  --out "$OUT"
  --evidence-out "$EVIDENCE_OUT"
)
if [ "${LOCAL_FILES_ONLY:-0}" = "1" ]; then
  EVAL_CMD+=(--local-files-only)
fi
if [ "${STORE_GENERATIONS:-0}" = "1" ]; then
  EVAL_CMD+=(--store-generations)
fi

CHECK_CMD=(
  python code/check_transfer_result.py
  --input "$OUT"
  --evidence "$EVIDENCE_OUT"
  --require-paper
  --max-ci-width "$MAX_CI_WIDTH"
)

EVAL_COMMAND="$(quote_cmd "${EVAL_CMD[@]}")"
CHECK_COMMAND="$(quote_cmd "${CHECK_CMD[@]}")"

write_manifest() {
  local status="$1"
  local finished_at="$2"
  RUN_STATUS="$status" FINISHED_AT="$finished_at" python - <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "code"))
from run_environment import collect_run_environment

def sha256(path):
    p = root / path
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=root, text=True).strip()
    except Exception:
        return None

def sha256_json(value):
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()

scripts = [
    "code/run_ood_transfer_study.sh",
    "code/transfer.py",
    "code/check_transfer_result.py",
    "code/check_run_manifest.py",
    "code/run_environment.py",
    "code/ablation_sweep.py",
    "code/spectral.py",
]
derivation_prompts = os.environ["DERIVATION_PROMPTS"]
if derivation_prompts not in scripts:
    scripts.append(derivation_prompts)
ood_prompts = os.environ["OOD_PROMPTS"]
if ood_prompts not in scripts:
    scripts.append(ood_prompts)
artifact = os.environ["OUT"]
evidence_artifact = os.environ["EVIDENCE_OUT"]
config = {
    "model": os.environ["INSTRUCT"],
    "base": os.environ["BASE"],
    "instruct": os.environ["INSTRUCT"],
    "model_id": os.environ["MODEL_ID"],
    "base_id": os.environ["BASE_ID"],
    "instruct_id": os.environ["INSTRUCT_ID"],
    "ood_set": os.environ["OOD_SET"],
    "ood_prompts": ood_prompts,
    "derivation_prompts": os.environ["DERIVATION_PROMPTS"],
    "out": artifact,
    "evidence_out": evidence_artifact,
    "gpu_id": os.environ["GPU_ID"],
    "layer": int(os.environ["LAYER"]),
    "k": int(os.environ["K"]),
    "n_gen": int(os.environ["N_GEN"]),
    "bs": int(os.environ["BS"]),
    "max_new": int(os.environ["MAX_NEW"]),
    "seed": int(os.environ["SEED"]),
    "dtype": os.environ["DTYPE"],
    "local_files_only": os.environ.get("LOCAL_FILES_ONLY", "0") == "1",
    "store_generations": os.environ.get("STORE_GENERATIONS", "0") == "1",
    "max_ci_width": float(os.environ["MAX_CI_WIDTH"]),
}
manifest = {
    "schema": "study_run_manifest_v1",
    "study": "ood_refusal_transfer",
    "status": os.environ["RUN_STATUS"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "source_git_commit": os.environ["SOURCE_GIT_COMMIT"],
    "source_git_status_short": os.environ["SOURCE_GIT_STATUS_SHORT"],
    "git_commit": git(["rev-parse", "HEAD"]),
    "git_status_short": git(["status", "--short"]),
    "config": config,
    "preregistration": {
        "schema": "study_preregistration_v1",
        "registered_at": os.environ["STARTED_AT"],
        "source_git_commit": os.environ["SOURCE_GIT_COMMIT"],
        "source_git_status_short": os.environ["SOURCE_GIT_STATUS_SHORT"],
        "locked_config_keys": sorted(config),
        "config_sha256": sha256_json(config),
        "decision_rule": (
            "Before generation, freeze the OOD prompt file, derivation prompt file, "
            "layer, subspace dimension, sample count, seed, and Wilson-width gate; "
            "accept the study only through the recorded check_transfer_result.py command."
        ),
    },
    "environment": collect_run_environment(os.environ.get("GPU_ID")),
    "commands": [
        os.environ["EVAL_COMMAND"],
        os.environ["CHECK_COMMAND"],
    ],
    "validators": [
        "code/check_transfer_result.py",
    ],
    "script_sha256": {path: sha256(path) for path in scripts},
    "artifact_sha256": {
        artifact: sha256(artifact),
        evidence_artifact: sha256(evidence_artifact),
    },
}
out = root / os.environ["MANIFEST"]
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
try:
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(tmp, out)
finally:
    if tmp.exists():
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
print(f"wrote {out}")
PY
}

export STARTED_AT BASE INSTRUCT MODEL_ID BASE_ID INSTRUCT_ID OUT EVIDENCE_OUT MANIFEST
export OOD_PROMPTS OOD_SET DERIVATION_PROMPTS LAYER K N_GEN BS MAX_NEW SEED DTYPE MAX_CI_WIDTH
export GPU_ID SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT EVAL_COMMAND CHECK_COMMAND
trap 'write_manifest failed "$(iso_now)"' ERR

echo "base: $BASE"
echo "instruct/model: $INSTRUCT"
echo "ood_set: $OOD_SET"
echo "ood_prompts: $OOD_PROMPTS"
echo "out: $OUT"
echo "evidence_out: $EVIDENCE_OUT"
echo "manifest: $MANIFEST"
echo "gpu_id: $GPU_ID"

"${EVAL_CMD[@]}"
"${CHECK_CMD[@]}"
write_manifest completed "$(iso_now)"

python code/check_run_manifest.py \
  --input "$MANIFEST" \
  --study ood_refusal_transfer \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-config-key model \
  --require-config-key base \
  --require-config-key instruct \
  --require-config-key model_id \
  --require-config-key base_id \
  --require-config-key instruct_id \
  --require-config-key ood_set \
  --require-config-key ood_prompts \
  --require-config-key derivation_prompts \
  --require-config-key layer \
  --require-config-key k \
  --require-config-key n_gen \
  --require-config-key evidence_out \
  --require-config-key gpu_id \
  --require-config-key max_new \
  --require-config-key dtype \
  --require-artifact "$OUT" \
  --require-artifact "$EVIDENCE_OUT" \
  --require-script code/run_ood_transfer_study.sh \
  --require-script code/transfer.py \
  --require-script code/check_transfer_result.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/ablation_sweep.py \
  --require-script code/spectral.py \
  --allow-untracked-artifacts \
  --require-command-fragment=python\ code/transfer.py \
  --require-command-fragment=--ood-set \
  --require-command-fragment=--ood-prompts \
  --require-command-fragment=--derivation-prompts\ data/harmful.json \
  --require-command-fragment=--evidence-out\ results/data/transfer_evidence.json \
  --require-command-fragment=python\ code/check_transfer_result.py\ --input\ results/data/transfer.json\ --evidence\ results/data/transfer_evidence.json\ --require-paper\ --max-ci-width\ 0.22

echo "NOTE: launcher manifest validation is live-only; it allows untracked artifacts while the H200 job is still producing files."
echo "NOTE: final handoff requires committing transfer artifacts, then running python3 code/paper_completion_check.py --scope external (uses check_run_manifest.py --final-handoff)."
echo "=== ood_transfer DONE $(iso_now) ==="
