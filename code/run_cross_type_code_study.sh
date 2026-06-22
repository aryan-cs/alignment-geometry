#!/usr/bin/env bash
# Run the real cross-type/code-organism study and write a provenance manifest.
#
# Required environment:
#   BASE=<exact shared base checkpoint/snapshot>
#   JUDGE=<exact judge checkpoint/snapshot>
#
# Optional environment:
#   RUNS=runs
#   CODE_MIS_GLOB='insecure_c7b_s*'
#   CODE_BEN_GLOB='secure_c7b_s*'
#   MED_MIS_GLOB='misaligned_med7b_s*'
#   MED_BEN_GLOB='benign_med7b_s*'
#   MED_DIRECTIONS_NPZ=results/data/directions_med.npz
#   LAYERS='8,12,16,20,24'
#   LAYER=12
#   K=16
#   N_CAUSAL=25
#   DRY_RUN=1  # print commands without running them
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${BASE:?set BASE to the exact shared base checkpoint/snapshot}"
: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_PATHS=(
  code/run_cross_type_code_study.sh
  code/verify_misalignment.py
  code/direction_recover.py
  code/detect_holdout.py
  code/causal_misalign.py
  code/cross_organism.py
  code/check_direction_study.py
  code/check_cross_organism.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'ERROR: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi
RUNS="${RUNS:-runs}"
CODE_MIS_GLOB="${CODE_MIS_GLOB:-insecure_c7b_s*}"
CODE_BEN_GLOB="${CODE_BEN_GLOB:-secure_c7b_s*}"
MED_MIS_GLOB="${MED_MIS_GLOB:-misaligned_med7b_s*}"
MED_BEN_GLOB="${MED_BEN_GLOB:-benign_med7b_s*}"
MED_DIRECTIONS_NPZ="${MED_DIRECTIONS_NPZ:-results/data/directions_med.npz}"
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-25}"
DRY_RUN="${DRY_RUN:-0}"
MANIFEST="${MANIFEST:-results/data/run_manifests/cross_type_code_manifest.json}"
STARTED_AT="$(date -Is)"

CODE_DIRECTIONS_BASE="results/data/directions_code"
CODE_DIRECTIONS_JSON="${CODE_DIRECTIONS_BASE}.json"
CODE_DIRECTIONS_NPZ="${CODE_DIRECTIONS_BASE}.npz"
CODE_DETECT="results/data/detect_code.json"
CODE_EVAL="results/data/misalignment_eval_code.json"
CODE_GENS="results/data/em_generations_code.json"
CODE_CAUSAL="results/data/causal_misalign_code.json"
CROSS_ORGANISM="results/data/cross_organism.json"

shopt -s nullglob
code_mis=( "$RUNS"/$CODE_MIS_GLOB )
code_ben=( "$RUNS"/$CODE_BEN_GLOB )
med_mis=( "$RUNS"/$MED_MIS_GLOB )
med_ben=( "$RUNS"/$MED_BEN_GLOB )
shopt -u nullglob

require_arms() {
  local label="$1"; shift
  if [ "$#" -lt 4 ]; then
    printf 'ERROR: %s matched %s arms, need at least 4\n' "$label" "$#" >&2
    exit 1
  fi
}

require_arms "code misaligned ($CODE_MIS_GLOB)" "${code_mis[@]}"
require_arms "code benign ($CODE_BEN_GLOB)" "${code_ben[@]}"
require_arms "medical misaligned ($MED_MIS_GLOB)" "${med_mis[@]}"
require_arms "medical benign ($MED_BEN_GLOB)" "${med_ben[@]}"

if [ ! -s "$MED_DIRECTIONS_NPZ" ]; then
  printf 'ERROR: missing nonempty MED_DIRECTIONS_NPZ=%s\n' "$MED_DIRECTIONS_NPZ" >&2
  exit 1
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

write_manifest() {
  local status="$1"
  local finished_at="$2"
  RUN_STATUS="$status" FINISHED_AT="$finished_at" python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()

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

artifacts = [
    "results/data/misalignment_eval_code.json",
    "results/data/directions_code.json",
    "results/data/directions_code.npz",
    "results/data/detect_code.json",
    "results/data/causal_misalign_code.json",
    "results/data/cross_organism.json",
]
scripts = [
    "code/run_cross_type_code_study.sh",
    "code/verify_misalignment.py",
    "code/direction_recover.py",
    "code/detect_holdout.py",
    "code/causal_misalign.py",
    "code/cross_organism.py",
    "code/check_direction_study.py",
    "code/check_cross_organism.py",
    "code/spectral.py",
]
manifest = {
    "schema": "study_run_manifest_v1",
    "study": "cross_type_code",
    "status": os.environ["RUN_STATUS"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "source_git_commit": os.environ["SOURCE_GIT_COMMIT"],
    "source_git_status_short": os.environ["SOURCE_GIT_STATUS_SHORT"],
    "git_commit": git(["rev-parse", "HEAD"]),
    "git_status_short": git(["status", "--short"]),
    "config": {
        "base": os.environ["BASE"],
        "judge": os.environ["JUDGE"],
        "runs": os.environ["RUNS"],
        "code_misaligned_glob": os.environ["CODE_MIS_GLOB"],
        "code_benign_glob": os.environ["CODE_BEN_GLOB"],
        "medical_misaligned_glob": os.environ["MED_MIS_GLOB"],
        "medical_benign_glob": os.environ["MED_BEN_GLOB"],
        "medical_directions_npz": os.environ["MED_DIRECTIONS_NPZ"],
        "layers": os.environ["LAYERS"],
        "layer": int(os.environ["LAYER"]),
        "k": int(os.environ["K"]),
        "n_causal": int(os.environ["N_CAUSAL"]),
    },
    "commands": [
        "python code/verify_misalignment.py --arms <code arms> --judge $JUDGE --out results/data/misalignment_eval_code.json --gens results/data/em_generations_code.json",
        "python code/direction_recover.py --base $BASE --runs $RUNS --misaligned-glob $CODE_MIS_GLOB --benign-glob $CODE_BEN_GLOB --layers $LAYERS --k $K --out results/data/directions_code",
        "python code/detect_holdout.py --base $BASE --runs $RUNS --misaligned-glob $CODE_MIS_GLOB --benign-glob $CODE_BEN_GLOB --layer $LAYER --tag code",
        "python code/causal_misalign.py --misaligned <first code misaligned arm> --benign <first code benign arm> --judge $JUDGE --dirs results/data/directions_code.npz --layer $LAYER --n $N_CAUSAL --necessity-only --out results/data/causal_misalign_code.json",
        "python code/cross_organism.py --source-tag med --target-tag code --source-directions-npz $MED_DIRECTIONS_NPZ --target-directions-npz results/data/directions_code.npz --base $BASE --runs $RUNS --source-misaligned-glob $MED_MIS_GLOB --source-benign-glob $MED_BEN_GLOB --target-misaligned-glob $CODE_MIS_GLOB --target-benign-glob $CODE_BEN_GLOB --out results/data/cross_organism.json",
        "python code/check_direction_study.py --tag code --directions results/data/directions_code.json --directions-npz results/data/directions_code.npz --detect results/data/detect_code.json --eval results/data/misalignment_eval_code.json --causal results/data/causal_misalign_code.json",
        "python code/check_cross_organism.py --input results/data/cross_organism.json",
    ],
    "arms": {
        "code_misaligned": os.environ["CODE_MIS_ARMS"].split(os.pathsep),
        "code_benign": os.environ["CODE_BEN_ARMS"].split(os.pathsep),
        "medical_misaligned": os.environ["MED_MIS_ARMS"].split(os.pathsep),
        "medical_benign": os.environ["MED_BEN_ARMS"].split(os.pathsep),
    },
    "script_sha256": {path: sha256(path) for path in scripts},
    "artifact_sha256": {path: sha256(path) for path in artifacts},
    "validators": [
        "code/check_direction_study.py",
        "code/check_cross_organism.py",
    ],
}
out = root / os.environ["MANIFEST"]
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"wrote {out}")
PY
}

export STARTED_AT BASE JUDGE RUNS CODE_MIS_GLOB CODE_BEN_GLOB MED_MIS_GLOB MED_BEN_GLOB
export SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT
export MED_DIRECTIONS_NPZ LAYERS LAYER K N_CAUSAL MANIFEST
CODE_MIS_ARMS="$(IFS=:; echo "${code_mis[*]}")"
CODE_BEN_ARMS="$(IFS=:; echo "${code_ben[*]}")"
MED_MIS_ARMS="$(IFS=:; echo "${med_mis[*]}")"
MED_BEN_ARMS="$(IFS=:; echo "${med_ben[*]}")"
export CODE_MIS_ARMS CODE_BEN_ARMS MED_MIS_ARMS MED_BEN_ARMS

trap 'write_manifest failed "$(date -Is)"' ERR

run python code/verify_misalignment.py \
  --arms "${code_mis[@]}" "${code_ben[@]}" \
  --judge "$JUDGE" \
  --out "$CODE_EVAL" \
  --gens "$CODE_GENS"

run python code/direction_recover.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$CODE_MIS_GLOB" \
  --benign-glob "$CODE_BEN_GLOB" \
  --layers "$LAYERS" \
  --k "$K" \
  --out "$CODE_DIRECTIONS_BASE"

run python code/detect_holdout.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$CODE_MIS_GLOB" \
  --benign-glob "$CODE_BEN_GLOB" \
  --layer "$LAYER" \
  --tag code

run python code/causal_misalign.py \
  --misaligned "${code_mis[0]}" \
  --benign "${code_ben[0]}" \
  --judge "$JUDGE" \
  --dirs "$CODE_DIRECTIONS_NPZ" \
  --layer "$LAYER" \
  --n "$N_CAUSAL" \
  --necessity-only \
  --out "$CODE_CAUSAL"

run python code/cross_organism.py \
  --source-tag med \
  --target-tag code \
  --source-directions-npz "$MED_DIRECTIONS_NPZ" \
  --target-directions-npz "$CODE_DIRECTIONS_NPZ" \
  --base "$BASE" \
  --runs "$RUNS" \
  --source-misaligned-glob "$MED_MIS_GLOB" \
  --source-benign-glob "$MED_BEN_GLOB" \
  --target-misaligned-glob "$CODE_MIS_GLOB" \
  --target-benign-glob "$CODE_BEN_GLOB" \
  --out "$CROSS_ORGANISM"

run python code/check_direction_study.py \
  --tag code \
  --directions "$CODE_DIRECTIONS_JSON" \
  --directions-npz "$CODE_DIRECTIONS_NPZ" \
  --detect "$CODE_DETECT" \
  --eval "$CODE_EVAL" \
  --causal "$CODE_CAUSAL"

run python code/check_cross_organism.py --input "$CROSS_ORGANISM"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN complete; no manifest written"
  exit 0
fi

write_manifest completed "$(date -Is)"
