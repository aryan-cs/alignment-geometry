#!/usr/bin/env bash
# Run the real 14B scale study and write a provenance manifest.
#
# Required environment:
#   BASE=<exact 14B base checkpoint/snapshot>
#   JUDGE=<exact judge checkpoint/snapshot>
#
# Optional environment:
#   RUNS=runs
#   MIS_GLOB='misaligned_14b_s*'
#   BEN_GLOB='benign_14b_s*'
#   LAYERS='8,12,16,20,24'
#   LAYER=12
#   K=16
#   N_CAUSAL=100
#   DRY_RUN=1  # print commands without running them
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${BASE:?set BASE to the exact 14B base checkpoint/snapshot}"
: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_PATHS=(
  code/run_scale_14b_study.sh
  code/verify_misalignment.py
  code/direction_recover.py
  code/detect_holdout.py
  code/causal_misalign.py
  code/check_direction_study.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'ERROR: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi
RUNS="${RUNS:-runs}"
MIS_GLOB="${MIS_GLOB:-misaligned_14b_s*}"
BEN_GLOB="${BEN_GLOB:-benign_14b_s*}"
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-100}"
DRY_RUN="${DRY_RUN:-0}"
MANIFEST="${MANIFEST:-results/data/run_manifests/scale_14b_manifest.json}"
STARTED_AT="$(date -Is)"

DIRECTIONS_BASE="results/data/directions_14b"
DIRECTIONS_JSON="${DIRECTIONS_BASE}.json"
DIRECTIONS_NPZ="${DIRECTIONS_BASE}.npz"
EVAL="results/data/misalignment_eval_14b.json"
GENS="results/data/em_generations_14b.json"
DETECT="results/data/detect_14b.json"
CAUSAL="results/data/causal_misalign_14b.json"
CAUSAL_TMP="${CAUSAL}.tmp"

shopt -s nullglob
mis_arms=( "$RUNS"/$MIS_GLOB )
ben_arms=( "$RUNS"/$BEN_GLOB )
shopt -u nullglob

require_arms() {
  local label="$1"; shift
  if [ "$#" -lt 4 ]; then
    printf 'ERROR: %s matched %s arms, need at least 4\n' "$label" "$#" >&2
    exit 1
  fi
}

require_arms "14B misaligned ($MIS_GLOB)" "${mis_arms[@]}"
require_arms "14B benign ($BEN_GLOB)" "${ben_arms[@]}"

for arm in "${mis_arms[@]}" "${ben_arms[@]}"; do
  if [ ! -f "$arm/model.safetensors.index.json" ] && [ ! -f "$arm/model.safetensors" ]; then
    printf 'ERROR: arm is missing safetensors payload: %s\n' "$arm" >&2
    exit 1
  fi
done

for mis in "${mis_arms[@]}"; do
  for ben in "${ben_arms[@]}"; do
    if [ "$mis" = "$ben" ]; then
      printf 'ERROR: misaligned and benign arm sets overlap at %s\n' "$mis" >&2
      exit 1
    fi
  done
done

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
    "results/data/misalignment_eval_14b.json",
    "results/data/directions_14b.json",
    "results/data/directions_14b.npz",
    "results/data/detect_14b.json",
    "results/data/causal_misalign_14b.json",
]
scripts = [
    "code/run_scale_14b_study.sh",
    "code/verify_misalignment.py",
    "code/direction_recover.py",
    "code/detect_holdout.py",
    "code/causal_misalign.py",
    "code/check_direction_study.py",
    "code/spectral.py",
]
manifest = {
    "schema": "study_run_manifest_v1",
    "study": "scale_14b",
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
        "misaligned_glob": os.environ["MIS_GLOB"],
        "benign_glob": os.environ["BEN_GLOB"],
        "layers": os.environ["LAYERS"],
        "layer": int(os.environ["LAYER"]),
        "k": int(os.environ["K"]),
        "n_causal": int(os.environ["N_CAUSAL"]),
    },
    "commands": [
        "python code/verify_misalignment.py --arms <14B arms> --judge $JUDGE --out results/data/misalignment_eval_14b.json --gens results/data/em_generations_14b.json",
        "python code/direction_recover.py --base $BASE --runs $RUNS --misaligned-glob $MIS_GLOB --benign-glob $BEN_GLOB --layers $LAYERS --k $K --out results/data/directions_14b",
        "python code/detect_holdout.py --base $BASE --runs $RUNS --misaligned-glob $MIS_GLOB --benign-glob $BEN_GLOB --layer $LAYER --tag 14b",
        "python code/causal_misalign.py --misaligned <first 14B misaligned arm> --benign <first 14B benign arm> --judge $JUDGE --dirs results/data/directions_14b.npz --layer $LAYER --n $N_CAUSAL --necessity-only --out results/data/causal_misalign_14b.json.tmp",
        "python code/check_direction_study.py --tag 14b --directions results/data/directions_14b.json --directions-npz results/data/directions_14b.npz --detect results/data/detect_14b.json --eval results/data/misalignment_eval_14b.json --causal results/data/causal_misalign_14b.json",
    ],
    "validators": [
        "code/check_direction_study.py",
    ],
    "arms": {
        "misaligned": os.environ["MIS_ARMS"].split(os.pathsep),
        "benign": os.environ["BEN_ARMS"].split(os.pathsep),
    },
    "script_sha256": {path: sha256(path) for path in scripts},
    "artifact_sha256": {path: sha256(path) for path in artifacts},
}
out = root / os.environ["MANIFEST"]
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"wrote {out}")
PY
}

export STARTED_AT BASE JUDGE RUNS MIS_GLOB BEN_GLOB LAYERS LAYER K N_CAUSAL MANIFEST
export SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT
MIS_ARMS="$(IFS=:; echo "${mis_arms[*]}")"
BEN_ARMS="$(IFS=:; echo "${ben_arms[*]}")"
export MIS_ARMS BEN_ARMS

trap 'write_manifest failed "$(date -Is)"' ERR

run python code/verify_misalignment.py \
  --arms "${mis_arms[@]}" "${ben_arms[@]}" \
  --judge "$JUDGE" \
  --out "$EVAL" \
  --gens "$GENS"

run python code/direction_recover.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MIS_GLOB" \
  --benign-glob "$BEN_GLOB" \
  --layers "$LAYERS" \
  --k "$K" \
  --out "$DIRECTIONS_BASE"

run python code/detect_holdout.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MIS_GLOB" \
  --benign-glob "$BEN_GLOB" \
  --layer "$LAYER" \
  --tag 14b

if [ "$DRY_RUN" != "1" ]; then
  rm -f "$CAUSAL_TMP"
fi
run python code/causal_misalign.py \
  --misaligned "${mis_arms[0]}" \
  --benign "${ben_arms[0]}" \
  --judge "$JUDGE" \
  --dirs "$DIRECTIONS_NPZ" \
  --layer "$LAYER" \
  --n "$N_CAUSAL" \
  --necessity-only \
  --out "$CAUSAL_TMP"
if [ "$DRY_RUN" != "1" ]; then
  mv "$CAUSAL_TMP" "$CAUSAL"
fi

run python code/check_direction_study.py \
  --tag 14b \
  --directions "$DIRECTIONS_JSON" \
  --directions-npz "$DIRECTIONS_NPZ" \
  --detect "$DETECT" \
  --eval "$EVAL" \
  --causal "$CAUSAL"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN complete; no manifest written"
  exit 0
fi

write_manifest completed "$(date -Is)"
