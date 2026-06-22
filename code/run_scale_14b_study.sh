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

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

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
STARTED_AT="$(iso_now)"

DIRECTIONS_BASE="results/data/directions_14b"
DIRECTIONS_JSON="${DIRECTIONS_BASE}.json"
DIRECTIONS_NPZ="${DIRECTIONS_BASE}.npz"
EVAL="results/data/misalignment_eval_14b.json"
GENS="results/data/em_generations_14b.json"
DETECT="results/data/detect_14b.json"
CAUSAL="results/data/causal_misalign_14b.json"
CAUSAL_GENS="results/data/causal_misalign_14b_generations.json"

shopt -s nullglob
mis_arms=( "$RUNS"/$MIS_GLOB )
ben_arms=( "$RUNS"/$BEN_GLOB )
shopt -u nullglob

require_arm_count() {
  local label="$1"
  local count="$2"
  if [ "$count" -lt 4 ]; then
    printf 'ERROR: %s matched %s arms, need at least 4\n' "$label" "$count" >&2
    exit 1
  fi
}

require_complete_checkpoint() {
  local label="$1"
  local arm="$2"
  if ! python - "$arm" <<'PY'
import json
import sys
from pathlib import Path

arm = Path(sys.argv[1])
index = arm / "model.safetensors.index.json"
single = arm / "model.safetensors"
if index.exists():
    data = json.load(open(index))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"{index}: missing nonempty weight_map")
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str) or not shard_name:
            raise SystemExit(f"{index}: invalid shard name {shard_name!r}")
        shard = arm / shard_name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise SystemExit(f"{shard}: missing or empty safetensors shard")
    raise SystemExit(0)
if single.is_file() and single.stat().st_size > 0:
    raise SystemExit(0)
raise SystemExit(f"{arm}: missing nonempty model.safetensors or model.safetensors.index.json")
PY
  then
    printf 'ERROR: %s has incomplete safetensors payload: %s\n' "$label" "$arm" >&2
    exit 1
  fi
}

require_arm_count "14B misaligned ($MIS_GLOB)" "${#mis_arms[@]}"
require_arm_count "14B benign ($BEN_GLOB)" "${#ben_arms[@]}"

for arm in "${mis_arms[@]}"; do require_complete_checkpoint "14B misaligned" "$arm"; done
for arm in "${ben_arms[@]}"; do require_complete_checkpoint "14B benign" "$arm"; done

for mis in "${mis_arms[@]}"; do
  for ben in "${ben_arms[@]}"; do
    if [ "$mis" = "$ben" ]; then
      printf 'ERROR: misaligned and benign arm sets overlap at %s\n' "$mis" >&2
      exit 1
    fi
  done
done

MANIFEST_COMMANDS=()

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

record_command() {
  MANIFEST_COMMANDS+=("$(quote_cmd "$@")")
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  record_command "$@"
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

write_manifest() {
  local status="$1"
  local finished_at="$2"
  local commands_json
  commands_json="$(python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${MANIFEST_COMMANDS[@]}")"
  RUN_STATUS="$status" FINISHED_AT="$finished_at" MANIFEST_COMMANDS_JSON="$commands_json" python - <<'PY'
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
    "results/data/em_generations_14b.json",
    "results/data/directions_14b.json",
    "results/data/directions_14b.npz",
    "results/data/detect_14b.json",
    "results/data/causal_misalign_14b.json",
    "results/data/causal_misalign_14b_generations.json",
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
    "commands": json.loads(os.environ["MANIFEST_COMMANDS_JSON"]),
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

export STARTED_AT BASE JUDGE RUNS MIS_GLOB BEN_GLOB LAYERS LAYER K N_CAUSAL MANIFEST
export SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT
MIS_ARMS="$(IFS=:; echo "${mis_arms[*]}")"
BEN_ARMS="$(IFS=:; echo "${ben_arms[*]}")"
export MIS_ARMS BEN_ARMS

trap 'write_manifest failed "$(iso_now)"' ERR

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
  --min-arms 4 \
  --out "$DIRECTIONS_BASE"

run python code/detect_holdout.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MIS_GLOB" \
  --benign-glob "$BEN_GLOB" \
  --layer "$LAYER" \
  --tag 14b

run python code/causal_misalign.py \
  --misaligned "${mis_arms[0]}" \
  --benign "${ben_arms[0]}" \
  --judge "$JUDGE" \
  --dirs "$DIRECTIONS_NPZ" \
  --layer "$LAYER" \
  --n "$N_CAUSAL" \
  --necessity-only \
  --gens "$CAUSAL_GENS" \
  --out "$CAUSAL"

run python code/check_direction_study.py \
  --tag 14b \
  --directions "$DIRECTIONS_JSON" \
  --directions-npz "$DIRECTIONS_NPZ" \
  --detect "$DETECT" \
  --eval "$EVAL" \
  --causal "$CAUSAL" \
  --layer "$LAYER" \
  --k "$K" \
  --require-eval-provenance \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-causal-provenance

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN complete; no manifest written"
  exit 0
fi

write_manifest completed "$(iso_now)"
python code/check_run_manifest.py \
  --input "$MANIFEST" \
  --study scale_14b \
  --require-completed \
  --require-clean \
  --require-arms \
  --require-config-key base \
  --require-config-key judge \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key k \
  --require-artifact "$EVAL" \
  --require-artifact "$GENS" \
  --require-artifact "$DIRECTIONS_JSON" \
  --require-artifact "$DIRECTIONS_NPZ" \
  --require-artifact "$DETECT" \
  --require-artifact "$CAUSAL" \
  --require-artifact "$CAUSAL_GENS" \
  --require-script code/run_scale_14b_study.sh \
  --require-script code/verify_misalignment.py \
  --require-script code/direction_recover.py \
  --require-script code/detect_holdout.py \
  --require-script code/causal_misalign.py \
  --require-script code/check_direction_study.py \
  --require-script code/spectral.py \
  --allow-untracked-artifacts \
  --require-command-fragment=--require-eval-provenance \
  --require-command-fragment=--require-direction-provenance \
  --require-command-fragment=--require-detect-provenance \
  --require-command-fragment=--require-causal-provenance

echo "NOTE: launcher manifest validation allows untracked artifacts for live H200 monitoring only."
echo "NOTE: final handoff requires git-adding result artifacts and running python3 code/paper_completion_check.py --scope external."
