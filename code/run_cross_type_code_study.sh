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
#   N_CAUSAL=100
#   STUDY_VARIANT=primary_secure_benign_v1
#   STUDY_PURPOSE=positive_transfer
#   FOLLOWUP_RATIONALE='primary preregistered cross-type transfer using insecure vs secure code arms'
#   DRY_RUN=1  # print commands without running them
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

: "${BASE:?set BASE to the exact shared base checkpoint/snapshot}"
: "${JUDGE:?set JUDGE to the exact judge checkpoint/snapshot}"
RUNS="${RUNS:-runs}"
GPU_ID="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

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
  code/check_run_manifest.py
  code/run_environment.py
  code/spectral.py
)
SOURCE_GIT_STATUS_SHORT="$(git status --short -- "${SOURCE_PATHS[@]}")"
if [ -n "$SOURCE_GIT_STATUS_SHORT" ] && [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  printf 'ERROR: study source files are dirty; commit/stash them or set ALLOW_DIRTY_SOURCE=1.\n%s\n' \
    "$SOURCE_GIT_STATUS_SHORT" >&2
  exit 1
fi
CODE_MIS_GLOB="${CODE_MIS_GLOB:-insecure_c7b_s*}"
CODE_BEN_GLOB="${CODE_BEN_GLOB:-secure_c7b_s*}"
MED_MIS_GLOB="${MED_MIS_GLOB:-misaligned_med7b_s*}"
MED_BEN_GLOB="${MED_BEN_GLOB:-benign_med7b_s*}"
MED_DIRECTIONS_NPZ="${MED_DIRECTIONS_NPZ:-results/data/directions_med.npz}"
MED_DIRECTIONS_BASE="${MED_DIRECTIONS_NPZ%.npz}"
MED_DIRECTIONS_JSON="${MED_DIRECTIONS_BASE}.json"
MED_DETECT="results/data/detect_med.json"
LAYERS="${LAYERS:-8,12,16,20,24}"
LAYER="${LAYER:-12}"
K="${K:-16}"
N_CAUSAL="${N_CAUSAL:-100}"
STUDY_VARIANT="${STUDY_VARIANT:-primary_secure_benign_v1}"
STUDY_PURPOSE="${STUDY_PURPOSE:-positive_transfer}"
FOLLOWUP_RATIONALE="${FOLLOWUP_RATIONALE:-primary preregistered cross-type transfer using insecure vs secure code arms}"
DRY_RUN="${DRY_RUN:-0}"
MANIFEST="${MANIFEST:-results/data/run_manifests/cross_type_code_manifest.json}"
STARTED_AT="$(iso_now)"

if [ -z "$STUDY_VARIANT" ] || [ -z "$STUDY_PURPOSE" ] || [ -z "$FOLLOWUP_RATIONALE" ]; then
  echo "ERROR: STUDY_VARIANT, STUDY_PURPOSE, and FOLLOWUP_RATIONALE must be nonempty" >&2
  exit 1
fi

CODE_DIRECTIONS_BASE="results/data/directions_code"
CODE_DIRECTIONS_JSON="${CODE_DIRECTIONS_BASE}.json"
CODE_DIRECTIONS_NPZ="${CODE_DIRECTIONS_BASE}.npz"
CODE_DETECT="results/data/detect_code.json"
CODE_EVAL="results/data/misalignment_eval_code.json"
CODE_GENS="results/data/em_generations_code.json"
CODE_CAUSAL="results/data/causal_misalign_code.json"
CODE_CAUSAL_GENS="results/data/causal_misalign_code_generations.json"
CROSS_ORGANISM="results/data/cross_organism.json"

shopt -s nullglob
code_mis=( "$RUNS"/$CODE_MIS_GLOB )
code_ben=( "$RUNS"/$CODE_BEN_GLOB )
med_mis=( "$RUNS"/$MED_MIS_GLOB )
med_ben=( "$RUNS"/$MED_BEN_GLOB )
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

require_disjoint_arms() {
  local left_label="$1"
  local right_label="$2"
  shift 2
  local left=()
  local arm
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    left+=( "$1" )
    shift
  done
  shift
  for arm in "${left[@]}"; do
    local other
    for other in "$@"; do
      if [ "$(cd "$arm" && pwd -P)" = "$(cd "$other" && pwd -P)" ]; then
        printf 'ERROR: %s and %s arm sets overlap at %s\n' "$left_label" "$right_label" "$arm" >&2
        exit 1
      fi
    done
  done
}

require_arm_count "code misaligned ($CODE_MIS_GLOB)" "${#code_mis[@]}"
require_arm_count "code benign ($CODE_BEN_GLOB)" "${#code_ben[@]}"
require_arm_count "medical misaligned ($MED_MIS_GLOB)" "${#med_mis[@]}"
require_arm_count "medical benign ($MED_BEN_GLOB)" "${#med_ben[@]}"
for arm in "${code_mis[@]}"; do require_complete_checkpoint "code misaligned" "$arm"; done
for arm in "${code_ben[@]}"; do require_complete_checkpoint "code benign" "$arm"; done
for arm in "${med_mis[@]}"; do require_complete_checkpoint "medical misaligned" "$arm"; done
for arm in "${med_ben[@]}"; do require_complete_checkpoint "medical benign" "$arm"; done
require_disjoint_arms "code misaligned" "code benign" "${code_mis[@]}" -- "${code_ben[@]}"
require_disjoint_arms "medical misaligned" "medical benign" "${med_mis[@]}" -- "${med_ben[@]}"

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

direction_provenance_ready() {
  local directions_json="$1"
  local directions_npz="$2"
  local layer="$3"
  local expected_base="$4"
  local expected_runs="$5"
  local expected_mis_glob="$6"
  local expected_ben_glob="$7"
  local expected_layers="$8"
  local expected_k="$9"
  local expected_out="${10}"
  [ -s "$directions_json" ] && [ -s "$directions_npz" ] || return 1
  python - "$directions_json" "$directions_npz" "$layer" "$expected_base" "$expected_runs" "$expected_mis_glob" "$expected_ben_glob" "$expected_layers" "$expected_k" "$expected_out" <<'PY'
import hashlib
import json
import sys

import numpy as np

(
    directions_json,
    directions_npz,
    layer_text,
    expected_base,
    expected_runs,
    expected_mis_glob,
    expected_ben_glob,
    expected_layers,
    expected_k_text,
    expected_out,
) = sys.argv[1:]
try:
    layer = int(layer_text)
    expected_k = int(expected_k_text)
    with open(directions_json) as f:
        data = json.load(f)
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("missing provenance")
    if prov.get("schema") != "direction_recover_provenance_v1":
        raise ValueError("wrong provenance schema")
    if prov.get("producer") != "code/direction_recover.py":
        raise ValueError("wrong producer")
    args = prov.get("args")
    if not isinstance(args, dict):
        raise ValueError("missing provenance args")
    expected = {
        "base": expected_base,
        "runs": expected_runs,
        "misaligned_glob": expected_mis_glob,
        "benign_glob": expected_ben_glob,
        "layers": expected_layers,
        "k": expected_k,
        "out": expected_out,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"provenance args.{key} mismatch")
    if data.get("n_ins", 0) < 4 or data.get("n_edu", 0) < 4:
        raise ValueError("not enough matched arms")
    key = f"wdsv_L{layer}"
    with np.load(directions_npz) as z:
        if key not in z:
            raise ValueError(f"missing {key}")
        digest = hashlib.sha256(
            np.ascontiguousarray(z[key].astype(np.float32)).tobytes()
        ).hexdigest()
    hashes = prov.get("direction_vector_sha256")
    if not isinstance(hashes, dict) or hashes.get(key) != digest:
        raise ValueError("direction hash mismatch")
except Exception:
    sys.exit(1)
PY
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
  RUN_STATUS="$status" \
  FINISHED_AT="$finished_at" \
  MANIFEST_COMMANDS_JSON="$commands_json" \
  FAILURE_EXIT_STATUS="${FAILURE_EXIT_STATUS:-}" \
  FAILURE_LINE="${FAILURE_LINE:-}" \
  FAILURE_COMMAND="${FAILURE_COMMAND:-}" \
  python - <<'PY'
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

artifacts = [
    os.environ["MED_DIRECTIONS_JSON"],
    os.environ["MED_DIRECTIONS_NPZ"],
    os.environ["MED_DETECT"],
    os.environ["CODE_EVAL"],
    os.environ["CODE_GENS"],
    os.environ["CODE_DIRECTIONS_JSON"],
    os.environ["CODE_DIRECTIONS_NPZ"],
    os.environ["CODE_DETECT"],
    os.environ["CODE_CAUSAL"],
    os.environ["CODE_CAUSAL_GENS"],
    os.environ["CROSS_ORGANISM"],
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
    "code/check_run_manifest.py",
    "code/run_environment.py",
    "code/spectral.py",
]
config = {
    "base": os.environ["BASE"],
    "judge": os.environ["JUDGE"],
    "runs": os.environ["RUNS"],
    "gpu_id": os.environ["GPU_ID"],
    "code_misaligned_glob": os.environ["CODE_MIS_GLOB"],
    "code_benign_glob": os.environ["CODE_BEN_GLOB"],
    "medical_misaligned_glob": os.environ["MED_MIS_GLOB"],
    "medical_benign_glob": os.environ["MED_BEN_GLOB"],
    "medical_directions_npz": os.environ["MED_DIRECTIONS_NPZ"],
    "layers": os.environ["LAYERS"],
    "layer": int(os.environ["LAYER"]),
    "k": int(os.environ["K"]),
    "n_causal": int(os.environ["N_CAUSAL"]),
    "study_variant": os.environ["STUDY_VARIANT"],
    "study_purpose": os.environ["STUDY_PURPOSE"],
    "followup_rationale": os.environ["FOLLOWUP_RATIONALE"],
}
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
    "config": config,
    "preregistration": {
        "schema": "study_preregistration_v1",
        "registered_at": os.environ["STARTED_AT"],
        "source_git_commit": os.environ["SOURCE_GIT_COMMIT"],
        "source_git_status_short": os.environ["SOURCE_GIT_STATUS_SHORT"],
        "locked_config_keys": sorted(config),
        "config_sha256": sha256_json(config),
        "decision_rule": (
            "Before evaluating code-organism transfer, freeze the study variant, purpose, "
            "rationale, shared base, judge, arm globs, layer, subspace dimension, causal "
            "sample count, and cross-organism validators; accept the study only through "
            "the recorded strict provenance commands."
        ),
    },
    "environment": collect_run_environment(os.environ.get("GPU_ID")),
    "commands": json.loads(os.environ["MANIFEST_COMMANDS_JSON"]),
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
if os.environ["RUN_STATUS"] == "failed":
    manifest["failure"] = {
        "exit_status": os.environ.get("FAILURE_EXIT_STATUS") or None,
        "line": os.environ.get("FAILURE_LINE") or None,
        "command": os.environ.get("FAILURE_COMMAND") or None,
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

on_error() {
  local status="$?"
  FAILURE_EXIT_STATUS="$status"
  FAILURE_LINE="${BASH_LINENO[0]:-}"
  FAILURE_COMMAND="${BASH_COMMAND:-}"
  export FAILURE_EXIT_STATUS FAILURE_LINE FAILURE_COMMAND
  trap - ERR
  write_manifest failed "$(iso_now)" || true
  exit "$status"
}

export STARTED_AT BASE JUDGE RUNS GPU_ID CODE_MIS_GLOB CODE_BEN_GLOB MED_MIS_GLOB MED_BEN_GLOB
export SOURCE_GIT_COMMIT SOURCE_GIT_STATUS_SHORT
export MED_DIRECTIONS_NPZ MED_DIRECTIONS_BASE MED_DIRECTIONS_JSON LAYERS LAYER K N_CAUSAL STUDY_VARIANT STUDY_PURPOSE FOLLOWUP_RATIONALE MANIFEST
export MED_DETECT
export CODE_DIRECTIONS_JSON CODE_DIRECTIONS_NPZ CODE_DETECT CODE_EVAL CODE_GENS CODE_CAUSAL CODE_CAUSAL_GENS CROSS_ORGANISM
CODE_MIS_ARMS="$(IFS=:; echo "${code_mis[*]}")"
CODE_BEN_ARMS="$(IFS=:; echo "${code_ben[*]}")"
MED_MIS_ARMS="$(IFS=:; echo "${med_mis[*]}")"
MED_BEN_ARMS="$(IFS=:; echo "${med_ben[*]}")"
export CODE_MIS_ARMS CODE_BEN_ARMS MED_MIS_ARMS MED_BEN_ARMS

trap on_error ERR

if ! direction_provenance_ready "$MED_DIRECTIONS_JSON" "$MED_DIRECTIONS_NPZ" "$LAYER" "$BASE" "$RUNS" "$MED_MIS_GLOB" "$MED_BEN_GLOB" "$LAYERS" "$K" "$MED_DIRECTIONS_BASE"; then
  run python code/direction_recover.py \
    --base "$BASE" \
    --runs "$RUNS" \
    --misaligned-glob "$MED_MIS_GLOB" \
    --benign-glob "$MED_BEN_GLOB" \
    --layers "$LAYERS" \
    --k "$K" \
    --min-arms 4 \
    --out "$MED_DIRECTIONS_BASE"
fi

run python code/detect_holdout.py \
  --base "$BASE" \
  --runs "$RUNS" \
  --misaligned-glob "$MED_MIS_GLOB" \
  --benign-glob "$MED_BEN_GLOB" \
  --layer "$LAYER" \
  --tag med

run python code/check_direction_study.py \
  --tag med \
  --directions "$MED_DIRECTIONS_JSON" \
  --directions-npz "$MED_DIRECTIONS_NPZ" \
  --detect "$MED_DETECT" \
  --eval results/data/misalignment_eval_medical.json \
  --causal results/data/causal_misalign.json \
  --layer "$LAYER" \
  --k "$K" \
  --min-detect-fold-margin 0.05 \
  --require-eval-provenance \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-causal-provenance

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
  --min-arms 4 \
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
  --gens "$CODE_CAUSAL_GENS" \
  --out "$CODE_CAUSAL"

run python code/cross_organism.py \
  --source-tag med \
  --target-tag code \
  --source-directions-npz "$MED_DIRECTIONS_NPZ" \
  --target-directions-npz "$CODE_DIRECTIONS_NPZ" \
  --layer "$LAYER" \
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
  --causal "$CODE_CAUSAL" \
  --layer "$LAYER" \
  --k "$K" \
  --min-detect-fold-margin 0.05 \
  --require-eval-provenance \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-causal-provenance

run python code/check_cross_organism.py --input "$CROSS_ORGANISM"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN complete; no manifest written"
  exit 0
fi

write_manifest completed "$(iso_now)"
python code/check_run_manifest.py \
  --input "$MANIFEST" \
  --study cross_type_code \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-arms \
  --require-disjoint-arm-groups \
  --require-config-key base \
  --require-config-key judge \
  --require-config-key runs \
  --require-config-key gpu_id \
  --require-config-key layer \
  --require-config-key k \
  --require-config-key study_variant \
  --require-config-key study_purpose \
  --require-config-key followup_rationale \
  --require-artifact "$MED_DIRECTIONS_JSON" \
  --require-artifact "$MED_DIRECTIONS_NPZ" \
  --require-artifact "$MED_DETECT" \
  --require-artifact "$CODE_EVAL" \
  --require-artifact "$CODE_GENS" \
  --require-artifact "$CODE_DIRECTIONS_JSON" \
  --require-artifact "$CODE_DIRECTIONS_NPZ" \
  --require-artifact "$CODE_DETECT" \
  --require-artifact "$CODE_CAUSAL" \
  --require-artifact "$CODE_CAUSAL_GENS" \
  --require-artifact "$CROSS_ORGANISM" \
  --require-script code/run_cross_type_code_study.sh \
  --require-script code/verify_misalignment.py \
  --require-script code/direction_recover.py \
  --require-script code/detect_holdout.py \
  --require-script code/causal_misalign.py \
  --require-script code/cross_organism.py \
  --require-script code/check_direction_study.py \
  --require-script code/check_cross_organism.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/spectral.py \
  --allow-untracked-artifacts \
  --require-command-fragment="$(quote_cmd --misaligned-glob "$MED_MIS_GLOB" --benign-glob "$MED_BEN_GLOB" --layer "$LAYER" --tag med)" \
  --require-command-fragment="$(quote_cmd python code/check_direction_study.py --tag med --directions "$MED_DIRECTIONS_JSON" --directions-npz "$MED_DIRECTIONS_NPZ" --detect "$MED_DETECT" --eval results/data/misalignment_eval_medical.json --causal results/data/causal_misalign.json --layer "$LAYER" --k "$K" --min-detect-fold-margin 0.05 --require-eval-provenance --require-direction-provenance --require-detect-provenance --require-causal-provenance)" \
  --require-command-fragment="$(quote_cmd python code/verify_misalignment.py --arms)" \
  --require-command-fragment="$(quote_cmd --out "$CODE_EVAL" --gens "$CODE_GENS")" \
  --require-command-fragment="$(quote_cmd python code/direction_recover.py --base)" \
  --require-command-fragment="$(quote_cmd --misaligned-glob "$CODE_MIS_GLOB" --benign-glob "$CODE_BEN_GLOB" --layers "$LAYERS" --k "$K" --min-arms 4 --out "$CODE_DIRECTIONS_BASE")" \
  --require-command-fragment="$(quote_cmd python code/detect_holdout.py --base)" \
  --require-command-fragment="$(quote_cmd --misaligned-glob "$CODE_MIS_GLOB" --benign-glob "$CODE_BEN_GLOB" --layer "$LAYER" --tag code)" \
  --require-command-fragment="$(quote_cmd python code/causal_misalign.py --misaligned)" \
  --require-command-fragment="$(quote_cmd --dirs "$CODE_DIRECTIONS_NPZ" --layer "$LAYER" --n "$N_CAUSAL" --necessity-only --gens "$CODE_CAUSAL_GENS" --out "$CODE_CAUSAL")" \
  --require-command-fragment="$(quote_cmd python code/cross_organism.py --source-tag med --target-tag code --source-directions-npz "$MED_DIRECTIONS_NPZ" --target-directions-npz "$CODE_DIRECTIONS_NPZ")" \
  --require-command-fragment="$(quote_cmd --source-misaligned-glob "$MED_MIS_GLOB" --source-benign-glob "$MED_BEN_GLOB" --target-misaligned-glob "$CODE_MIS_GLOB" --target-benign-glob "$CODE_BEN_GLOB" --out "$CROSS_ORGANISM")" \
  --require-command-fragment="$(quote_cmd python code/check_direction_study.py --tag code --directions "$CODE_DIRECTIONS_JSON" --directions-npz "$CODE_DIRECTIONS_NPZ" --detect "$CODE_DETECT" --eval "$CODE_EVAL" --causal "$CODE_CAUSAL" --layer "$LAYER" --k "$K" --min-detect-fold-margin 0.05 --require-eval-provenance --require-direction-provenance --require-detect-provenance --require-causal-provenance)" \
  --require-command-fragment="$(quote_cmd python code/check_cross_organism.py --input "$CROSS_ORGANISM")" \
  --require-command-fragment=--require-eval-provenance \
  --require-command-fragment=--require-direction-provenance \
  --require-command-fragment=--require-detect-provenance \
  --require-command-fragment=--require-causal-provenance

echo "NOTE: launcher manifest validation is live-only; it allows untracked artifacts while the H200 job is still producing files."
echo "NOTE: final handoff requires committing result artifacts, then running python3 code/paper_completion_check.py --scope external (uses check_run_manifest.py --final-handoff)."
