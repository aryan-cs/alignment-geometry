#!/usr/bin/env bash
# Watch a long-running study log and validate its manifest when it finishes.
#
# Example:
#   bash code/monitor_job.sh \
#     --log results/logs/capability_eval.log \
#     --manifest results/data/run_manifests/capability_manifest.json \
#     --validator python3 code/check_run_manifest.py --input results/data/run_manifests/capability_manifest.json --study capability_preservation --require-completed --require-clean --require-preregistration
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG=""
MANIFEST=""
INTERVAL=60
TIMEOUT=0
FAIL_RE='(Traceback \(most recent call last\)|^ERROR:|^FAIL:|CUDA out of memory|OutOfMemoryError|Killed$|No space left on device)'
DONE_RE='(=== .* DONE |validated run manifest|wrote .*/run_manifests/.*\.json)'
VALIDATOR=()

usage() {
  cat <<'EOF'
Usage:
  bash code/monitor_job.sh --log <path> [--manifest <path>] [options] [--validator <command> ...]

Options:
  --interval <seconds>       Poll interval; default 60.
  --timeout <seconds>        Stop with failure after this many seconds; default 0 means no timeout.
  --fail-regex <regex>       Override failure regex.
  --done-regex <regex>       Override completion regex.
  --validator <command> ...  Command to run when the manifest exists; must be last.

The monitor exits 0 after the validator passes, or after the done regex appears
when no validator is supplied. It exits nonzero on failure regex, timeout, or a
failing validator.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --log)
      LOG="${2:?missing value for --log}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:?missing value for --manifest}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:?missing value for --interval}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:?missing value for --timeout}"
      shift 2
      ;;
    --fail-regex)
      FAIL_RE="${2:?missing value for --fail-regex}"
      shift 2
      ;;
    --done-regex)
      DONE_RE="${2:?missing value for --done-regex}"
      shift 2
      ;;
    --validator)
      shift
      VALIDATOR=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$LOG" ]; then
  echo "ERROR: --log is required" >&2
  usage >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 1 ]; then
  echo "ERROR: --interval must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --timeout must be a non-negative integer" >&2
  exit 2
fi
if [ "${#VALIDATOR[@]}" -gt 0 ] && [ -z "$MANIFEST" ]; then
  echo "ERROR: --manifest is required when --validator is supplied" >&2
  exit 2
fi

started_epoch="$(date +%s)"
echo "monitor_job: watching $LOG"
[ -n "$MANIFEST" ] && echo "monitor_job: expecting $MANIFEST"

while true; do
  now_epoch="$(date +%s)"
  if [ "$TIMEOUT" -gt 0 ] && [ $((now_epoch - started_epoch)) -ge "$TIMEOUT" ]; then
    echo "FAIL: monitor_job timed out after ${TIMEOUT}s" >&2
    exit 124
  fi

  done_seen=0
  if [ -f "$LOG" ]; then
    if tail -n 200 "$LOG" | grep -E "$FAIL_RE" >/dev/null; then
      echo "FAIL: failure pattern found in $LOG" >&2
      tail -n 80 "$LOG" >&2
      exit 1
    fi
    if tail -n 200 "$LOG" | grep -E "$DONE_RE" >/dev/null; then
      done_seen=1
    fi
    if [ -z "$MANIFEST" ] && [ "$done_seen" = "1" ]; then
      echo "monitor_job: completion pattern found"
      exit 0
    fi
  else
    echo "monitor_job: waiting for log $LOG"
  fi

  if [ -n "$MANIFEST" ] && [ -s "$MANIFEST" ]; then
    if [ "${#VALIDATOR[@]}" -gt 0 ]; then
      echo "monitor_job: validating $MANIFEST"
      if "${VALIDATOR[@]}"; then
        echo "monitor_job: validator passed"
        exit 0
      else
        validator_status=$?
      fi
      if [ "$done_seen" = "1" ]; then
        echo "FAIL: validator failed after completion pattern; status=${validator_status}" >&2
        exit "$validator_status"
      fi
      echo "monitor_job: validator failed while job is still running; retrying"
    fi
    if [ "$done_seen" = "1" ]; then
      echo "monitor_job: manifest exists and completion pattern found"
      exit 0
    fi
  fi

  sleep "$INTERVAL"
done
