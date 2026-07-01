#!/usr/bin/env python3
"""Validate the machine-readable disclosure of scale-14B causal attempts."""

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "results/data/scale_14b_attempt_history.json"
SCHEMA = "scale_14b_attempt_history_v1"
RECEIPT_PATH = "results/data/run_logs/scale14b_evidence_20260701.log"
RECEIPT_SHA256 = "f46977104e11daed05e4eda5d8cc341ea87afad0aa3eb48b5ee50131c681f940"
RECEIPT_SIZE = 195318
RECEIPT_LINES = 2012
PROTOCOL_COMMIT = "f4ae64ec32c8cf65eda6b1091aa7c14e5f88b08b"
FINAL_ID = "scale14b-causal-seeded-primary-audit-v1"
SAMPLING_STRATEGY = "torch_cuda_per_condition_offset_v1"
SCALE_ARTIFACT = "results/data/causal_misalign_14b.json"
GENERATION_ARTIFACT = "results/data/causal_misalign_14b_generations.json"
RUN_MANIFEST = "results/data/run_manifests/scale_14b_manifest.json"
CONDITIONS = ("misaligned_baseline", "ablate_v", "ablate_random")

EXPECTED_RECEIPT = {
    "path": RECEIPT_PATH,
    "sha256": RECEIPT_SHA256,
    "size_bytes": RECEIPT_SIZE,
    "line_count": RECEIPT_LINES,
}

EXPECTED_PROTOCOL = {
    "source_git_commit": PROTOCOL_COMMIT,
    "outcome_mode": "negative_or_inconclusive_audit",
    "sampling_seed": 0,
    "one_shot": True,
    "retries_allowed": False,
}

EXPECTED_COUNTING = {
    "scheduler_record_count": 5,
    "completed_unseeded_causal_observation_count": 2,
    "pre_freeze_primary_causal_observation_count": 0,
    "pre_freeze_independent_causal_observation_count": 0,
}

EXPECTED_SCHEDULERS = {
    "scheduler-preexisting-pid-12283": {
        "id": "scheduler-preexisting-pid-12283",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "launch_observed": False,
        "first_observed_at": "2026-06-30T05:54:38Z",
        "last_observed_at": "2026-06-30T06:45:39Z",
        "status": "ended",
        "termination_recorded": True,
        "causal_observation_ids": [],
        "log_lines": [1, 2, 3],
    },
    "scheduler-20260630T064539Z": {
        "id": "scheduler-20260630T064539Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "launch_observed": True,
        "started_at": "2026-06-30T06:45:39Z",
        "status": "incomplete",
        "termination_recorded": False,
        "causal_observation_ids": [],
        "log_lines": [4, 6, 25, 28, 49, 50],
    },
    "scheduler-20260630T150704Z": {
        "id": "scheduler-20260630T150704Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "launch_observed": True,
        "started_at": "2026-06-30T15:07:04Z",
        "finished_at": "2026-07-01T01:15:47Z",
        "status": "exited_nonzero_after_causal_completion",
        "exit_code": 1,
        "termination_recorded": True,
        "causal_observation_ids": ["causal-unseeded-20260701T011547Z"],
        "log_lines": [51, 52, 54, 74, 1749, 1794, 1822, 1824, 1825, 1831, 1832],
    },
    "scheduler-20260701T011647Z": {
        "id": "scheduler-20260701T011647Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "launch_observed": True,
        "started_at": "2026-07-01T01:16:47Z",
        "finished_at": "2026-07-01T05:17:48Z",
        "status": "exited_nonzero_after_causal_completion",
        "exit_code": 1,
        "termination_recorded": True,
        "retry_of": "scheduler-20260630T150704Z",
        "causal_observation_ids": ["causal-unseeded-20260701T051748Z"],
        "log_lines": [1832, 1833, 1835, 1861, 1864, 1909, 1937, 1939, 1940, 1946, 1947],
    },
    "scheduler-20260701T051848Z": {
        "id": "scheduler-20260701T051848Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "launch_observed": True,
        "started_at": "2026-07-01T05:18:48Z",
        "status": "incomplete",
        "termination_recorded": False,
        "retry_of": "scheduler-20260701T011647Z",
        "causal_observation_ids": [],
        "last_receipt_line": 2012,
        "log_lines": [1947, 1948, 1950, 1976, 1979, 2008, 2009, 2012],
    },
}

EXPECTED_OBSERVATION_META = {
    "causal-unseeded-20260701T011547Z": {
        "id": "causal-unseeded-20260701T011547Z",
        "scheduler_invocation_id": "scheduler-20260630T150704Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "independent": False,
        "status": "completed",
        "necessity_only": True,
        "sampling": {"seeded": False},
        "log_lines": [1794, 1806, 1813, 1821, 1822, 1824, 1825],
    },
    "causal-unseeded-20260701T051748Z": {
        "id": "causal-unseeded-20260701T051748Z",
        "scheduler_invocation_id": "scheduler-20260701T011647Z",
        "phase": "pre_freeze",
        "analysis_role": "exploratory",
        "primary": False,
        "independent": False,
        "status": "completed",
        "necessity_only": True,
        "sampling": {"seeded": False},
        "log_lines": [1909, 1922, 1929, 1936, 1937, 1939, 1940],
    },
}

EXPECTED_HISTORICAL_COUNTS = {
    "causal-unseeded-20260701T011547Z": {
        "misaligned_baseline": (800, 741, 20),
        "ablate_v": (800, 743, 30),
        "ablate_random": (800, 745, 21),
    },
    "causal-unseeded-20260701T051748Z": {
        "misaligned_baseline": (800, 735, 21),
        "ablate_v": (800, 756, 24),
        "ablate_random": (800, 729, 31),
    },
}

FINAL_BASE = {
    "id": FINAL_ID,
    "phase": "post_freeze",
    "analysis_role": "primary",
    "primary": True,
    "independent_replication": False,
    "status": "reserved",
    "one_shot": True,
    "retries_allowed": False,
    "sampling": {
        "seeded": True,
        "base_seed": 0,
        "strategy": SAMPLING_STRATEGY,
    },
    "expected_artifacts": {
        "causal_summary": SCALE_ARTIFACT,
        "causal_generations": GENERATION_ARTIFACT,
        "run_manifest": RUN_MANIFEST,
    },
}

FINAL_COMPLETION_FIELDS = {
    "started_at",
    "completed_at",
    "source_git_commit",
    "outcome",
    "metrics",
    "artifact_sha256",
}

LOG_ANCHORS = {
    1: "[2026-06-30T05:54:38Z] service runner started",
    2: "existing finish_queue.sh process(es) active: 12283",
    3: "existing finish_queue.sh process ended",
    4: "[2026-06-30T06:45:39Z] starting finish_queue.sh in foreground",
    51: "[2026-06-30T15:07:04Z] service runner started",
    1794: "python code/causal_misalign.py",
    1822: "ABLATION SENSITIVITY: base 0.027 -> ablate_v 0.040 (random 0.028)",
    1825: "wrote results/data/causal_misalign_14b.json",
    1831: "[2026-07-01T01:15:47Z] finish_queue.sh exited rc=1",
    1833: "[2026-07-01T01:16:47Z] starting finish_queue.sh in foreground",
    1909: "python code/causal_misalign.py",
    1937: "ABLATION SENSITIVITY: base 0.029 -> ablate_v 0.032 (random 0.043)",
    1940: "wrote results/data/causal_misalign_14b.json",
    1946: "[2026-07-01T05:17:48Z] finish_queue.sh exited rc=1",
    1948: "[2026-07-01T05:18:48Z] starting finish_queue.sh in foreground",
    2012: "L12: WDSV_sv=0.1733 convergence=0.933",
}


def add(errors, context, message):
    errors.append(f"{context}: {message}")


def file_sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path_text, overrides=None):
    if overrides is not None:
        return overrides.get(path_text)
    path = ROOT / path_text
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def reject_json_constant(value):
    raise ValueError(f"non-finite JSON number {value}")


def load_json_bytes(data, context, errors):
    try:
        return json.loads(data, parse_constant=reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        add(errors, context, f"invalid JSON: {exc}")
        return None


def no_nulls(value, context, errors):
    if value is None:
        add(errors, context, "null placeholders are forbidden; omit optional fields")
    elif isinstance(value, dict):
        for key, child in value.items():
            no_nulls(child, f"{context}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            no_nulls(child, f"{context}[{index}]", errors)


def exact_json_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_metrics(metrics, context, errors, expected_counts=None, require_800=True):
    if not isinstance(metrics, dict):
        add(errors, context, "must be an object")
        return
    if set(metrics) != set(CONDITIONS):
        add(errors, context, f"must contain exactly {list(CONDITIONS)}")
    for condition in CONDITIONS:
        row = metrics.get(condition)
        row_context = f"{context}.{condition}"
        if not isinstance(row, dict):
            add(errors, row_context, "must be an object")
            continue
        if set(row) != {"n_generated", "n_scored", "n_misaligned", "rate"}:
            add(errors, row_context, "must contain n_generated, n_scored, n_misaligned, and rate")
            continue
        generated = row["n_generated"]
        scored = row["n_scored"]
        misaligned = row["n_misaligned"]
        rate = row["rate"]
        if any(type(value) is not int for value in (generated, scored, misaligned)):
            add(errors, row_context, "counts must be integers")
            continue
        if not (0 <= misaligned <= scored <= generated):
            add(errors, row_context, "counts must satisfy 0 <= n_misaligned <= n_scored <= n_generated")
            continue
        if require_800 and generated != 800:
            add(errors, row_context, "n_generated must be 800 for the frozen n=100 x 8-question protocol")
        if not finite_number(rate):
            add(errors, row_context, "rate must be a finite number")
        else:
            recomputed = misaligned / scored if scored else 0.0
            if not math.isclose(float(rate), recomputed, rel_tol=1e-12, abs_tol=1e-15):
                add(errors, row_context, f"rate {rate!r} does not equal {misaligned}/{scored}")
        if expected_counts is not None:
            expected = expected_counts.get(condition)
            if expected is None or (generated, scored, misaligned) != expected:
                add(errors, row_context, f"counts do not match receipt evidence {expected}")


def validate_scheduler_records(records, errors):
    if not isinstance(records, list):
        add(errors, "scheduler_invocations", "must be an array")
        return
    indexed = {}
    for index, record in enumerate(records):
        context = f"scheduler_invocations[{index}]"
        if not isinstance(record, dict):
            add(errors, context, "must be an object")
            continue
        if record.get("status") == "incomplete" and any(
            key in record for key in ("metrics", "causal_metrics")
        ):
            add(errors, context, "incomplete scheduler invocation must not contain causal metrics")
        record_id = record.get("id")
        if not isinstance(record_id, str):
            add(errors, context, "id must be a string")
            continue
        if record_id in indexed:
            add(errors, context, f"duplicate id {record_id}")
        indexed[record_id] = record
    if set(indexed) != set(EXPECTED_SCHEDULERS):
        add(errors, "scheduler_invocations", "IDs do not match the complete receipt-backed scheduler record set")
    for record_id, expected in EXPECTED_SCHEDULERS.items():
        actual = indexed.get(record_id)
        if actual is not None and not exact_json_equal(actual, expected):
            add(errors, f"scheduler_invocations.{record_id}", "record differs from locked receipt evidence")


def validate_historical_observations(observations, errors):
    if not isinstance(observations, list):
        add(errors, "causal_observations", "must be an array")
        return
    indexed = {}
    for index, observation in enumerate(observations):
        context = f"causal_observations[{index}]"
        if not isinstance(observation, dict):
            add(errors, context, "must be an object")
            continue
        if observation.get("status") != "completed" and "metrics" in observation:
            add(errors, context, "incomplete causal observation must not contain metrics")
        observation_id = observation.get("id")
        if not isinstance(observation_id, str):
            add(errors, context, "id must be a string")
            continue
        if observation_id in indexed:
            add(errors, context, f"duplicate id {observation_id}")
        indexed[observation_id] = observation
    if set(indexed) != set(EXPECTED_OBSERVATION_META):
        add(errors, "causal_observations", "must contain exactly the two completed unseeded observations")
    for observation_id, expected_meta in EXPECTED_OBSERVATION_META.items():
        actual = indexed.get(observation_id)
        if actual is None:
            continue
        meta = {key: value for key, value in actual.items() if key != "metrics"}
        if not exact_json_equal(meta, expected_meta):
            add(errors, f"causal_observations.{observation_id}", "metadata differs from locked receipt evidence")
        if actual.get("primary") is not False or actual.get("independent") is not False:
            add(errors, f"causal_observations.{observation_id}", "must be non-primary and non-independent")
        validate_metrics(
            actual.get("metrics"),
            f"causal_observations.{observation_id}.metrics",
            errors,
            expected_counts=EXPECTED_HISTORICAL_COUNTS[observation_id],
        )


def validate_final_provenance(summary, manifest, generation_data, final, errors):
    provenance = summary.get("provenance") if isinstance(summary, dict) else None
    if not isinstance(provenance, dict):
        add(errors, SCALE_ARTIFACT, "missing provenance object")
        return
    args = provenance.get("args")
    schedule = provenance.get("sampling_seed_schedule")
    if type(provenance.get("sampling_seed")) is not int or provenance.get("sampling_seed") != 0:
        add(errors, SCALE_ARTIFACT, "provenance sampling_seed must be 0")
    if (
        not isinstance(args, dict)
        or type(args.get("sampling_seed")) is not int
        or args.get("sampling_seed") != 0
        or args.get("necessity_only") is not True
    ):
        add(errors, SCALE_ARTIFACT, "provenance args must record sampling_seed=0 and necessity_only=true")
    expected_schedule = {
        "schema": "causal_sampling_seed_schedule_v1",
        "strategy": SAMPLING_STRATEGY,
        "base_seed": 0,
        "condition_order": list(CONDITIONS),
        "condition_seeds": {
            "misaligned_baseline": 0,
            "ablate_v": 1,
            "ablate_random": 2,
        },
    }
    if not exact_json_equal(schedule, expected_schedule):
        add(errors, SCALE_ARTIFACT, "sampling seed schedule differs from the frozen one-shot protocol")
    generation_hash = final["artifact_sha256"].get(GENERATION_ARTIFACT)
    if provenance.get("causal_generations_sha256") != generation_hash:
        add(errors, SCALE_ARTIFACT, "causal_generations_sha256 does not match disclosed artifact hash")

    if not isinstance(generation_data, dict):
        add(errors, GENERATION_ARTIFACT, "must be a JSON object")
    else:
        conditions = generation_data.get("conditions")
        if not isinstance(conditions, dict):
            add(errors, GENERATION_ARTIFACT, "missing conditions object")
        else:
            final_metrics = final.get("metrics")
            for condition in CONDITIONS:
                rows = conditions.get(condition)
                metric_row = final_metrics.get(condition) if isinstance(final_metrics, dict) else None
                expected_n = metric_row.get("n_generated") if isinstance(metric_row, dict) else None
                if type(expected_n) is not int:
                    add(errors, f"final_seeded_primary.metrics.{condition}", "missing integer n_generated")
                    continue
                if not isinstance(rows, list) or len(rows) != expected_n:
                    add(errors, f"{GENERATION_ARTIFACT}.{condition}", f"must contain {expected_n} generation rows")

    if not isinstance(manifest, dict):
        add(errors, RUN_MANIFEST, "must be a JSON object")
        return
    config = manifest.get("config")
    manifest_hashes = manifest.get("artifact_sha256")
    if manifest.get("study") != "scale_14b" or manifest.get("status") != "completed":
        add(errors, RUN_MANIFEST, "must be a completed scale_14b manifest")
    if not isinstance(config, dict) or config.get("causal_outcome_mode") != "negative_or_inconclusive_audit":
        add(errors, RUN_MANIFEST, "config must record negative_or_inconclusive_audit mode")
    if (
        not isinstance(config, dict)
        or type(config.get("causal_sampling_seed")) is not int
        or config.get("causal_sampling_seed") != 0
        or type(config.get("n_causal")) is not int
        or config.get("n_causal") != 100
    ):
        add(errors, RUN_MANIFEST, "config must record causal_sampling_seed=0 and n_causal=100")
    if not isinstance(manifest_hashes, dict):
        add(errors, RUN_MANIFEST, "missing artifact_sha256 object")
    else:
        for path in (SCALE_ARTIFACT, GENERATION_ARTIFACT):
            if manifest_hashes.get(path) != final["artifact_sha256"].get(path):
                add(errors, RUN_MANIFEST, f"artifact hash for {path} does not match disclosure")
    if manifest.get("started_at") != final.get("started_at") or manifest.get("finished_at") != final.get("completed_at"):
        add(errors, RUN_MANIFEST, "run timestamps do not match final attempt-history metadata")
    if manifest.get("source_git_commit") != final.get("source_git_commit"):
        add(errors, RUN_MANIFEST, "source_git_commit does not match final attempt-history metadata")


def validate_final(final, errors, artifact_overrides=None):
    if not isinstance(final, dict):
        add(errors, "final_seeded_primary", "must be an object")
        return
    for key, value in FINAL_BASE.items():
        if key == "status":
            continue
        if not exact_json_equal(final.get(key), value):
            add(errors, f"final_seeded_primary.{key}", "differs from the frozen primary protocol")

    scale_bytes = read_bytes(SCALE_ARTIFACT, artifact_overrides)
    if scale_bytes is None:
        allowed = set(FINAL_BASE)
        if set(final) != allowed:
            add(errors, "final_seeded_primary", "completion metadata must be omitted while the scale artifact is absent")
        if final.get("status") != "reserved":
            add(errors, "final_seeded_primary.status", "must be reserved while the scale artifact is absent")
        for field in FINAL_COMPLETION_FIELDS:
            if field in final:
                add(errors, f"final_seeded_primary.{field}", "must be omitted before the one-shot audit completes")
        return

    required_keys = set(FINAL_BASE) | FINAL_COMPLETION_FIELDS
    if set(final) != required_keys:
        add(errors, "final_seeded_primary", "completed scale artifact requires every final completion field and no extras")
    if final.get("status") != "completed":
        add(errors, "final_seeded_primary.status", "must be completed when the scale artifact exists")
    if final.get("outcome") != "negative_or_inconclusive_audit":
        add(errors, "final_seeded_primary.outcome", "must match the frozen audit outcome mode")
    started = parse_time(final.get("started_at"))
    completed = parse_time(final.get("completed_at"))
    if started is None or completed is None or completed < started:
        add(errors, "final_seeded_primary", "started_at/completed_at must be ordered timezone-aware timestamps")
    if not isinstance(final.get("source_git_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", final.get("source_git_commit", "")
    ):
        add(errors, "final_seeded_primary.source_git_commit", "must be a full lowercase git SHA")
    validate_metrics(final.get("metrics"), "final_seeded_primary.metrics", errors)

    hashes = final.get("artifact_sha256")
    expected_paths = {SCALE_ARTIFACT, GENERATION_ARTIFACT, RUN_MANIFEST}
    if not isinstance(hashes, dict) or set(hashes) != expected_paths:
        add(errors, "final_seeded_primary.artifact_sha256", "must hash exactly the causal summary, generations, and run manifest")
        return
    artifact_data = {}
    for path in sorted(expected_paths):
        data = read_bytes(path, artifact_overrides)
        artifact_data[path] = data
        digest = hashes.get(path)
        if data is None:
            add(errors, f"final_seeded_primary.artifact_sha256.{path}", "referenced artifact is missing")
        elif not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            add(errors, f"final_seeded_primary.artifact_sha256.{path}", "must be a lowercase SHA256 digest")
        elif file_sha256_bytes(data) != digest:
            add(errors, f"final_seeded_primary.artifact_sha256.{path}", "hash mismatch")
    if any(data is None for data in artifact_data.values()):
        return

    summary = load_json_bytes(artifact_data[SCALE_ARTIFACT], SCALE_ARTIFACT, errors)
    generation_data = load_json_bytes(artifact_data[GENERATION_ARTIFACT], GENERATION_ARTIFACT, errors)
    manifest = load_json_bytes(artifact_data[RUN_MANIFEST], RUN_MANIFEST, errors)
    if not isinstance(summary, dict):
        return
    necessity = summary.get("necessity")
    if not isinstance(necessity, dict):
        add(errors, SCALE_ARTIFACT, "missing necessity metrics")
    else:
        final_metrics = final.get("metrics")
        for condition in CONDITIONS:
            row = necessity.get(condition)
            disclosed = final_metrics.get(condition, {}) if isinstance(final_metrics, dict) else {}
            if not isinstance(row, dict):
                add(errors, f"{SCALE_ARTIFACT}.necessity.{condition}", "missing metric row")
                continue
            if type(row.get("n_ok")) is not int or type(row.get("n_mis")) is not int:
                add(errors, f"{SCALE_ARTIFACT}.necessity.{condition}", "n_ok and n_mis must be integers")
            elif row.get("n_ok") != disclosed.get("n_scored") or row.get("n_mis") != disclosed.get("n_misaligned"):
                add(errors, f"{SCALE_ARTIFACT}.necessity.{condition}", "counts do not match final attempt-history metrics")
            artifact_rate = row.get("rate")
            disclosed_rate = disclosed.get("rate")
            if not finite_number(artifact_rate) or not finite_number(disclosed_rate):
                add(errors, f"{SCALE_ARTIFACT}.necessity.{condition}", "rates must be finite numbers")
            elif not math.isclose(float(artifact_rate), float(disclosed_rate), rel_tol=1e-12, abs_tol=1e-15):
                add(errors, f"{SCALE_ARTIFACT}.necessity.{condition}", "rate does not match final attempt-history metrics")
    if isinstance(manifest, dict) and isinstance(generation_data, dict):
        validate_final_provenance(summary, manifest, generation_data, final, errors)


def validate_history(data, artifact_overrides=None):
    errors = []
    if not isinstance(data, dict):
        return ["root: must be an object"]
    no_nulls(data, "root", errors)
    expected_top = {
        "schema",
        "study",
        "scope",
        "receipt",
        "protocol_freeze",
        "counting",
        "scheduler_invocations",
        "causal_observations",
        "final_seeded_primary",
    }
    if set(data) != expected_top:
        add(errors, "root", f"must contain exactly {sorted(expected_top)}")
    if data.get("schema") != SCHEMA:
        add(errors, "schema", f"must be {SCHEMA}")
    if data.get("study") != "scale_14b":
        add(errors, "study", "must be scale_14b")
    if data.get("scope") != "causal_necessity_attempt_disclosure":
        add(errors, "scope", "must be causal_necessity_attempt_disclosure")
    if not exact_json_equal(data.get("receipt"), EXPECTED_RECEIPT):
        add(errors, "receipt", "metadata differs from the known raw-log receipt")
    if not exact_json_equal(data.get("protocol_freeze"), EXPECTED_PROTOCOL):
        add(errors, "protocol_freeze", "differs from the frozen seeded one-shot protocol")
    if not exact_json_equal(data.get("counting"), EXPECTED_COUNTING):
        add(errors, "counting", "does not preserve scheduler/causal counting separation")
    validate_scheduler_records(data.get("scheduler_invocations"), errors)
    validate_historical_observations(data.get("causal_observations"), errors)
    validate_final(data.get("final_seeded_primary"), errors, artifact_overrides)
    return errors


def validate_receipt_blob(blob, receipt):
    errors = []
    if len(blob) != receipt.get("size_bytes"):
        add(errors, "receipt", f"size is {len(blob)}, expected {receipt.get('size_bytes')}")
    digest = file_sha256_bytes(blob)
    if digest != receipt.get("sha256"):
        add(errors, "receipt", f"SHA256 is {digest}, expected {receipt.get('sha256')}")
    lines = blob.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) != receipt.get("line_count"):
        add(errors, "receipt", f"line count is {len(lines)}, expected {receipt.get('line_count')}")
    if receipt == EXPECTED_RECEIPT and len(lines) == RECEIPT_LINES:
        decoded = [line.decode("utf-8", errors="replace") for line in lines]
        for line_number, anchor in LOG_ANCHORS.items():
            if anchor not in decoded[line_number - 1]:
                add(errors, f"receipt.line[{line_number}]", f"missing expected anchor {anchor!r}")
        for line_number in (1794, 1909):
            if "--sampling-seed" in decoded[line_number - 1]:
                add(errors, f"receipt.line[{line_number}]", "historical causal command must be unseeded")
    return errors


def historical_metrics(observation_id):
    return {
        condition: {
            "n_generated": generated,
            "n_scored": scored,
            "n_misaligned": misaligned,
            "rate": misaligned / scored,
        }
        for condition, (generated, scored, misaligned) in EXPECTED_HISTORICAL_COUNTS[observation_id].items()
    }


def reserved_fixture():
    observations = []
    for observation_id, meta in EXPECTED_OBSERVATION_META.items():
        row = copy.deepcopy(meta)
        row["metrics"] = historical_metrics(observation_id)
        observations.append(row)
    return {
        "schema": SCHEMA,
        "study": "scale_14b",
        "scope": "causal_necessity_attempt_disclosure",
        "receipt": copy.deepcopy(EXPECTED_RECEIPT),
        "protocol_freeze": copy.deepcopy(EXPECTED_PROTOCOL),
        "counting": copy.deepcopy(EXPECTED_COUNTING),
        "scheduler_invocations": [copy.deepcopy(value) for value in EXPECTED_SCHEDULERS.values()],
        "causal_observations": observations,
        "final_seeded_primary": copy.deepcopy(FINAL_BASE),
    }


def completed_fixture():
    history = reserved_fixture()
    metrics = {
        "misaligned_baseline": {"n_generated": 800, "n_scored": 700, "n_misaligned": 21, "rate": 21 / 700},
        "ablate_v": {"n_generated": 800, "n_scored": 710, "n_misaligned": 18, "rate": 18 / 710},
        "ablate_random": {"n_generated": 800, "n_scored": 705, "n_misaligned": 24, "rate": 24 / 705},
    }
    started_at = "2026-07-01T08:00:00Z"
    completed_at = "2026-07-01T12:00:00Z"
    source_commit = "a" * 40
    generation = {
        "schema": "causal_misalign_generations_v1",
        "sampling_seed": 0,
        "conditions": {condition: [{} for _ in range(800)] for condition in CONDITIONS},
    }
    generation_bytes = (json.dumps(generation, sort_keys=True) + "\n").encode()
    generation_hash = file_sha256_bytes(generation_bytes)
    schedule = {
        "schema": "causal_sampling_seed_schedule_v1",
        "strategy": SAMPLING_STRATEGY,
        "base_seed": 0,
        "condition_order": list(CONDITIONS),
        "condition_seeds": {"misaligned_baseline": 0, "ablate_v": 1, "ablate_random": 2},
    }
    summary = {
        "necessity": {
            condition: {
                "rate": row["rate"],
                "n_mis": row["n_misaligned"],
                "n_ok": row["n_scored"],
            }
            for condition, row in metrics.items()
        },
        "provenance": {
            "started_at": started_at,
            "finished_at": completed_at,
            "sampling_seed": 0,
            "sampling_seed_schedule": schedule,
            "args": {"sampling_seed": 0, "necessity_only": True},
            "causal_generations_sha256": generation_hash,
        },
    }
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_hash = file_sha256_bytes(summary_bytes)
    manifest = {
        "schema": "study_run_manifest_v1",
        "study": "scale_14b",
        "status": "completed",
        "started_at": started_at,
        "finished_at": completed_at,
        "source_git_commit": source_commit,
        "config": {
            "causal_outcome_mode": "negative_or_inconclusive_audit",
            "causal_sampling_seed": 0,
            "n_causal": 100,
        },
        "artifact_sha256": {
            SCALE_ARTIFACT: summary_hash,
            GENERATION_ARTIFACT: generation_hash,
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    artifacts = {
        SCALE_ARTIFACT: summary_bytes,
        GENERATION_ARTIFACT: generation_bytes,
        RUN_MANIFEST: manifest_bytes,
    }
    final = copy.deepcopy(FINAL_BASE)
    final.update(
        {
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "source_git_commit": source_commit,
            "outcome": "negative_or_inconclusive_audit",
            "metrics": metrics,
            "artifact_sha256": {path: file_sha256_bytes(blob) for path, blob in artifacts.items()},
        }
    )
    history["final_seeded_primary"] = final
    return history, artifacts


def self_test():
    failures = []

    fixture = reserved_fixture()
    if validate_history(fixture, artifact_overrides={}):
        failures.append("reserved real-shaped fixture was rejected")

    bad_rate = copy.deepcopy(fixture)
    bad_rate["causal_observations"][0]["metrics"]["ablate_v"]["rate"] = 0.5
    if not any("does not equal" in error for error in validate_history(bad_rate, artifact_overrides={})):
        failures.append("rate mismatch was not rejected")

    incomplete = copy.deepcopy(fixture)
    incomplete["causal_observations"][0]["status"] = "incomplete"
    if not any("incomplete causal observation" in error for error in validate_history(incomplete, artifact_overrides={})):
        failures.append("causal metrics on an incomplete observation were not rejected")

    promoted = copy.deepcopy(fixture)
    promoted["causal_observations"][0]["primary"] = True
    if not any("non-primary and non-independent" in error for error in validate_history(promoted, artifact_overrides={})):
        failures.append("historical unseeded observation could be promoted to primary")

    type_confused = copy.deepcopy(fixture)
    type_confused["protocol_freeze"]["sampling_seed"] = False
    if not any("protocol_freeze" in error for error in validate_history(type_confused, artifact_overrides={})):
        failures.append("boolean sampling seed was accepted as integer zero")

    premature = copy.deepcopy(fixture)
    premature["final_seeded_primary"]["metrics"] = historical_metrics("causal-unseeded-20260701T011547Z")
    if not any("must be omitted" in error for error in validate_history(premature, artifact_overrides={})):
        failures.append("premature final metrics were not rejected")

    if not any("requires every final completion field" in error for error in validate_history(fixture, artifact_overrides={SCALE_ARTIFACT: b"{}"})):
        failures.append("present scale artifact did not require final metadata")

    completed, artifacts = completed_fixture()
    if validate_history(completed, artifact_overrides=artifacts):
        failures.append("completed real-shaped fixture was rejected")

    malformed_completed = copy.deepcopy(completed)
    malformed_artifacts = dict(artifacts)
    malformed_summary = json.loads(malformed_artifacts[SCALE_ARTIFACT])
    malformed_summary["necessity"]["ablate_v"]["rate"] = "not-a-rate"
    malformed_artifacts[SCALE_ARTIFACT] = (json.dumps(malformed_summary, sort_keys=True) + "\n").encode()
    malformed_completed["final_seeded_primary"]["artifact_sha256"][SCALE_ARTIFACT] = file_sha256_bytes(
        malformed_artifacts[SCALE_ARTIFACT]
    )
    malformed_errors = validate_history(malformed_completed, artifact_overrides=malformed_artifacts)
    if not any("rates must be finite numbers" in error for error in malformed_errors):
        failures.append("malformed completed-artifact rate was not rejected cleanly")

    tampered_artifacts = dict(artifacts)
    tampered_artifacts[GENERATION_ARTIFACT] += b" "
    if not any("hash mismatch" in error for error in validate_history(completed, artifact_overrides=tampered_artifacts)):
        failures.append("tampered final artifact was not rejected")

    receipt_blob = b"one\ntwo\n"
    receipt = {
        "sha256": file_sha256_bytes(receipt_blob),
        "size_bytes": len(receipt_blob),
        "line_count": 2,
    }
    if validate_receipt_blob(receipt_blob, receipt):
        failures.append("valid in-memory receipt was rejected")
    if not validate_receipt_blob(receipt_blob + b"three\n", receipt):
        failures.append("tampered in-memory receipt was accepted")

    return failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--raw-log", help="override the planned repository receipt path")
    parser.add_argument("--require-raw-log", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        failures = self_test()
        if failures:
            for failure in failures:
                print(f"ERROR: self-test: {failure}", file=sys.stderr)
            return 1
        print("scale 14B attempt-history self-test passed")
        return 0

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    try:
        input_bytes = input_path.read_bytes()
    except OSError as exc:
        print(f"ERROR: {input_path}: {exc}", file=sys.stderr)
        return 1
    load_errors = []
    data = load_json_bytes(input_bytes, str(input_path), load_errors)
    errors = load_errors + ([] if data is None else validate_history(data))

    receipt_path = Path(args.raw_log) if args.raw_log else ROOT / RECEIPT_PATH
    receipt_present = receipt_path.is_file()
    if receipt_present:
        try:
            errors.extend(validate_receipt_blob(receipt_path.read_bytes(), EXPECTED_RECEIPT))
        except OSError as exc:
            add(errors, str(receipt_path), str(exc))
    elif args.require_raw_log:
        add(errors, str(receipt_path), "required raw-log receipt is missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    suffix = f" and receipt {receipt_path}" if receipt_present else " (receipt not present)"
    print(f"validated scale 14B attempt history {input_path}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
