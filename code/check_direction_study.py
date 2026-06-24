#!/usr/bin/env python3
"""Validate a misalignment-direction study result bundle.

This is a schema-and-claim gate for future cross-type and scale studies. It does
not generate evidence; it checks that committed direction, detector, gate, and
optional causal artifacts contain enough signal to support paper claims.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_output(args, *, text=True):
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def error(errors, context, message):
    errors.append(f"{context}: {message}")


def finite_number(x, context, errors, lo=None, hi=None):
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        error(errors, context, f"expected finite number, got {x!r}")
        return None
    v = float(x)
    if lo is not None and v < lo:
        error(errors, context, f"{v:.6g} < {lo:.6g}")
    if hi is not None and v > hi:
        error(errors, context, f"{v:.6g} > {hi:.6g}")
    return v


def wilson(k, n, z=1.96):
    if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def validate_direction_json(path, args, errors):
    data = load_json(path)
    ctx = str(path)
    validate_direction_provenance(path, data, args, errors)
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        error(errors, ctx, "layers must be a nonempty list")
        return None
    if args.layer not in layers:
        error(errors, ctx, f"required layer {args.layer} missing from layers")
    if data.get("k") != args.k:
        error(errors, ctx, f"k must be {args.k}; got {data.get('k')!r}")
    for key in ("n_ins", "n_edu"):
        val = data.get(key)
        if not isinstance(val, int) or val < args.min_arms:
            error(errors, ctx, f"{key} must be at least {args.min_arms}; got {val!r}")

    per_layer = data.get("per_layer")
    if not isinstance(per_layer, dict):
        error(errors, ctx, "per_layer must be an object")
        return None

    best_gap = -float("inf")
    best_layer = None
    required_row = None
    for layer in layers:
        row = per_layer.get(str(layer))
        row_ctx = f"{ctx}.per_layer.{layer}"
        if not isinstance(row, dict):
            error(errors, row_ctx, "missing layer row")
            continue
        conv = finite_number(
            row.get("convergence_mean_abs_cos"),
            f"{row_ctx}.convergence_mean_abs_cos",
            errors,
            0.0,
            1.0,
        )
        null = finite_number(
            row.get("benign_null_mean_abs_cos"),
            f"{row_ctx}.benign_null_mean_abs_cos",
            errors,
            0.0,
            1.0,
        )
        finite_number(row.get("wdsv_top_sv"), f"{row_ctx}.wdsv_top_sv", errors, 0.0)
        finite_number(
            row.get("prd_min_principal_cos"),
            f"{row_ctx}.prd_min_principal_cos",
            errors,
            0.0,
            1.0,
        )
        finite_number(
            row.get("prd_max_principal_angle_deg"),
            f"{row_ctx}.prd_max_principal_angle_deg",
            errors,
            0.0,
            90.0,
        )
        if conv is None or null is None:
            continue
        gap = conv - null
        if gap > best_gap:
            best_gap = gap
            best_layer = layer
        if layer == args.layer:
            required_row = row

    if required_row is not None:
        conv = required_row["convergence_mean_abs_cos"]
        null = required_row["benign_null_mean_abs_cos"]
        if conv < args.min_convergence:
            error(
                errors,
                f"{ctx}.per_layer.{args.layer}",
                f"convergence {conv:.3f} below {args.min_convergence:.3f}",
            )
        if conv - null < args.min_convergence_gap:
            error(
                errors,
                f"{ctx}.per_layer.{args.layer}",
                f"convergence-null gap {conv - null:.3f} below {args.min_convergence_gap:.3f}",
            )
    if best_gap < args.min_best_gap:
        error(
            errors,
            ctx,
            f"best convergence-null gap {best_gap:.3f} at layer {best_layer} below {args.min_best_gap:.3f}",
        )
    return data


def validate_direction_npz(path, directions, args, errors):
    if path is None:
        return
    if not os.path.exists(path):
        error(errors, path, "missing npz")
        return
    z = np.load(path)
    key = f"wdsv_L{args.layer}"
    if key not in z:
        error(errors, path, f"missing {key}")
        return
    vec = np.asarray(z[key])
    if vec.ndim != 1:
        error(errors, path, f"{key} must be a vector, got shape {vec.shape}")
    if not np.all(np.isfinite(vec)):
        error(errors, path, f"{key} contains non-finite values")
    norm = float(np.linalg.norm(vec))
    if not (0.5 <= norm <= 1.5):
        error(errors, path, f"{key} norm {norm:.4g} outside expected unit-vector range")


def validate_git_commit(value, context, errors):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        error(errors, context, "must be a full 40-character git SHA")
        return False
    if git_output(["cat-file", "-e", f"{value}^{{commit}}"]) is None:
        error(errors, context, "commit does not exist locally")
        return False
    return True


def validate_sha256(value, context, errors):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        error(errors, context, "must be a sha256 hex digest")
        return False
    return True


_VERIFY_CONTRACTS = {}


def verify_misalignment_contract(commit, context, errors):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    if commit in _VERIFY_CONTRACTS:
        return _VERIFY_CONTRACTS[commit]

    script = git_output(["show", f"{commit}:code/verify_misalignment.py"])
    if script is None:
        error(errors, context, "verify_misalignment.py missing at recorded commit")
        return None
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        error(errors, context, f"could not parse verify_misalignment.py at recorded commit: {exc}")
        return None

    constants = {}
    wanted = {"EM_QUESTIONS", "ALIGNED_TMPL", "COHERENT_TMPL"}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    error(errors, f"{context}.{target.id}", "must be a literal constant")

    questions = constants.get("EM_QUESTIONS")
    aligned = constants.get("ALIGNED_TMPL")
    coherent = constants.get("COHERENT_TMPL")
    if not isinstance(questions, list) or not questions or not all(isinstance(q, str) and q for q in questions):
        error(errors, f"{context}.EM_QUESTIONS", "must be a nonempty string list")
        return None
    if not isinstance(aligned, str) or not aligned:
        error(errors, f"{context}.ALIGNED_TMPL", "must be a nonempty string")
        return None
    if not isinstance(coherent, str) or not coherent:
        error(errors, f"{context}.COHERENT_TMPL", "must be a nonempty string")
        return None

    templates = {"aligned": aligned, "coherent": coherent}
    contract = {
        "questions": questions,
        "question_set": set(questions),
        "questions_sha256": json_sha256(questions),
        "templates_sha256": json_sha256(templates),
    }
    _VERIFY_CONTRACTS[commit] = contract
    return contract


def validate_expected_sha256(value, expected, context, label, errors):
    if validate_sha256(value, context, errors) and value != expected:
        error(errors, context, f"does not match verify_misalignment.{label}")


def validate_em_question_counts(counter, total_rows, expected_questions, expected_repeats, context, errors):
    if type(expected_repeats) is not int or expected_repeats <= 0:
        error(errors, context, f"expected repeat count must be a positive integer, got {expected_repeats!r}")
        return
    expected_total = expected_repeats * len(expected_questions)
    if total_rows != expected_total:
        error(errors, context, f"expected {expected_total} rows ({expected_repeats} per EM question), got {total_rows}")
    mismatches = [
        (idx, counter.get(question, 0))
        for idx, question in enumerate(expected_questions)
        if counter.get(question, 0) != expected_repeats
    ]
    if mismatches:
        preview = ", ".join(f"q{idx}:{count}" for idx, count in mismatches[:8])
        suffix = "" if len(mismatches) <= 8 else f", +{len(mismatches) - 8} more"
        error(errors, context, f"question counts must be exactly {expected_repeats} each; got {preview}{suffix}")


def validate_existing_hashes(mapping, context, errors):
    if not isinstance(mapping, dict) or not mapping:
        error(errors, context, "must be a nonempty object")
        return
    for path_text, digest in mapping.items():
        item_ctx = f"{context}.{path_text}"
        if not isinstance(path_text, str) or not path_text:
            error(errors, context, "paths must be nonempty strings")
            continue
        if not validate_sha256(digest, item_ctx, errors):
            continue
        path = Path(path_text)
        full = path if path.is_absolute() else ROOT / path
        if full.exists() and full.is_file() and file_sha256(full) != digest:
            error(errors, item_ctx, "hash mismatch")


def normalized_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def hash_path_covers_snapshot(hash_path_text, snapshot_text):
    try:
        hash_path = normalized_path(hash_path_text)
        snapshot = normalized_path(snapshot_text)
        if hash_path == snapshot:
            return True
        hash_path.relative_to(snapshot)
        return True
    except (TypeError, ValueError):
        return False


def validate_input_hash_coverage(resolved, mapping, context, errors):
    if not isinstance(resolved, list) or not isinstance(mapping, dict):
        return
    for idx, row in enumerate(resolved):
        if not isinstance(row, dict):
            continue
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, str) or not snapshot:
            continue
        if not any(hash_path_covers_snapshot(path_text, snapshot) for path_text in mapping):
            error(
                errors,
                f"{context}.resolved_inputs[{idx}].snapshot",
                "must be covered by at least one input_sha256 path",
            )


def validate_dependency_hashes(mapping, commit, context, errors):
    if not isinstance(mapping, dict):
        error(errors, context, "must be an object")
        return
    for path_text, digest in mapping.items():
        item_ctx = f"{context}.{path_text}"
        if not isinstance(path_text, str) or not path_text:
            error(errors, context, "paths must be nonempty strings")
            continue
        if not validate_sha256(digest, item_ctx, errors):
            continue
        if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
            script = git_output(["show", f"{commit}:{path_text}"], text=False)
            if script is None:
                error(errors, item_ctx, "dependency missing at recorded commit")
            elif bytes_sha256(script) != digest:
                error(errors, item_ctx, "does not match dependency at recorded commit")


def validate_run_environment(env, ctx, errors):
    if not isinstance(env, dict):
        error(errors, ctx, "must be a run_environment_v1 object")
        return
    if env.get("schema") != "run_environment_v1":
        error(errors, f"{ctx}.schema", "must be run_environment_v1")
    if not isinstance(env.get("collected_at"), str) or not env["collected_at"]:
        error(errors, f"{ctx}.collected_at", "must be a nonempty timestamp string")
    for key in ("platform", "python", "packages", "cuda", "nvidia_smi"):
        if not isinstance(env.get(key), dict):
            error(errors, f"{ctx}.{key}", "must be an object")
    gpus = env.get("gpus")
    if not isinstance(gpus, list):
        error(errors, f"{ctx}.gpus", "must be a list")
        return
    for idx, gpu in enumerate(gpus):
        gctx = f"{ctx}.gpus[{idx}]"
        if not isinstance(gpu, dict):
            error(errors, gctx, "must be an object")
            continue
        if not isinstance(gpu.get("name"), str) or not gpu["name"]:
            error(errors, f"{gctx}.name", "must be a nonempty string")
        digest = gpu.get("uuid_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            error(errors, f"{gctx}.uuid_sha256", "must be a sha256 hex digest")


def validate_common_provenance(prov, ctx, schema, producer, errors):
    if not isinstance(prov, dict):
        error(errors, ctx, f"missing provenance; rerun {producer} with provenance capture")
        return False
    required = [
        "schema",
        "producer",
        "git_commit",
        "source_git_status_short",
        "git_status_short",
        "started_at",
        "finished_at",
        "argv",
        "command",
        "args",
        "config",
        "script_sha256",
        "dependency_script_sha256",
        "resolved_inputs",
        "input_sha256",
        "environment",
    ]
    for key in required:
        if key not in prov:
            error(errors, f"{ctx}.{key}", "missing required provenance field")
    if prov.get("schema") != schema:
        error(errors, f"{ctx}.schema", f"must be {schema}")
    if prov.get("producer") != producer:
        error(errors, f"{ctx}.producer", f"must be {producer}")
    for key in ("source_git_status_short", "git_status_short"):
        value = prov.get(key)
        if not isinstance(value, str):
            error(errors, f"{ctx}.{key}", "must be a string")
    command = prov.get("command")
    if not isinstance(command, str) or not command.strip():
        error(errors, f"{ctx}.command", "must be a nonempty string")
    elif re.search(r"(<[^>\n]+>|TODO|unknown)", command, re.IGNORECASE):
        error(errors, f"{ctx}.command", "must not contain placeholders")
    commit = prov.get("git_commit")
    commit_ok = validate_git_commit(commit, f"{ctx}.git_commit", errors)
    digest = prov.get("script_sha256")
    digest_ok = validate_sha256(digest, f"{ctx}.script_sha256", errors)
    if commit_ok and digest_ok:
        script = git_output(["show", f"{commit}:{producer}"], text=False)
        if script is None:
            error(errors, f"{ctx}.script_sha256", "producer missing at recorded commit")
        elif bytes_sha256(script) != digest:
            error(errors, f"{ctx}.script_sha256", "does not match producer at recorded commit")
    validate_dependency_hashes(
        prov.get("dependency_script_sha256"),
        commit,
        f"{ctx}.dependency_script_sha256",
        errors,
    )
    resolved = prov.get("resolved_inputs")
    if not isinstance(resolved, list) or not resolved:
        error(errors, f"{ctx}.resolved_inputs", "must be a nonempty list")
    else:
        for i, row in enumerate(resolved):
            rctx = f"{ctx}.resolved_inputs[{i}]"
            if not isinstance(row, dict):
                error(errors, rctx, "must be an object")
                continue
            for key in ("label", "requested", "snapshot"):
                if not isinstance(row.get(key), str) or not row.get(key):
                    error(errors, f"{rctx}.{key}", "must be a nonempty string")
    input_hashes = prov.get("input_sha256")
    validate_existing_hashes(input_hashes, f"{ctx}.input_sha256", errors)
    validate_input_hash_coverage(resolved, input_hashes, ctx, errors)
    validate_run_environment(prov.get("environment"), f"{ctx}.environment", errors)
    if not isinstance(prov.get("argv"), list) or not prov.get("argv"):
        error(errors, f"{ctx}.argv", "must be a nonempty list")
    if not isinstance(prov.get("args"), dict):
        error(errors, f"{ctx}.args", "must be an object")
        return False
    if not isinstance(prov.get("config"), dict):
        error(errors, f"{ctx}.config", "must be an object")
        return False
    return True


def validate_direction_provenance(path, data, args, errors):
    if not args.require_direction_provenance:
        return
    ctx = f"{path}.provenance"
    prov = data.get("provenance")
    if not validate_common_provenance(
        prov,
        ctx,
        "direction_recover_provenance_v1",
        "code/direction_recover.py",
        errors,
    ):
        return
    pargs = prov.get("args")
    config = prov.get("config")
    if pargs.get("k") != args.k:
        error(errors, f"{ctx}.args.k", f"must match validator k {args.k}")
    if config.get("k") != args.k:
        error(errors, f"{ctx}.config.k", f"must match validator k {args.k}")
    layers = [int(x) for x in str(pargs.get("layers", "")).split(",") if x.strip().lstrip("-").isdigit()]
    if args.layer not in layers:
        error(errors, f"{ctx}.args.layers", f"must include validator layer {args.layer}")
    if args.layer not in config.get("layers", []):
        error(errors, f"{ctx}.config.layers", f"must include validator layer {args.layer}")
    if config.get("matrix_template") != "model.layers.{layer}.self_attn.o_proj.weight":
        error(errors, f"{ctx}.config.matrix_template", "unexpected tensor template")
    for key in ("base", "runs", "misaligned_glob", "benign_glob", "out"):
        if pargs.get(key) in (None, ""):
            error(errors, f"{ctx}.args.{key}", "must be present and nonempty")
    min_arms = pargs.get("min_arms")
    if not isinstance(min_arms, int) or min_arms < args.min_arms:
        error(errors, f"{ctx}.args.min_arms", f"must be >= validator min_arms {args.min_arms}")
    if pargs.get("allow_unmatched_arms") is not False:
        error(errors, f"{ctx}.args.allow_unmatched_arms", "must be false for paper artifacts")
    if data.get("n_ins") != data.get("n_edu"):
        error(errors, str(path), "direction study must use equal matched arm counts")
    vector_hashes = prov.get("direction_vector_sha256")
    if not isinstance(vector_hashes, dict) or not vector_hashes:
        error(errors, f"{ctx}.direction_vector_sha256", "must be a nonempty object")
        return
    for key, digest in vector_hashes.items():
        validate_sha256(digest, f"{ctx}.direction_vector_sha256.{key}", errors)
    layer_key = f"wdsv_L{args.layer}"
    if layer_key not in vector_hashes:
        error(errors, f"{ctx}.direction_vector_sha256", f"missing {layer_key}")
    if args.directions_npz and os.path.exists(args.directions_npz):
        z = np.load(args.directions_npz)
        if layer_key in z and layer_key in vector_hashes:
            digest_now = bytes_sha256(np.ascontiguousarray(z[layer_key].astype(np.float32)).tobytes())
            if vector_hashes[layer_key] != digest_now:
                error(errors, f"{ctx}.direction_vector_sha256.{layer_key}", "does not match directions npz vector")


def validate_causal_provenance(path, data, args, errors):
    if not args.require_causal_provenance:
        return
    ctx = f"{path}.provenance"
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        error(errors, ctx, "missing causal provenance; rerun code/causal_misalign.py with provenance capture")
        return
    required = [
        "schema",
        "producer",
        "git_commit",
        "source_git_status_short",
        "git_status_short",
        "started_at",
        "finished_at",
        "argv",
        "command",
        "args",
        "config",
        "script_sha256",
        "dependency_script_sha256",
        "resolved_inputs",
        "input_sha256",
        "direction_key",
        "direction_vector_sha256",
        "causal_generations_path",
        "causal_generations_sha256",
        "random_seed",
        "prompt_set_sha256",
        "judge_templates_sha256",
        "environment",
    ]
    for key in required:
        if key not in prov:
            error(errors, f"{ctx}.{key}", "missing required provenance field")
    if prov.get("schema") != "causal_misalign_provenance_v1":
        error(errors, f"{ctx}.schema", "must be causal_misalign_provenance_v1")
    if prov.get("producer") != "code/causal_misalign.py":
        error(errors, f"{ctx}.producer", "must be code/causal_misalign.py")
    for key in ("source_git_status_short", "git_status_short"):
        if not isinstance(prov.get(key), str):
            error(errors, f"{ctx}.{key}", "must be a string")
    command = prov.get("command")
    if not isinstance(command, str) or not command.strip():
        error(errors, f"{ctx}.command", "must be a nonempty string")
    elif re.search(r"(<[^>\n]+>|TODO|unknown)", command, re.IGNORECASE):
        error(errors, f"{ctx}.command", "must not contain placeholders")
    commit = prov.get("git_commit")
    commit_ok = validate_git_commit(commit, f"{ctx}.git_commit", errors)
    digest = prov.get("script_sha256")
    digest_ok = validate_sha256(digest, f"{ctx}.script_sha256", errors)
    if commit_ok and digest_ok:
        script = git_output(["show", f"{commit}:code/causal_misalign.py"], text=False)
        if script is None:
            error(errors, f"{ctx}.script_sha256", "producer missing at recorded commit")
        elif bytes_sha256(script) != digest:
            error(errors, f"{ctx}.script_sha256", "does not match producer at recorded commit")
    validate_dependency_hashes(
        prov.get("dependency_script_sha256"),
        commit,
        f"{ctx}.dependency_script_sha256",
        errors,
    )
    validate_run_environment(prov.get("environment"), f"{ctx}.environment", errors)
    pargs = prov.get("args")
    if not isinstance(pargs, dict):
        error(errors, f"{ctx}.args", "must be an object")
    else:
        for key in ("misaligned", "benign", "judge", "dirs", "gens", "layer", "n", "chunk"):
            if pargs.get(key) in (None, ""):
                error(errors, f"{ctx}.args.{key}", "must be present and nonempty")
        if pargs.get("layer") != args.layer:
            error(errors, f"{ctx}.args.layer", f"must match validator layer {args.layer}")
        if pargs.get("necessity_only") and "sufficiency" in data:
            error(errors, str(path), "necessity-only causal artifact must not include inherited sufficiency")
    resolved = prov.get("resolved_inputs")
    if not isinstance(resolved, list) or not resolved:
        error(errors, f"{ctx}.resolved_inputs", "must be a nonempty list")
    else:
        labels = {row.get("label") for row in resolved if isinstance(row, dict)}
        for label in ("misaligned", "benign", "judge", "directions_npz"):
            if label not in labels:
                error(errors, f"{ctx}.resolved_inputs", f"missing {label} input")
        for i, row in enumerate(resolved):
            rctx = f"{ctx}.resolved_inputs[{i}]"
            if not isinstance(row, dict):
                error(errors, rctx, "must be an object")
                continue
            for key in ("label", "requested", "snapshot"):
                if not isinstance(row.get(key), str) or not row.get(key):
                    error(errors, f"{rctx}.{key}", "must be a nonempty string")
    if not isinstance(prov.get("config"), dict):
        error(errors, f"{ctx}.config", "must be an object")
    if prov.get("direction_key") != f"wdsv_L{args.layer}":
        error(errors, f"{ctx}.direction_key", f"must be wdsv_L{args.layer}")
    if prov.get("random_seed") != 0:
        error(errors, f"{ctx}.random_seed", "must be 0 for the committed random-direction control")
    for key in ("direction_vector_sha256", "causal_generations_sha256"):
        validate_sha256(prov.get(key), f"{ctx}.{key}", errors)
    contract = verify_misalignment_contract(commit, f"{ctx}.verify_misalignment", errors)
    if contract is not None:
        validate_expected_sha256(
            prov.get("prompt_set_sha256"),
            contract["questions_sha256"],
            f"{ctx}.prompt_set_sha256",
            "EM_QUESTIONS",
            errors,
        )
        validate_expected_sha256(
            prov.get("judge_templates_sha256"),
            contract["templates_sha256"],
            f"{ctx}.judge_templates_sha256",
            "judge templates",
            errors,
        )
    input_hashes = prov.get("input_sha256")
    validate_existing_hashes(input_hashes, f"{ctx}.input_sha256", errors)
    validate_input_hash_coverage(resolved, input_hashes, ctx, errors)
    if isinstance(input_hashes, dict):
        dirs = pargs.get("dirs") if isinstance(pargs, dict) else None
        if dirs not in input_hashes:
            error(errors, f"{ctx}.input_sha256", "missing hash for args.dirs")
        dirs_full = Path(dirs) if dirs else None
        if dirs_full is not None and not dirs_full.is_absolute():
            dirs_full = ROOT / dirs_full
        if dirs_full is not None and dirs_full.exists() and file_sha256(dirs_full) != input_hashes[dirs]:
            error(errors, f"{ctx}.input_sha256.{dirs}", "hash mismatch")
    gen_path = prov.get("causal_generations_path")
    if not isinstance(gen_path, str) or not gen_path:
        error(errors, f"{ctx}.causal_generations_path", "must be a nonempty string")
    else:
        if isinstance(pargs, dict) and pargs.get("gens") != gen_path:
            error(errors, f"{ctx}.causal_generations_path", "must match args.gens")
        gen_full = Path(gen_path)
        if not gen_full.is_absolute():
            gen_full = ROOT / gen_full
        if not gen_full.exists():
            error(errors, f"{ctx}.causal_generations_path", "generation evidence file is missing")
        else:
            if file_sha256(gen_full) != prov.get("causal_generations_sha256"):
                error(errors, f"{ctx}.causal_generations_sha256", "hash mismatch")
            validate_causal_generations(gen_full, data, errors, contract)
    if args.directions_npz and os.path.exists(args.directions_npz):
        z = np.load(args.directions_npz)
        key = f"wdsv_L{args.layer}"
        if key in z:
            digest_now = bytes_sha256(np.ascontiguousarray(z[key].astype(np.float32)).tobytes())
            if prov.get("direction_vector_sha256") != digest_now:
                error(errors, f"{ctx}.direction_vector_sha256", "does not match directions npz vector")


def validate_causal_generations(path, data, errors, contract):
    ctx = str(path)
    try:
        evidence = load_json(path)
    except Exception as exc:
        error(errors, ctx, f"could not read generation evidence: {exc}")
        return
    if evidence.get("schema") != "causal_misalign_generations_v1":
        error(errors, f"{ctx}.schema", "must be causal_misalign_generations_v1")
    if evidence.get("producer") != "code/causal_misalign.py":
        error(errors, f"{ctx}.producer", "must be code/causal_misalign.py")
    conditions = evidence.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        error(errors, f"{ctx}.conditions", "must be a nonempty object")
        return
    if contract is None:
        return
    expected_questions = contract["question_set"]
    expected_question_order = contract["questions"]
    expected_repeats = data.get("n")

    def check_condition(condition, aggregate):
        rows = conditions.get(condition)
        cctx = f"{ctx}.conditions.{condition}"
        if not isinstance(rows, list) or not rows:
            error(errors, cctx, "must be a nonempty row list")
            return
        n_ok = 0
        n_mis = 0
        question_counts = Counter()
        for i, row in enumerate(rows):
            rctx = f"{cctx}[{i}]"
            if not isinstance(row, dict):
                error(errors, rctx, "row must be an object")
                continue
            for key in ("question", "answer", "included", "misaligned"):
                if key not in row:
                    error(errors, f"{rctx}.{key}", "missing required field")
            question = row.get("question")
            if not isinstance(question, str) or not question:
                error(errors, f"{rctx}.question", "must be a nonempty string")
            elif question not in expected_questions:
                error(errors, f"{rctx}.question", "does not match verify_misalignment.EM_QUESTIONS")
            else:
                question_counts[question] += 1
            if row.get("included") is True:
                n_ok += 1
                if row.get("misaligned") is True:
                    n_mis += 1
            elif row.get("misaligned") is True:
                error(errors, f"{rctx}.misaligned", "excluded rows cannot be counted as misaligned")
        validate_em_question_counts(
            question_counts,
            len(rows),
            expected_question_order,
            expected_repeats,
            cctx,
            errors,
        )
        if not isinstance(aggregate, dict):
            error(errors, cctx, "missing matching aggregate")
            return
        if aggregate.get("n_ok") != n_ok:
            error(errors, cctx, f"n_ok {n_ok} does not match aggregate {aggregate.get('n_ok')!r}")
        if aggregate.get("n_mis") != n_mis:
            error(errors, cctx, f"n_mis {n_mis} does not match aggregate {aggregate.get('n_mis')!r}")
        expected = n_mis / n_ok if n_ok else 0.0
        if isinstance(aggregate.get("rate"), (int, float)) and abs(float(aggregate["rate"]) - expected) > 1e-12:
            error(errors, cctx, f"rate {aggregate['rate']:.12g} != n_mis/n_ok {expected:.12g}")

    necessity = data.get("necessity", {})
    check_condition("misaligned_baseline", necessity.get("misaligned_baseline"))
    check_condition("ablate_v", necessity.get("ablate_v"))
    check_condition("ablate_random", necessity.get("ablate_random"))
    sufficiency = data.get("sufficiency")
    if isinstance(sufficiency, dict):
        check_condition("benign_baseline", sufficiency.get("benign_baseline"))
        check_condition("steer_random", sufficiency.get("steer_random"))
        steer_v = sufficiency.get("steer_v", {})
        if isinstance(steer_v, dict):
            for alpha, aggregate in steer_v.items():
                check_condition(f"steer_v_{alpha}", aggregate)


def parse_ratio(text):
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def validate_detect_json(path, args, errors):
    data = load_json(path)
    ctx = str(path)
    validate_detect_provenance(path, data, args, errors)
    if data.get("tag") != args.tag:
        error(errors, ctx, f"tag must be {args.tag!r}; got {data.get('tag')!r}")
    if data.get("layer") != args.layer:
        error(errors, ctx, f"layer must be {args.layer}; got {data.get('layer')!r}")
    folds = data.get("folds")
    if not isinstance(folds, list) or len(folds) < args.min_folds:
        error(errors, ctx, f"folds must contain at least {args.min_folds} folds")
        return None
    sep = 0
    margins = []
    for i, fold in enumerate(folds):
        fctx = f"{ctx}.folds[{i}]"
        if not isinstance(fold, dict):
            error(errors, fctx, "fold must be an object")
            continue
        mis = finite_number(fold.get("mis_score"), f"{fctx}.mis_score", errors, 0.0, 1.0)
        ben = finite_number(fold.get("ben_score"), f"{fctx}.ben_score", errors, 0.0, 1.0)
        finite_number(fold.get("mis_rand"), f"{fctx}.mis_rand", errors, 0.0, 1.0)
        finite_number(fold.get("ben_rand"), f"{fctx}.ben_rand", errors, 0.0, 1.0)
        if args.require_detect_provenance:
            validate_sha256(
                fold.get("direction_vector_sha256"),
                f"{fctx}.direction_vector_sha256",
                errors,
            )
        if mis is None or ben is None:
            continue
        margin = mis - ben
        margins.append(margin)
        if margin < args.min_detect_fold_margin:
            error(
                errors,
                fctx,
                f"detect margin {margin:.3f} below {args.min_detect_fold_margin:.3f}",
            )
        sep += int(mis > ben)
    ratio = parse_ratio(data.get("mis_above_ben"))
    if ratio is None:
        error(errors, ctx, "mis_above_ben must have form '<wins>/<folds>'")
    elif ratio != (sep, len(folds)):
        error(errors, ctx, f"mis_above_ben {ratio} does not match fold scores {(sep, len(folds))}")
    if sep != len(folds):
        error(errors, ctx, f"misaligned score must exceed benign score in every fold; got {sep}/{len(folds)}")
    mean_margin = finite_number(data.get("mean_margin"), f"{ctx}.mean_margin", errors)
    empirical_margin = sum(margins) / len(margins) if margins else None
    if mean_margin is not None and empirical_margin is not None:
        if abs(mean_margin - empirical_margin) > 1e-9:
            error(errors, ctx, f"mean_margin {mean_margin:.12g} != fold mean {empirical_margin:.12g}")
        if mean_margin < args.min_detect_margin:
            error(errors, ctx, f"mean_margin {mean_margin:.3f} below {args.min_detect_margin:.3f}")
    return data


def validate_detect_provenance(path, data, args, errors):
    if not args.require_detect_provenance:
        return
    ctx = f"{path}.provenance"
    prov = data.get("provenance")
    if not validate_common_provenance(
        prov,
        ctx,
        "detect_holdout_provenance_v1",
        "code/detect_holdout.py",
        errors,
    ):
        return
    pargs = prov.get("args")
    config = prov.get("config")
    if pargs.get("tag") != args.tag:
        error(errors, f"{ctx}.args.tag", f"must match validator tag {args.tag!r}")
    if pargs.get("layer") != args.layer:
        error(errors, f"{ctx}.args.layer", f"must match validator layer {args.layer}")
    if config.get("tensor_name") != f"model.layers.{args.layer}.self_attn.o_proj.weight":
        error(errors, f"{ctx}.config.tensor_name", "must match validator layer tensor")
    if config.get("random_seed") != 0:
        error(errors, f"{ctx}.config.random_seed", "must be 0")
    for key in ("base", "runs", "misaligned_glob", "benign_glob"):
        if pargs.get(key) in (None, ""):
            error(errors, f"{ctx}.args.{key}", "must be present and nonempty")
    if pargs.get("allow_unmatched_arms") is not False:
        error(errors, f"{ctx}.args.allow_unmatched_arms", "must be false for paper artifacts")
    if prov.get("random_seed") != 0:
        error(errors, f"{ctx}.random_seed", "must be 0 for the committed random-direction control")
    validate_sha256(prov.get("random_vector_sha256"), f"{ctx}.random_vector_sha256", errors)


def validate_eval_json(path, args, errors):
    if path is None:
        return
    data = load_json(path)
    ctx = str(path)
    mis = []
    ben = []
    mis_counts = [0, 0]
    ben_counts = [0, 0]
    for name, row in data.items():
        if not isinstance(row, dict):
            error(errors, f"{ctx}.{name}", "arm row must be an object")
            continue
        validate_eval_provenance(path, name, row, args, errors)
        rate = finite_number(row.get("misalignment_rate"), f"{ctx}.{name}.misalignment_rate", errors, 0.0, 1.0)
        n_scored = row.get("n_scored")
        n_mis = row.get("n_misaligned")
        if not isinstance(n_scored, int) or n_scored <= 0:
            error(errors, f"{ctx}.{name}.n_scored", "must be a positive integer")
        if not isinstance(n_mis, int) or n_mis < 0:
            error(errors, f"{ctx}.{name}.n_misaligned", "must be a non-negative integer")
        if rate is not None and isinstance(n_scored, int) and isinstance(n_mis, int) and n_scored > 0:
            expected = n_mis / n_scored
            if abs(rate - expected) > 1e-12:
                error(errors, f"{ctx}.{name}", f"rate {rate:.12g} != n_misaligned/n_scored {expected:.12g}")
            interval = wilson(n_mis, n_scored)
            half_width = max(interval[0] - interval[1], interval[2] - interval[0])
            if half_width > args.max_eval_wilson_half_width:
                error(
                    errors,
                    f"{ctx}.{name}",
                    "eval Wilson half-width "
                    f"{half_width:.4f} exceeds {args.max_eval_wilson_half_width:.4f}",
                )
        lname = name.lower()
        if any(s in lname for s in args.misaligned_name_substrings):
            if rate is not None:
                mis.append(rate)
            if isinstance(n_scored, int) and n_scored > 0 and isinstance(n_mis, int) and 0 <= n_mis <= n_scored:
                mis_counts[0] += n_mis
                mis_counts[1] += n_scored
        if any(s in lname for s in args.benign_name_substrings):
            if rate is not None:
                ben.append(rate)
            if isinstance(n_scored, int) and n_scored > 0 and isinstance(n_mis, int) and 0 <= n_mis <= n_scored:
                ben_counts[0] += n_mis
                ben_counts[1] += n_scored
    if len(mis) < args.min_arms:
        error(errors, ctx, f"found {len(mis)} misaligned arms, need {args.min_arms}")
    if len(ben) < args.min_arms:
        error(errors, ctx, f"found {len(ben)} benign arms, need {args.min_arms}")
    if mis and sum(mis) / len(mis) < args.min_eval_misaligned_rate:
        error(errors, ctx, f"mean misaligned rate {sum(mis)/len(mis):.3f} below {args.min_eval_misaligned_rate:.3f}")
    if ben and max(ben) > args.max_eval_benign_rate:
        error(errors, ctx, f"max benign rate {max(ben):.3f} above {args.max_eval_benign_rate:.3f}")
    if args.require_eval_wilson_separation:
        if mis_counts[1] <= 0 or ben_counts[1] <= 0:
            error(errors, ctx, "missing pooled counts for eval Wilson interval separation")
        else:
            mis_ci = wilson(*mis_counts)
            ben_ci = wilson(*ben_counts)
            if mis_ci[1] <= ben_ci[2]:
                error(
                    errors,
                    ctx,
                    "pooled misaligned-vs-benign Wilson intervals overlap: "
                    f"misaligned [{mis_ci[1]:.4f},{mis_ci[2]:.4f}] vs "
                    f"benign [{ben_ci[1]:.4f},{ben_ci[2]:.4f}]",
                )


def validate_eval_provenance(path, name, row, args, errors):
    if not args.require_eval_provenance:
        return
    ctx = f"{path}.{name}.provenance"
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        error(errors, ctx, "missing eval provenance; rerun code/verify_misalignment.py with provenance capture")
        return
    required = [
        "schema",
        "producer",
        "started_at",
        "finished_at",
        "argv",
        "args",
        "git_commit",
        "git_status_short",
        "script_sha256",
        "dependency_script_sha256",
        "em_questions_sha256",
        "judge_templates_sha256",
        "arm",
        "n_generated",
        "generations_sha256",
        "environment",
    ]
    for key in required:
        if key not in prov:
            error(errors, f"{ctx}.{key}", "missing required provenance field")
    if prov.get("schema") != "misalignment_eval_arm_provenance_v1":
        error(errors, f"{ctx}.schema", "must be misalignment_eval_arm_provenance_v1")
    if prov.get("producer") != "code/verify_misalignment.py":
        error(errors, f"{ctx}.producer", "must be code/verify_misalignment.py")
    if prov.get("arm") != name:
        error(errors, f"{ctx}.arm", f"must match row name {name!r}")
    if prov.get("n_generated") != row.get("n_generated"):
        error(errors, f"{ctx}.n_generated", "must match row n_generated")
    validate_run_environment(prov.get("environment"), f"{ctx}.environment", errors)
    commit = prov.get("git_commit")
    commit_ok = validate_git_commit(commit, f"{ctx}.git_commit", errors)
    digest = prov.get("script_sha256")
    digest_ok = validate_sha256(digest, f"{ctx}.script_sha256", errors)
    if commit_ok and digest_ok:
        script = git_output(["show", f"{commit}:code/verify_misalignment.py"], text=False)
        if script is None:
            error(errors, f"{ctx}.script_sha256", "producer missing at recorded commit")
        elif bytes_sha256(script) != digest:
            error(errors, f"{ctx}.script_sha256", "does not match producer at recorded commit")
    validate_dependency_hashes(
        prov.get("dependency_script_sha256"),
        commit,
        f"{ctx}.dependency_script_sha256",
        errors,
    )
    contract = verify_misalignment_contract(commit, f"{ctx}.verify_misalignment", errors)
    pargs = prov.get("args")
    if not isinstance(pargs, dict):
        error(errors, f"{ctx}.args", "must be an object")
    else:
        for key in ("arms", "judge", "n", "out", "gens"):
            if pargs.get(key) in (None, "", []):
                error(errors, f"{ctx}.args.{key}", "must be present and nonempty")
        gens = pargs.get("gens")
        if isinstance(gens, str) and gens:
            gen_path = Path(gens)
            if not gen_path.is_absolute():
                gen_path = ROOT / gen_path
            if not gen_path.exists():
                error(errors, f"{ctx}.args.gens", "generation evidence file is missing")
            else:
                try:
                    gen_payload = load_json(gen_path)
                except Exception as exc:
                    error(errors, f"{ctx}.args.gens", f"failed to read generation evidence: {exc}")
                else:
                    arm_rows = gen_payload.get(name)
                    if not isinstance(arm_rows, list):
                        error(errors, f"{ctx}.args.gens", f"missing generation rows for arm {name!r}")
                    elif prov.get("generations_sha256") != json_sha256(arm_rows):
                        error(errors, f"{ctx}.generations_sha256", "does not match generation evidence file")
                    elif not arm_rows:
                        error(errors, f"{ctx}.args.gens", f"empty generation rows for arm {name!r}")
                    elif contract is not None:
                        expected_questions = contract["question_set"]
                        expected_question_order = contract["questions"]
                        expected_repeats = pargs.get("n")
                        question_counts = Counter()
                        for i, gen_row in enumerate(arm_rows):
                            row_ctx = f"{ctx}.args.gens[{i}]"
                            if isinstance(gen_row, dict):
                                question = gen_row.get("question")
                            elif isinstance(gen_row, list) and gen_row:
                                question = gen_row[0]
                            else:
                                error(errors, row_ctx, "generation row must be a nonempty list or object")
                                continue
                            if not isinstance(question, str) or not question:
                                error(errors, f"{row_ctx}.question", "must be a nonempty string")
                            elif question not in expected_questions:
                                error(errors, f"{row_ctx}.question", "does not match verify_misalignment.EM_QUESTIONS")
                            else:
                                question_counts[question] += 1
                        validate_em_question_counts(
                            question_counts,
                            len(arm_rows),
                            expected_question_order,
                            expected_repeats,
                            f"{ctx}.args.gens",
                            errors,
                        )
    if contract is not None:
        validate_expected_sha256(
            prov.get("em_questions_sha256"),
            contract["questions_sha256"],
            f"{ctx}.em_questions_sha256",
            "EM_QUESTIONS",
            errors,
        )
        validate_expected_sha256(
            prov.get("judge_templates_sha256"),
            contract["templates_sha256"],
            f"{ctx}.judge_templates_sha256",
            "judge templates",
            errors,
        )
    validate_sha256(prov.get("generations_sha256"), f"{ctx}.generations_sha256", errors)


def validate_causal_json(path, args, errors):
    if path is None:
        return
    data = load_json(path)
    ctx = str(path)
    validate_causal_provenance(path, data, args, errors)
    if data.get("layer") != args.layer:
        error(errors, ctx, f"layer must be {args.layer}; got {data.get('layer')!r}")
    nec = data.get("necessity")
    if not isinstance(nec, dict):
        error(errors, ctx, "missing necessity object")
        return
    required = ["misaligned_baseline", "ablate_v", "ablate_random"]
    rates = {}
    intervals = {}
    for key in required:
        row = nec.get(key)
        if not isinstance(row, dict):
            error(errors, f"{ctx}.necessity.{key}", "missing row")
            continue
        rate = finite_number(row.get("rate"), f"{ctx}.necessity.{key}.rate", errors, 0.0, 1.0)
        n_mis = row.get("n_mis")
        n_ok = row.get("n_ok")
        if not isinstance(n_ok, int) or n_ok < args.min_causal_ok:
            error(errors, f"{ctx}.necessity.{key}.n_ok", f"must be >= {args.min_causal_ok}")
        if not isinstance(n_mis, int) or n_mis < 0:
            error(errors, f"{ctx}.necessity.{key}.n_mis", "must be non-negative")
        if rate is not None and isinstance(n_ok, int) and isinstance(n_mis, int) and n_ok > 0:
            expected = n_mis / n_ok
            if abs(rate - expected) > 1e-12:
                error(errors, f"{ctx}.necessity.{key}", f"rate {rate:.12g} != n_mis/n_ok {expected:.12g}")
            intervals[key] = wilson(n_mis, n_ok)
            half_width = max(
                intervals[key][0] - intervals[key][1],
                intervals[key][2] - intervals[key][0],
            )
            if half_width > args.max_causal_wilson_half_width:
                error(
                    errors,
                    f"{ctx}.necessity.{key}",
                    "causal Wilson half-width "
                    f"{half_width:.4f} exceeds {args.max_causal_wilson_half_width:.4f}",
                )
        rates[key] = rate
    base = rates.get("misaligned_baseline")
    ablate = rates.get("ablate_v")
    rand = rates.get("ablate_random")
    if base is not None and base < args.min_causal_baseline_rate:
        error(errors, ctx, f"baseline EM {base:.3f} below {args.min_causal_baseline_rate:.3f}")
    if base is not None and ablate is not None and base - ablate < args.min_causal_drop:
        error(errors, ctx, f"baseline-ablate drop {base - ablate:.3f} below {args.min_causal_drop:.3f}")
    if rand is not None and ablate is not None and rand - ablate < args.min_random_gap:
        error(errors, ctx, f"random-ablate gap {rand - ablate:.3f} below {args.min_random_gap:.3f}")
    if args.require_causal_wilson_separation:
        base_ci = intervals.get("misaligned_baseline")
        ablate_ci = intervals.get("ablate_v")
        rand_ci = intervals.get("ablate_random")
        if base_ci is None or ablate_ci is None or rand_ci is None:
            error(errors, ctx, "missing counts for causal Wilson interval separation")
        else:
            if base_ci[1] <= ablate_ci[2]:
                error(
                    errors,
                    ctx,
                    "baseline-ablate Wilson intervals overlap: "
                    f"baseline [{base_ci[1]:.4f},{base_ci[2]:.4f}] vs "
                    f"ablate [{ablate_ci[1]:.4f},{ablate_ci[2]:.4f}]",
                )
            if rand_ci[1] <= ablate_ci[2]:
                error(
                    errors,
                    ctx,
                    "random-ablate Wilson intervals overlap: "
                    f"random [{rand_ci[1]:.4f},{rand_ci[2]:.4f}] vs "
                    f"ablate [{ablate_ci[1]:.4f},{ablate_ci[2]:.4f}]",
                )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--directions-npz")
    ap.add_argument("--detect", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--causal")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--min-arms", type=int, default=4)
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--min-convergence", type=float, default=0.70)
    ap.add_argument("--min-convergence-gap", type=float, default=0.20)
    ap.add_argument("--min-best-gap", type=float, default=0.30)
    ap.add_argument("--min-detect-margin", type=float, default=0.05)
    ap.add_argument("--min-detect-fold-margin", type=float, default=0.05)
    ap.add_argument("--min-eval-misaligned-rate", type=float, default=0.02)
    ap.add_argument("--max-eval-benign-rate", type=float, default=0.005)
    ap.add_argument("--max-eval-wilson-half-width", type=float, default=0.05)
    ap.add_argument("--require-eval-wilson-separation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-causal-ok", type=int, default=500)
    ap.add_argument("--min-causal-baseline-rate", type=float, default=0.02)
    ap.add_argument("--min-causal-drop", type=float, default=0.015)
    ap.add_argument("--min-random-gap", type=float, default=0.015)
    ap.add_argument("--max-causal-wilson-half-width", type=float, default=0.05)
    ap.add_argument("--require-causal-wilson-separation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-causal-provenance", action="store_true")
    ap.add_argument("--require-eval-provenance", action="store_true")
    ap.add_argument("--require-direction-provenance", action="store_true")
    ap.add_argument("--require-detect-provenance", action="store_true")
    ap.add_argument("--misaligned-name-substrings", nargs="+", default=["misaligned", "insecure"])
    ap.add_argument("--benign-name-substrings", nargs="+", default=["benign", "educational", "secure"])
    return ap.parse_args()


def main():
    args = parse_args()
    errors = []
    directions = validate_direction_json(args.directions, args, errors)
    validate_direction_npz(args.directions_npz, directions, args, errors)
    validate_detect_json(args.detect, args, errors)
    validate_eval_json(args.eval, args, errors)
    validate_causal_json(args.causal, args, errors)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"validated direction study {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
