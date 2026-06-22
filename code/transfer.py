"""Cross-distribution transfer: is the spectral subspace 'refusal' or
'AdvBench features'?

The refusal direction and the o_proj increment subspace are derived from one
harmful distribution (AdvBench). We then test refusal suppression under ablation
on a separately supplied harmful prompt set, recording prompt-file and
selected-set hashes in the output artifact. If the top-128 spectral ablation
still suppresses measured refusal there, the subspace is less likely to be only
an artifact of the prompts used to read the direction.

Conditions on the OOD set: baseline, ablate top-128 spectral (AdvBench-derived),
ablate random-128. Refusal rate with raw counts, per-prompt evidence, and
Wilson CIs. GPU. Writes results/data/transfer.json and
results/data/transfer_evidence.json.
"""
import os
import sys
import json
import argparse
import hashlib
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "code/transfer.py",
    "code/check_transfer_result.py",
    "code/ablation_sweep.py",
    "code/spectral.py",
    "data/harmful.json",
]


def file_sha256(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(obj, path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    tmp = path.with_suffix(path.suffix + ".tmp")
    os.makedirs(path.parent, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def repo_rel(path_text):
    path = Path(path_text)
    full = path if path.is_absolute() else ROOT / path
    try:
        return str(full.resolve().relative_to(ROOT))
    except ValueError:
        return None


def git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=ROOT, text=True).strip()
    except Exception:
        return None


def producer_metadata(command, prompt_path):
    paths = list(SOURCE_PATHS)
    prompt_rel = repo_rel(prompt_path)
    if prompt_rel and prompt_rel not in paths:
        paths.append(prompt_rel)
    status = os.environ.get("SOURCE_GIT_STATUS_SHORT")
    if status is None:
        status = git(["status", "--short", "--"] + paths) or ""
    return {
        "schema": "ood_refusal_transfer_producer_v1",
        "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT") or git(["rev-parse", "HEAD"]),
        "source_git_status_short": status,
        "command": os.environ.get("EVAL_COMMAND") or command,
        "scorer": "ood_refusal_transfer_v2_per_prompt_evidence",
        "script_sha256": {path: file_sha256(path) for path in SOURCE_PATHS},
    }


def positive_int(name, value):
    if value <= 0:
        raise SystemExit(f"--{name} must be positive")


def nonnegative_int(name, value):
    if value < 0:
        raise SystemExit(f"--{name} must be non-negative")


def load_prompt_list(path, label):
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"{label} must be a JSON list")
    if any(not isinstance(p, str) or not p.strip() for p in rows):
        raise SystemExit(f"{label} must contain only nonempty strings")
    return rows


def prompt_artifact(args, ood_all, selected):
    prompt_rel = repo_rel(args.ood_prompts)
    derivation_rel = repo_rel(args.derivation_prompts)
    if prompt_rel is None:
        raise SystemExit("--ood-prompts must point inside the repository")
    if derivation_rel is None:
        raise SystemExit("--derivation-prompts must point inside the repository")
    if prompt_rel == derivation_rel:
        raise SystemExit("OOD prompt file must not be the derivation prompt file")
    derivation = load_prompt_list(args.derivation_prompts, "derivation prompt file")
    derivation_set = set(derivation)
    selected_overlap = sorted(set(selected).intersection(derivation_set))
    if selected_overlap:
        raise SystemExit(
            "selected OOD prompts overlap the derivation prompt file "
            f"({len(selected_overlap)} exact duplicate(s))"
        )
    return {
        "path": prompt_rel,
        "sha256": file_sha256(prompt_rel),
        "n_available": len(ood_all),
        "selected_sha256": json_sha256(selected),
        "selection": {
            "method": "prefix",
            "indices": list(range(len(selected))),
        },
        "selected_prompt_sha256": [json_sha256(prompt) for prompt in selected],
        "derivation_prompt_checks": {
            "derivation_path": derivation_rel,
            "derivation_sha256": file_sha256(derivation_rel),
            "n_derivation": len(derivation),
            "same_file": prompt_rel == derivation_rel,
            "exact_overlap_count": len(selected_overlap),
        },
    }


def matched_refusal_substrings(text, substrings):
    lowered = text.strip().lower()
    return [substr for substr in substrings if substr in lowered]


def measure_refusal_condition(
    model,
    tok,
    prompts,
    selected_indices,
    device,
    reg,
    bs,
    max_new,
    store_generations,
    helpers,
):
    chat, encode_chat, is_refusal, substrings, wilson = helpers
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    handles = reg(model) if reg else []
    refusals = 0
    evidence = []
    try:
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i:i + bs]
            chunk = [chat(tok, prompt) for prompt in batch_prompts]
            enc = encode_chat(tok, chunk).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            for j, row in enumerate(out[:, enc["input_ids"].shape[1]:]):
                generated = tok.decode(row, skip_special_tokens=True)
                matches = matched_refusal_substrings(generated, substrings)
                refused = bool(is_refusal(generated))
                refusals += int(refused)
                record = {
                    "selected_index": int(selected_indices[i + j]),
                    "prompt_sha256": json_sha256(batch_prompts[j]),
                    "generation_sha256": text_sha256(generated),
                    "generation_chars": len(generated),
                    "generation_text_stored": bool(store_generations),
                    "matched_refusal_substrings": matches,
                    "is_refusal": refused,
                }
                if store_generations:
                    record["generation_text"] = generated
                evidence.append(record)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "refusals": int(refusals),
        "n": len(prompts),
        "refusal_rate": list(wilson(refusals, len(prompts))),
    }, evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--model-id", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--base-id", default="NousResearch/Meta-Llama-3-8B")
    ap.add_argument("--instruct-id", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--ood-set", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument(
        "--ood-prompts",
        required=True,
        help="tracked JSON list of harmful OOD prompts not used to derive the refusal direction",
    )
    ap.add_argument(
        "--derivation-prompts",
        default="data/harmful.json",
        help="prompt file used to derive the refusal direction; exact selected OOD overlap is rejected",
    )
    ap.add_argument("--out", default="results/data/transfer.json")
    ap.add_argument("--evidence-out", default="results/data/transfer_evidence.json")
    ap.add_argument(
        "--store-generations",
        action="store_true",
        help="store full generated text in evidence; default stores hashes and scorer outputs only",
    )
    args = ap.parse_args()
    nonnegative_int("layer", args.layer)
    positive_int("k", args.k)
    positive_int("n-gen", args.n_gen)
    positive_int("bs", args.bs)
    positive_int("max-new", args.max_new)

    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from spectral import WeightStore  # noqa: E402
    from ablation_sweep import (  # noqa: E402
        REFUSAL_SUBSTR,
        ablation,
        chat,
        encode_chat,
        is_refusal,
        wilson,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    ood_all = load_prompt_list(args.ood_prompts, "OOD prompt file")
    if len(ood_all) < args.n_gen:
        raise SystemExit(
            "OOD prompt file must contain at least %d prompts; got %d"
            % (args.n_gen, len(ood_all))
        )
    ood = ood_all[:args.n_gen]
    selected_indices = list(range(args.n_gen))
    prompt_meta = prompt_artifact(args, ood_all, ood)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    print("model loaded", flush=True)

    # subspace derived from AdvBench layer (no OOD info used)
    bws, iws = WeightStore(args.base), WeightStore(args.instruct)
    nm = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    D = iws.get(nm).astype(np.float64) - bws.get(nm).astype(np.float64)
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    d = U.shape[0]
    rng = np.random.default_rng(args.seed)
    k = args.k

    helpers = (chat, encode_chat, is_refusal, REFUSAL_SUBSTR, wilson)
    bk, base_evidence = measure_refusal_condition(
        model, tok, ood, selected_indices, device, None, args.bs,
        args.max_new, args.store_generations, helpers,
    )
    tk, top_evidence = measure_refusal_condition(
        model, tok, ood, selected_indices, device, ablation(U[:, :k].T, device),
        args.bs, args.max_new, args.store_generations, helpers,
    )
    Rk, _ = np.linalg.qr(rng.standard_normal((d, k)))
    rk, rand_evidence = measure_refusal_condition(
        model, tok, ood, selected_indices, device, ablation(Rk.T, device),
        args.bs, args.max_new, args.store_generations, helpers,
    )

    command = " ".join(shlex.quote(arg) for arg in [sys.executable] + sys.argv)
    producer = producer_metadata(command, args.ood_prompts)
    condition_evidence = {
        "baseline": base_evidence,
        "ablate_topk_advbench_derived": top_evidence,
        "ablate_randk": rand_evidence,
    }

    res = {
        "schema": "ood_refusal_transfer_v1",
        "ood_set": args.ood_set, "n_gen": args.n_gen, "k": k,
        "layer": args.layer,
        "seed": int(args.seed),
        "model": args.model,
        "base": args.base,
        "instruct": args.instruct,
        "model_ids": {
            "model": args.model_id,
            "base": args.base_id,
            "instruct": args.instruct_id,
        },
        "prompt_artifact": prompt_meta,
        "evidence_path": args.evidence_out,
        "producer": producer,
        "eval_config": {
            "dtype": args.dtype,
            "device": device,
            "local_files_only": bool(args.local_files_only),
            "bs": int(args.bs),
            "max_new": int(args.max_new),
            "store_generations": bool(args.store_generations),
        },
        "intervention": {
            "matrix": nm,
            "basis": "top-k left singular vectors of W_instruct - W_base",
            "source_layer": int(args.layer),
            "projection": "h <- h - (h @ Q) @ Q.T",
            "applied_to": "every decoder layer residual stream",
            "random_control_seed": int(args.seed),
        },
        "claim_scope": {
            "harmful_prompt_refusal_transfer": (
                "substring refusal on the supplied OOD harmful prompt set only"
            ),
            "capability_retention": "not_measured",
            "harmless_prompt_behavior": "not_measured",
        },
        "conditions": {
            "baseline": bk,
            "ablate_topk_advbench_derived": tk,
            "ablate_randk": rk,
        },
    }
    evidence = {
        "schema": "ood_refusal_transfer_evidence_v1",
        "result_path": args.out,
        "ood_set": args.ood_set,
        "n_gen": int(args.n_gen),
        "k": int(k),
        "layer": int(args.layer),
        "seed": int(args.seed),
        "model": args.model,
        "base": args.base,
        "instruct": args.instruct,
        "model_ids": res["model_ids"],
        "prompt_artifact": prompt_meta,
        "producer": producer,
        "condition_evidence": condition_evidence,
    }
    print("baseline (OOD):", res["conditions"]["baseline"], flush=True)
    print(
        "ablate top-k (AdvBench-derived):",
        res["conditions"]["ablate_topk_advbench_derived"],
        flush=True,
    )
    print("ablate random-k:", res["conditions"]["ablate_randk"], flush=True)
    write_json_atomic(evidence, args.evidence_out)
    write_json_atomic(res, args.out)
    print("wrote", args.out, "and", args.evidence_out, flush=True)


if __name__ == "__main__":
    main()
