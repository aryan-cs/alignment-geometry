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


ROOT = Path(__file__).resolve().parents[1]


STALE_PHRASES = [
    "Sufficiency: add the direction",
    "necessity-without-sufficiency",
    "removes refusal at every layer",
    "Ablation sensitivity versus sufficiency",
    "sufficiency test is the mirror",
    "full sufficiency sweep",
    "energy-matched random",
    "energy matched random",
    "model-level verdict",
    "the experiments have not been run",
    "betrays misalignment",
    "directions removes refusal",
    "ablations remove the behavior",
    "spectrum is a stand-alone",
]


CORE_ARTIFACTS = [
    "docs/paper.pdf",
    "docs/proof.pdf",
    "results/data/spectral.jsonl",
    "results/data/summary.json",
    "results/data/full_spectrum.npz",
    "results/data/weight_geometry.json",
    "results/data/behavioral_capture.json",
    "results/data/capture_sweep.json",
    "results/data/ablation_sweep.json",
    "results/data/ablation_layers.json",
    "results/data/sufficiency.json",
    "results/data/transfer.json",
    "results/data/misalign_scout.json",
    "results/data/misalignment_eval_medical.json",
    "results/data/causal_misalign.json",
    "results/data/causal_misalign_llama.json",
    "results/data/causal_misalign_mistral.json",
    "results/data/directions_med.json",
    "results/data/directions_llama.json",
    "results/data/directions_mistral.json",
    "results/data/detect_med.json",
    "results/data/detect_llama.json",
    "results/data/detect_mistral.json",
    "results/data/traj_med.json",
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
        "results/data/directions_code.json",
        "results/data/directions_code.npz",
        "results/data/detect_code.json",
    ],
    "scale_14b": [
        "results/data/directions_14b.json",
        "results/data/directions_14b.npz",
        "results/data/causal_misalign_14b.json",
        "results/data/detect_14b.json",
    ],
    "baseline_bakeoff": [
        "results/data/baselines.json",
    ],
}


PENDING_VALIDATORS = {
    "cross_type_transfer": [
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
    ],
    "scale_14b": [
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
        "--causal",
        "results/data/causal_misalign_14b.json",
    ],
    "baseline_bakeoff": [
        sys.executable,
        "code/check_baselines.py",
        "--input",
        "results/data/baselines.json",
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


def add(gates, name, ok, detail):
    gates.append({"name": name, "ok": bool(ok), "detail": detail})


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
    ok = fields.get("Pages") == "19" and fields.get("Page size") == "612 x 792 pts (letter)"
    add(
        gates,
        "paper_pdf_shape",
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


def check_command(gates, name, args, timeout=120):
    code, out = run_cmd(args, timeout=timeout)
    first = out.splitlines()[0] if out else "no output"
    add(gates, name, code == 0, first)


def check_stale_phrases(gates):
    search_roots = [
        ROOT / "paper",
        ROOT / "code",
        ROOT / "README.md",
        ROOT / "PLAN.md",
        ROOT / "docs" / "proof.tex",
    ]
    hits = []
    lowered = [(p, p.lower()) for p in STALE_PHRASES]
    for root in search_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.resolve() == Path(__file__).resolve():
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
    if not path.exists():
        add(
            gates,
            "capability_preservation_validated",
            False,
            "missing results/data/capability.json",
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
    )


def check_pending_studies(gates):
    for name, paths in EXPECTED_PENDING_ARTIFACTS.items():
        if name == "capability_preservation":
            continue
        missing = [p for p in paths if not (ROOT / p).exists()]
        add(
            gates,
            f"{name}_artifacts_present",
            not missing,
            "all expected artifacts present" if not missing else "missing: " + ", ".join(missing),
        )
        validator = PENDING_VALIDATORS.get(name)
        if validator is None:
            add(gates, f"{name}_validated", False, "no committed validator for this pending study")
        elif missing:
            add(gates, f"{name}_validated", False, "artifacts missing; validator not run")
        else:
            code, out = run_cmd(validator)
            add(
                gates,
                f"{name}_validated",
                code == 0,
                out.splitlines()[0] if out else "validator produced no output",
            )


def check_git_clean_enough(gates):
    code, out = run_cmd(["git", "status", "--short"])
    if code != 0:
        add(gates, "git_status_available", False, out)
        return
    substantive = []
    ignored_prefixes = (" M code/__pycache__/", "?? code/__pycache__/")
    ignored_exact = {"?? HANDOFF.md"}
    for line in out.splitlines():
        if line in ignored_exact or line.startswith(ignored_prefixes):
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
    check_pdf_freshness(gates)
    check_referenced_figures(gates)
    check_visual_qa_receipt(gates)
    check_command(gates, "paper_numbers_valid", [sys.executable, "code/check_paper_numbers.py"])
    check_command(gates, "synthetic_bbp_valid", [sys.executable, "code/synthetic_bbp.py", "--check"])
    check_stale_phrases(gates)
    check_capability(gates)
    check_pending_studies(gates)
    check_git_clean_enough(gates)
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    gates = collect_gates()
    complete = all(g["ok"] for g in gates)
    payload = {"complete": complete, "gates": gates}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("paper completion:", "complete" if complete else "incomplete")
        for gate in gates:
            mark = "PASS" if gate["ok"] else "FAIL"
            print(f"[{mark}] {gate['name']}: {gate['detail']}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
