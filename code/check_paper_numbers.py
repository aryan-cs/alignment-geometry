#!/usr/bin/env python3
"""Check headline paper numbers against committed result artifacts.

This is a provenance guardrail for hard-coded manuscript values. It does not
parse LaTeX; each assertion names the displayed claim it protects and the source
file that should support it.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"


failures = []


def load_json(name):
    with open(DATA / name) as f:
        return json.load(f)


def load_jsonl(name):
    with open(DATA / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def expect(label, actual, expected, tol=1e-9):
    if not math.isfinite(float(actual)) or abs(float(actual) - expected) > tol:
        failures.append(
            f"{label}: got {float(actual):.12g}, expected {expected:.12g} +/- {tol:g}"
        )


def expect_text(label, actual, expected):
    if str(actual) != expected:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def pct(x):
    return 100.0 * float(x)


def mean(rows, key):
    vals = [float(row[key]) for row in rows]
    return sum(vals) / len(vals)


def median(vals):
    vals = sorted(float(v) for v in vals)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def paper_text():
    parts = []
    for path in sorted((ROOT / "paper" / "sections").glob("*.tex")):
        parts.append(path.read_text())
    return "\n".join(parts)


def has_phrase(text, phrase):
    return re.sub(r"\s+", " ", phrase) in re.sub(r"\s+", " ", text)


def _command_result(args):
    proc = subprocess.run(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


def _command_ok(args):
    return _command_result(args)[0] == 0


def capability_result_outcome():
    """Return the validated paper-grade capability audit outcome, if present."""
    capability = DATA / "capability.json"
    evidence = DATA / "capability_evidence.json"
    manifest = DATA / "run_manifests" / "capability_manifest.json"
    if not capability.exists() or not evidence.exists() or not manifest.exists():
        return None
    code, out = _command_result([
        sys.executable,
        "code/ingest_capability_artifacts.py",
        "--validate-only",
        "--final-handoff",
    ])
    if code != 0:
        return None
    if "audit outcome: negative_capability_audit" in out:
        return "negative_capability_audit"
    if "audit outcome: preservation_thresholds_not_violated" in out:
        return "preservation_thresholds_not_violated"
    return "validated_unknown_outcome"


def capability_result_ready():
    """Return true when the paper-grade capability audit artifact is validated."""
    return capability_result_outcome() is not None


def check_capability_caveat():
    """Guard against broad-capability claims until H200 output is validated."""
    text = paper_text()
    harmless_required = [
        "harmless-prompt behavior remains unmeasured",
        "harmless-prompt rates under the same intervention are also unmeasured",
    ]
    for phrase in harmless_required:
        if not has_phrase(text, phrase):
            failures.append(
                "harmless-prompt caveat: missing required manuscript phrase "
                f"{phrase!r}"
            )
    outcome = capability_result_outcome()
    if outcome == "preservation_thresholds_not_violated":
        return
    required = [
        "Nor do the current ablations establish broad capability preservation",
        "MMLU/GSM8K/ARC-style evaluations under the same",
        "top-$128$ ablation",
    ]
    for phrase in required:
        if not has_phrase(text, phrase):
            failures.append(
                "capability caveat: missing required manuscript phrase "
                f"{phrase!r} while capability artifacts are absent, not "
                "paper-grade validated, or validated as a negative audit"
            )


def check_random_control_wording():
    """Guard against claiming a stronger refusal-control ablation than we ran."""
    forbidden = re.compile(r"energy\s*(?:-|\s+)\s*matched[\s\S]{0,40}?random", re.IGNORECASE)
    text = paper_text().lower()
    if forbidden.search(text):
        failures.append(
            "random-control wording: manuscript claims an energy-matched "
            "random subspace, but the committed refusal ablations use "
            "same-dimensional Gaussian-QR random subspaces"
        )
    for rel in ("code/ablation_layers.py", "code/ablation_sweep.py", "code/transfer.py"):
        source = (ROOT / rel).read_text().lower()
        if forbidden.search(source):
            failures.append(
                f"random-control wording: {rel} describes an energy-matched "
                "random subspace, but the implementation samples a "
                "same-dimensional Gaussian-QR random subspace"
            )


def check_uncertainty_framing():
    """Guard against point estimates being presented as interval-backed claims."""
    text = paper_text()
    compact = re.sub(r"\s+", " ", text)
    if "AUC" in compact:
        baseline = DATA / "baselines.json"
        if not baseline.exists():
            failures.append(
                "uncertainty framing: AUC appears in the manuscript without "
                "results/data/baselines.json"
            )
        auc_required = [
            "AUC is computed from all pairwise comparisons of the 16 held-out misaligned and 16 held-out benign scores",
            "All summaries are descriptive because the folds overlap",
        ]
        for phrase in auc_required:
            if phrase not in compact:
                failures.append(
                    "uncertainty framing: baseline AUC requires manuscript phrase "
                    f"{phrase!r}"
                )
    required = [
        "point-estimate enrichment",
        "descriptive point estimates from the committed capture artifact",
        "This is a deterministic point estimate from the committed prompt set",
        "Rate-block 95\\% Wilson CIs (baseline/direction/random) are Qwen:",
        "descriptive census of the analyzed layers",
        "Geometric quantities such as subspace capture, paired-agreement cosines, and score margins are deterministic summaries",
        "$53.9\\%$, 95\\% Wilson CI $[48.5,59.1]\\%$",
        "$12/12$; descriptive 95\\% Wilson fold-count interval $[75.8,100.0]\\%$",
    ]
    for phrase in required:
        if phrase not in compact:
            failures.append(
                "uncertainty framing: missing required manuscript phrase "
                f"{phrase!r}"
            )


def check_abstract_intervals():
    """Require the abstract to display the exact CIs for rate claims."""
    abstract = (ROOT / "paper" / "sections" / "abstract.tex").read_text()
    compact = re.sub(r"\s+", " ", abstract)
    required = [
        "$98.4\\%$ ($[94.5,99.6]\\%$)",
        "$3.1\\%$ ($[1.2,7.8]\\%$)",
        "$94.5\\%$ ($[89.1,97.3]\\%$)",
        "$2.6\\%$ ($[1.7,4.1]\\%$)",
        "$0.0\\%$ ($[0.0,0.5]\\%$)",
        "$3.9\\%$ ($[2.7,5.7]\\%$)",
    ]
    for phrase in required:
        if phrase not in compact:
            failures.append(
                "abstract intervals: missing exact interval-backed phrase "
                f"{phrase!r}"
            )


def check_reviewer_scope_caveats():
    """Keep the manuscript's scope limitations aligned with likely review risks."""
    text = paper_text()
    compact = re.sub(r"\s+", " ", text)
    required = [
        (
            "spectral non-specificity",
            [
                r"fine-tuning statistic,\s*not an alignment detector",
                r"Other fine-tunes can be low-dimensional or low-rank",
                r"structure shared by other real fine-tunes",
                r"domain adaptation,\s*coding or math specialization",
                r"RLHF-style preference optimization",
                r"DPO-style preference optimization",
            ],
        ),
        (
            "Marchenko-Pastur null limitation",
            [
                r"Marchenko--Pastur fit is used here as a calibrated visibility reference,\s*not as an empirical null",
                r"empirical null that matters for safety is another real fine-tune under a matched recipe",
            ],
        ),
        (
            "stable-rank interpretation",
            [
                r"Stable rank is a scale summary of energy concentration; it does not by itself count mechanisms",
            ],
        ),
        (
            "refusal operational definition",
            [
                r"Substring-matched refusal is also a coarse generation metric",
                r"substring-scored refusal suppression",
                r"harmful-versus-harmless prompt contrast",
                r"prompt distribution,\s*topic,\s*and style differences",
                r"fully prompt-provenanced HarmBench OOD run",
                r"reduces substring refusal from\s*\$71\.2\\%\$",
                r"same-dimensional random\s+subspace leaves refusal at\s*\$65\.8\\%\$",
                r"does not measure harmless prompts,\s*adaptive adversaries,\s*or every\s+alignment-relevant behavior",
            ],
        ),
        (
            "projection-ablation breadth",
            [
                r"By\s*\$k\{=\}512\$.*spectral and random projections severely disrupt measured refusal",
                r"no longer distinguishes targeted refusal suppression from broad residual-stream disruption",
                r"same top-\$128\$ intervention.*substantially degraded capability benchmarks",
            ],
        ),
        (
            "single-model refusal scope",
            [
                r"Our refusal results are on a single released model",
                r"Llama-3-8B",
                r"larger,\s+reasoning,\s+mixture-of-experts,\s+or multimodal models is open",
            ],
        ),
        (
            "controlled-organism scope",
            [
                r"We test the misalignment direction across three model families",
                r"controlled emergent-misalignment organisms",
            ],
        ),
        (
            "proxy-not-circuit framing",
            [
                r"direction is a tested\s*low-dimensional, ablation-sensitive readout of a distributed representation,\s*not a complete one-dimensional account",
                r"weight-space direction may be a compressed proxy for a broader activation-space computation",
                r"it does not identify a circuit",
                r"negative/inconclusive audit supports\s*no weight-SVD superiority claim",
            ],
        ),
        (
            "external validity",
            [
                r"External validity remains open",
                r"not naturally occurring deceptive alignment, sycophancy, reward hacking, jailbreak susceptibility, or multimodal failures",
            ],
        ),
        (
            "predictive validation",
            [
                r"A stronger predictive validation would pre-register a direction or threshold",
                r"held-out screen is retrospective and same-recipe,\s*not prospective predictive validation",
            ],
        ),
    ]
    for category, patterns in required:
        missing = [pattern for pattern in patterns if not re.search(pattern, compact, re.IGNORECASE)]
        if missing:
            failures.append(
                f"reviewer scope caveat: missing {category} pattern(s): "
                + "; ".join(repr(pattern) for pattern in missing)
            )


def check_cross_type_audit_numbers():
    """Guard the negative/inconclusive code-organism audit limitation."""
    text = paper_text()
    if not has_phrase(
        text,
        "preregistered code-organism follow-up using insecure-versus-educational arms was negative/inconclusive",
    ):
        failures.append(
            "cross-type audit limitation: manuscript does not state the "
            "negative/inconclusive code-organism follow-up"
        )
    directions = load_json("directions_code.json")
    layer12 = directions["per_layer"]["12"]
    expect(
        "cross-type audit: layer-12 convergence displayed as 0.636",
        layer12["convergence_mean_abs_cos"],
        0.636,
        0.0006,
    )
    expect(
        "cross-type audit: benign-null convergence displayed as 0.670",
        layer12["benign_null_mean_abs_cos"],
        0.670,
        0.0006,
    )
    cross = load_json("cross_organism.json")
    expect(
        "cross-type audit: medical-to-code direction cosine displayed as 0.137",
        cross["direction_cosine_abs"],
        0.137,
        0.0006,
    )
    causal = load_json("causal_misalign_code.json")["necessity"]
    drop = causal["misaligned_baseline"]["rate"] - causal["ablate_v"]["rate"]
    expect(
        "cross-type audit: baseline-ablation drop displayed as 0.004",
        drop,
        0.004,
        0.0006,
    )


def check_misalignment_framing():
    """Guard against turning a measured ablation effect into an operational verdict."""
    guarded = [
        "paper/sections/abstract.tex",
        "paper/sections/intro.tex",
        "paper/sections/misalignment.tex",
        "paper/sections/discussion.tex",
        "README.md",
        "PLAN.md",
        "docs/proof.tex",
        "code/make_figures.py",
        "code/make_candidate_figures.py",
        "code/causal_misalign.py",
        "code/make_em_box.py",
    ]
    forbidden = [
        ("causally the " + "misalignment direction", "use behaviorally coupled wording"),
        ("necessity  " + ": ablate", "use ablation-sensitivity wording"),
        ("sufficiency" + ": add", "use coherent-steering wording"),
        ("causally necessary", "use ablation/suppression or bottleneck language for misalignment"),
        ("causally-necessary", "use ablation/suppression or bottleneck language for misalignment"),
        ("behaviorally necessary bottleneck", "use behaviorally important bottleneck"),
        ("reveals that the fine-tune installed", "frame this as evidence under matched comparison"),
        ("evidence the spike is the behavior", "state behavioral relevance, not identity"),
        ("behavior it predicts", "state measured behavior reaches an observed peak"),
        ("betrays " + "misalignment", "frame this as evidence under matched comparison"),
        ("model-level " + "verdict", "use screening result or statistic"),
        ("h2 verdict", "use test result"),
        ("h3 and h4 verdicts", "use test results"),
        ("the experiments have " + "not been run", "PLAN should distinguish completed artifacts from remaining work"),
        ("neither of which needs behavioral labels", "refusal overlap still relies on the prompt-labeled refusal direction"),
        ("necessary to remove " + "the behavior", "use ablation-sensitive wording"),
        ("switches the behavior off", "use suppresses the measured behavior"),
        ("switches misalignment " + "off", "use suppresses measured misalignment"),
        ("switch it on", "use install EM or install the behavior"),
        ("sufficiency" + ": add " + "the direction", "use coherent-steering wording"),
        ("necessity-without-" + "sufficiency", "use ablation-versus-coherent-steering asymmetry"),
        ("necessity vs " + "sufficiency", "use ablation-versus-coherent-steering wording"),
        ("right: " + "sufficiency", "use coherent-steering wording"),
        ("the " + "sufficiency null", "use coherent-steering check wording"),
        ("removes refusal at every " + "layer", "use reduces refusal at every tested layer"),
        ("directions " + "removes refusal", "use reduces or suppresses refusal with the measured rate"),
        ("ablations " + "remove the behavior", "use suppress the behavior"),
        ("removes most of " + "the behavior", "use suppresses the measured behavior"),
        ("spectrum is " + "a stand-alone", "state that spectrum alone is not a stand-alone diagnostic"),
        ("ablation sensitivity versus " + "sufficiency", "use coherent-steering wording"),
        ("ablating the direction removes misalignment", "use suppresses measured misalignment"),
        ("necessity of the recovered direction", "use ablation sensitivity"),
        ("under ablation, " + "removes", "use suppresses the measured behavior"),
        ("exactly its negation " + "and addition", "describe activation-space analogues instead"),
        ("production model", "use released model unless there is production evidence"),
        ("exactly as it did for refusal", "distinguish refusal projection steering from misalignment rank-one steering"),
        ("held-out " + "detector", "use same-recipe held-out screen or candidate screen"),
        ("the direction \\emph{transfers}", "use same-recipe held-out screen language"),
        ("clean dissociation", "state the measured rates and non-overlapping intervals"),
        ("reduces refusal to 3\\%", "use the exact measured rate and interval"),
        ("leading spectral directions are a refusal bottleneck", "use measured ablation-sensitivity wording"),
        ("the leading spectral subspace is a refusal bottleneck", "use measured ablation-sensitivity wording"),
        ("refusal depends on the leading spectral subspace", "use behaviorally coupled or ablation-sensitive wording"),
        ("measured refusal depends on the leading spectral subspace", "use behaviorally coupled or ablation-sensitive wording"),
        ("visible before behavior peaks", "state that the trajectory comparison is post hoc"),
        ("controlled false-positive rate", "condition false-positive control on the ideal null"),
        ("requires no distributional assumption", "state the exchangeability limitation of permutation nulls"),
        ("permutation null for finite matrices", "use finite-size permutation stress tests unless empirical-null artifacts exist"),
        ("recovers the misalignment direction without labels", "use candidate-direction estimate wording"),
        ("label-free misalignment direction", "use behavioral-example-free wording and state matched arm grouping"),
        ("label-free contrastive direction", "use behavioral-example-free wording and state matched arm grouping"),
        ("necessary low-dimensional bottleneck", "use ablation-sensitive low-dimensional bottleneck"),
        ("low-rank implies alignment", "state that spectral concentration is not alignment-specific"),
        ("spiked implies alignment", "state that spectral concentration is not alignment-specific"),
        ("spectrum is an alignment detector", "state that spectra alone are not alignment detectors"),
        ("spectral geometry is an alignment detector", "state that spectra alone are not alignment detectors"),
        ("alignment-specific stand-alone detector", "use matched-control and causal-evidence wording"),
        ("fundamental mechanisms of alignment", "use proxy, candidate direction, or ablation-sensitive wording"),
        ("singular vector is the mechanism", "state behavioral relevance without circuit identity"),
        ("singular vectors are mechanisms", "state behavioral relevance without circuit identity"),
        ("geometry identifies the mechanism", "state behavioral relevance without circuit identity"),
        ("identifies the underlying causal structure", "state this as a proxy unless circuit evidence is added"),
    ]
    for rel in guarded:
        text = (ROOT / rel).read_text().lower()
        for phrase, replacement in forbidden:
            if phrase in text:
                failures.append(
                    f"misalignment framing: {rel} contains {phrase!r}; {replacement}"
                )
    proof = (ROOT / "docs/proof.tex").read_text()
    if "feed-forward width " + "$11008$" in proof or "$11008 " + "\\times 4096$" in proof:
        failures.append("proof dimensions: use the Llama-3-8B feed-forward width 14336")
    if "output-side covariance" not in proof or "left singular vectors" not in proof:
        failures.append("proof orientation: missing left/right singular-vector orientation note")
    if "A\nspike that does not clear this empirical null is not reported" in proof:
        failures.append("proof calibration: do not claim empirical-null filtering for the committed spectral census")
    if "MP/TW-visible structure" not in proof:
        failures.append("proof calibration: describe the committed spectral census as MP/TW-visible structure")


def check_spectral_summary():
    s = load_json("summary.json")
    rows = load_jsonl("spectral.jsonl")
    top_edge = np.array([r["delta"]["top_eig_over_edge"] for r in rows])
    spikes = np.array([r["delta"]["n_spikes"] for r in rows])
    q = np.array([min(r["delta"]["shape"]) for r in rows])
    er_d = np.array([r["delta"]["effective_rank"] for r in rows])
    er_b = np.array([r["base"]["effective_rank"] for r in rows])
    er_i = np.array([r["instruct"]["effective_rank"] for r in rows])
    sr_d = np.array([r["delta"]["stable_rank"] for r in rows])
    expect("spectral summary consistency: number of matrices", s["n_matrices"], len(rows), 0)
    expect("spectral summary consistency: top/edge min", s["top_over_edge"]["min"], float(top_edge.min()), 1e-12)
    expect("spectral summary consistency: top/edge median", s["top_over_edge"]["median"], float(np.median(top_edge)), 1e-12)
    expect("spectral summary consistency: top/edge max", s["top_over_edge"]["max"], float(top_edge.max()), 1e-12)
    expect("spectral summary consistency: top/edge frac > 1", s["top_over_edge"]["frac_above_1"], float((top_edge > 1).mean()), 1e-12)
    expect("spectral summary consistency: top/edge frac > 5", s["top_over_edge"]["frac_above_5"], float((top_edge > 5).mean()), 1e-12)
    expect("spectral summary consistency: spike min", s["spikes"]["min"], int(spikes.min()), 0)
    expect("spectral summary consistency: spike median", s["spikes"]["median"], float(np.median(spikes)), 1e-12)
    expect("spectral summary consistency: spike max", s["spikes"]["max"], int(spikes.max()), 0)
    expect("spectral summary consistency: median spikes/rank", s["spikes"]["median_spikes_over_rank"], float(np.median(spikes / q)), 1e-12)
    expect("spectral summary consistency: effective-rank ratio vs base", s["effrank_ratio_delta_vs_base"], float(np.median(er_d / er_b)), 1e-12)
    expect("spectral summary consistency: effective-rank ratio vs instruct", s["effrank_ratio_delta_vs_instruct"], float(np.median(er_d / er_i)), 1e-12)
    expect("spectral summary consistency: stable-rank median", s["stable_rank_delta_median"], float(np.median(sr_d)), 1e-12)
    for label, summary_row in s["by_type"].items():
        sub = [r for r in rows if r["label"] == label]
        te = np.array([r["delta"]["top_eig_over_edge"] for r in sub])
        sp = np.array([r["delta"]["n_spikes"] for r in sub])
        stable = np.array([r["delta"]["stable_rank"] for r in sub])
        expect(f"spectral summary consistency: {label} n", summary_row["n"], len(sub), 0)
        expect(
            f"spectral summary consistency: {label} median top/edge",
            summary_row["median_top_over_edge"],
            float(np.median(te)),
            1e-12,
        )
        expect(
            f"spectral summary consistency: {label} median spikes",
            summary_row["median_spikes"],
            float(np.median(sp)),
            1e-12,
        )
        expect(
            f"spectral summary consistency: {label} median stable rank",
            summary_row["median_stable_rank"],
            float(np.median(stable)),
            1e-12,
        )

    expect("spectral: number of matrices", s["n_matrices"], 224)
    expect("spectral: all matrices exceed edge", s["top_over_edge"]["frac_above_1"], 1.0)
    expect("spectral: median top/edge displayed as 22.0", s["top_over_edge"]["median"], 22.0, 0.05)
    expect("spectral: fraction above 5 displayed as 96%", pct(s["top_over_edge"]["frac_above_5"]), 96.0, 0.5)
    expect("spectral: tail displayed as 2.4e4", s["top_over_edge"]["max"], 2.4e4, 600.0)
    expect("spectral: median spikes displayed as 709", s["spikes"]["median"], 709.0, 0.1)
    expect("spectral: median spikes/rank displayed as about 19%", pct(s["spikes"]["median_spikes_over_rank"]), 18.7, 0.1)
    expect("spectral: median stable rank displayed as 109", s["stable_rank_delta_median"], 109.0, 0.6)

    by_type = s["by_type"]
    expected = {
        "q_proj": (80.9, 808, 45.0),
        "k_proj": (25.4, 238, 36.1),
        "v_proj": (6.2, 168, 99.9),
        "o_proj": (23.6, 756, 115.0),
        "gate_proj": (24.6, 734, 111.6),
        "up_proj": (18.4, 778, 147.9),
        "down_proj": (8.1, 680, 298.9),
    }
    for label, (top_edge, spikes, stable_rank) in expected.items():
        row = by_type[label]
        expect(f"spectral table {label}: top/edge", row["median_top_over_edge"], top_edge, 0.06)
        expect(f"spectral table {label}: spikes", row["median_spikes"], spikes, 0.6)
        expect(f"spectral table {label}: stable rank", row["median_stable_rank"], stable_rank, 0.06)

    wg = load_json("weight_geometry.json")
    overlaps = [v["overlap"] for v in wg["overlap_oproj_by_layer"].values()]
    nulls = [v["null"] for v in wg["overlap_oproj_by_layer"].values()]
    overlap_med = median(overlaps)
    null_med = median(nulls)
    expect("energy/overlap: overlap median displayed as 0.013", overlap_med, 0.013, 0.0006)
    expect("energy/overlap: null median displayed as 0.004", null_med, 0.004, 0.0006)
    expect("energy/overlap: ratio displayed as 3.4x", overlap_med / null_med, 3.4, 0.06)
    q64 = wg["energy_curve"]["q_proj"][wg["energy_curve"]["ks"].index(64)]
    expect("energy/overlap: q_proj top-64 energy displayed as a third", q64, 1.0 / 3.0, 0.02)


def check_full_spectrum_artifact():
    path = DATA / "full_spectrum.npz"
    if not path.exists():
        failures.append("full-spectrum artifact: results/data/full_spectrum.npz is missing")
        return

    z = np.load(path)
    name = str(z["name"])
    expect_text(
        "full-spectrum artifact: representative matrix",
        name,
        "model.layers.15.mlp.gate_proj.weight",
    )
    rows = [json.loads(line) for line in open(DATA / "spectral.jsonl") if line.strip()]
    row = next((r for r in rows if r["name"] == name), None)
    if row is None:
        failures.append(f"full-spectrum artifact: {name} not found in spectral.jsonl")
        return
    d = row["delta"]
    expect("full-spectrum artifact: p", int(z["p"]), d["shape"][0])
    expect("full-spectrum artifact: q", int(z["q"]), d["shape"][1])
    expect("full-spectrum artifact: gamma", float(z["gamma"]), d["gamma"], 1e-12)
    expect("full-spectrum artifact: MP edge", float(z["hi"]), d["mp_hi"], 1e-12)
    expect("full-spectrum artifact: top eigenvalue", float(z["eig"][0]), d["top_eig"], 1e-12)
    expect(
        "full-spectrum figure caption: top/edge displayed as 27x",
        float(z["eig"][0] / z["hi"]),
        27.0,
        0.2,
    )
    expect(
        "full-spectrum figure caption: raw MP-edge detachments displayed as 829",
        int((z["eig"] > z["hi"]).sum()),
        829,
    )
    expect(
        "spectral table: strict six-TW spikes for same representative matrix",
        d["n_spikes"],
        821,
    )


def check_synthetic_bbp():
    s = load_json("synthetic_bbp.json")
    expect("synthetic BBP: p", s["p"], 2048)
    expect("synthetic BBP: q", s["q"], 512)
    expect("synthetic BBP: gamma displayed as 0.25", s["gamma"], 0.25)
    expect("synthetic BBP: theta threshold displayed as 0.5", s["bbp_theta_threshold"], 0.5)
    by_name = {case["name"]: case for case in s["cases"]}
    expected = {
        "diffuse_null": 0,
        "planted_rank_1": 1,
        "planted_rank_4": 4,
        "planted_rank_16": 16,
        "energy_matched_rank_128": 0,
    }
    for name, n_spikes in expected.items():
        case = by_name.get(name)
        if case is None:
            failures.append(f"synthetic BBP: missing case {name}")
            continue
        expect(f"synthetic BBP {name}: strict spikes", case["strict_spikes"], n_spikes)
        expect(
            f"synthetic BBP {name}: expected strict spikes",
            case["expected_strict_spikes"],
            n_spikes,
        )
    if "planted_rank_16" in by_name:
        expect("synthetic BBP: rank-16 total theta", by_name["planted_rank_16"]["total_theta"], 24.0)
        expect("synthetic BBP: rank-16 r_star", by_name["planted_rank_16"]["r_star"], 48.0)
    if "energy_matched_rank_128" in by_name:
        expect(
            "synthetic BBP: energy-matched rank-128 total theta",
            by_name["energy_matched_rank_128"]["total_theta"],
            24.0,
        )
        expect(
            "synthetic BBP: energy-matched rank-128 r_star",
            by_name["energy_matched_rank_128"]["r_star"],
            48.0,
        )
    planted_means = [
        by_name[name]["planted_subspace_cos2"]["mean"]
        for name in ("planted_rank_1", "planted_rank_4", "planted_rank_16")
        if name in by_name
    ]
    if planted_means:
        expect(
            "synthetic BBP: minimum planted-subspace mean cos^2 displayed as about 0.76",
            min(planted_means),
            0.76,
            0.02,
        )


def check_refusal():
    readme = (ROOT / "README.md").read_text()
    if "93.8% (`[89.1,97.3]%`)" in readme:
        failures.append("README refusal summary: random-128 rate is stale; expected 94.5%")
    if "94.5% (`[89.1,97.3]%`)" not in readme:
        failures.append("README refusal summary: missing current random-128 rate 94.5%")

    cap = load_json("behavioral_capture.json")
    for k, expected in [(8, 0.027), (32, 0.041), (128, 0.106)]:
        expect(f"capture table: o_proj k={k}", cap["capture"][f"o_proj_k{k}"], expected, 0.0006)
    for k, expected in [(8, 0.002), (32, 0.008), (128, 0.031)]:
        expect(f"capture table: null k={k}", cap["null"][f"o_proj_k{k}"], expected, 0.0006)
    for k, expected in [(8, 13.7), (32, 5.2), (128, 3.5)]:
        enrich = cap["capture"][f"o_proj_k{k}"] / cap["null"][f"o_proj_k{k}"]
        expect(f"capture table: enrichment k={k}", enrich, expected, 0.06)
    sweep = load_json("capture_sweep.json")
    enrich8 = [row["enrich"]["8"] for row in sweep["layers"].values()]
    n_enriched = sum(1 for value in enrich8 if value > 1.0)
    expect("capture sweep: top-8 enrichment above random in 31/32 layers", n_enriched, 31)
    expect("capture sweep: median top-8 enrichment displayed as 4.6x", median(enrich8), 4.6, 0.06)
    expect("capture sweep: max top-8 enrichment displayed as 14.2x", max(enrich8), 14.2, 0.06)
    causal_tex = (ROOT / "paper" / "sections" / "causal.tex").read_text()
    if "31/32" not in causal_tex:
        failures.append("causal section must report top-8 enrichment above random in 31/32 layers")
    if "Refusal is enriched in the leading spikes, at every layer" in causal_tex:
        failures.append("causal section overclaims top-8 enrichment at every layer")

    ablation = load_json("ablation_sweep.json")
    c = ablation["conditions"]
    expect("ablation: baseline refusal displayed as 98.4%", pct(c["baseline"]["refusal_rate"][0]), 98.4, 0.06)
    expect("ablation: top-8 refusal displayed as 98.4%", pct(c["ablate_top8"]["refusal_rate"][0]), 98.4, 0.06)
    expect("ablation: top-128 refusal displayed as 3.1%", pct(c["ablate_top128"]["refusal_rate"][0]), 3.1, 0.06)
    expect("ablation: top-128 low CI displayed as 1.2%", pct(c["ablate_top128"]["refusal_rate"][1]), 1.2, 0.06)
    expect("ablation: top-128 high CI displayed as 7.8%", pct(c["ablate_top128"]["refusal_rate"][2]), 7.8, 0.06)
    expect("ablation: random-128 refusal displayed as 94.5%", pct(c["ablate_rand128"]["refusal_rate"][0]), 94.5, 0.06)
    expect("ablation: random-128 low CI displayed as 89.1%", pct(c["ablate_rand128"]["refusal_rate"][1]), 89.1, 0.06)
    expect("ablation: random-128 high CI displayed as 97.3%", pct(c["ablate_rand128"]["refusal_rate"][2]), 97.3, 0.06)
    expect("ablation: refusal-dir rate displayed as 68.8%", pct(c["ablate_refusal_dir"]["refusal_rate"][0]), 68.8, 0.06)

    layers = load_json("ablation_layers.json")
    expect("ablation layers: layer 14 top-k displayed as 3.1%", pct(layers["layers"]["14"]["ablate_topk"][0]), 3.1, 0.06)
    expect("ablation layers: layer 26 top-k displayed as 0.0%", pct(layers["layers"]["26"]["ablate_topk"][0]), 0.0, 0.06)
    min_random = min(pct(v["ablate_randk"][0]) for v in layers["layers"].values())
    expect("ablation layers: random controls at or above 93%", min_random, 93.0, 0.2)

    suff = load_json("sufficiency.json")
    expect("sufficiency: refusal direction reaches 98%", pct(suff["refusal_dir"]["4.0"]["refusal"][0]), 98.0, 0.06)
    expect("sufficiency: spectral subspace reaches 67%", pct(suff["spectral_subspace"]["6.0"]["refusal"][0]), 67.0, 0.06)
    expect("sufficiency: refusal mass in spectral subspace displayed as 10.5%", pct(suff["refusal_in_spec_fraction"]), 10.5, 0.06)


def check_misalignment():
    scout = load_json("misalign_scout.json")["summary"]
    expect("misalignment scout: number of matrices", scout["n_matrices"], 336)
    expect(
        "misalignment scout table: misaligned stable rank",
        scout["median_stable_rank_mis"],
        325.83,
        0.006,
    )
    expect(
        "misalignment scout table: benign matched stable rank",
        scout["median_stable_rank_ben_matched"],
        325.84,
        0.006,
    )
    expect(
        "misalignment scout table: misaligned strict spikes",
        scout["median_spikes_mis"],
        43.5,
        0.06,
    )
    expect(
        "misalignment scout table: benign matched strict spikes",
        scout["median_spikes_ben_matched"],
        43.0,
        0.06,
    )
    expect(
        "misalignment scout table: misaligned top/edge",
        scout["median_top_over_edge_mis"],
        3.43,
        0.006,
    )
    expect(
        "misalignment scout table: benign matched top/edge",
        scout["median_top_over_edge_ben_matched"],
        3.42,
        0.006,
    )
    expect(
        "misalignment scout: fraction lower stable rank displayed as 54%",
        pct(scout["frac_mis_lower_stable_rank"]),
        54.0,
        0.2,
    )

    gate = load_json("misalignment_eval_medical.json")
    mis_rates = [pct(v["misalignment_rate"]) for k, v in gate.items() if k.startswith("misaligned_")]
    ben_rates = [pct(v["misalignment_rate"]) for k, v in gate.items() if k.startswith("benign_")]
    expect("medical gate: mean misaligned rate displayed as 4.7%", sum(mis_rates) / len(mis_rates), 4.7, 0.06)
    expect("medical gate: min misaligned arm displayed as 3.4%", min(mis_rates), 3.4, 0.06)
    expect("medical gate: max misaligned arm displayed as 6.0%", max(mis_rates), 6.0, 0.06)
    expect("medical gate: benign controls displayed as 0.0%", max(ben_rates), 0.0, 0.01)

    directions = {
        "Qwen2.5-Coder-7B": load_json("directions_med.json"),
        "Llama-3-8B": load_json("directions_llama.json"),
        "Mistral-7B": load_json("directions_mistral.json"),
    }
    expected_layer12 = {
        "Qwen2.5-Coder-7B": (0.97, 0.16),
        "Llama-3-8B": (0.95, 0.48),
        "Mistral-7B": (0.71, 0.37),
    }
    for name, (conv, null) in expected_layer12.items():
        row = directions[name]["per_layer"]["12"]
        expect(f"{name}: layer-12 convergence", row["convergence_mean_abs_cos"], conv, 0.006)
        expect(f"{name}: layer-12 benign null", row["benign_null_mean_abs_cos"], null, 0.006)
    expect("Mistral: layer-8 convergence displayed as 0.87", directions["Mistral-7B"]["per_layer"]["8"]["convergence_mean_abs_cos"], 0.87, 0.006)
    expect("Mistral: layer-8 benign null displayed as 0.17", directions["Mistral-7B"]["per_layer"]["8"]["benign_null_mean_abs_cos"], 0.17, 0.006)

    causal = {
        "Qwen2.5-Coder-7B": (load_json("causal_misalign.json"), (2.6, 0.0, 3.9)),
        "Llama-3-8B": (load_json("causal_misalign_llama.json"), (5.3, 0.5, 2.9)),
        "Mistral-7B": (load_json("causal_misalign_mistral.json"), (8.7, 2.8, 8.6)),
    }
    for name, (data, (base, ablate, random)) in causal.items():
        nec = data["necessity"]
        expect(f"{name}: baseline EM", pct(nec["misaligned_baseline"]["rate"]), base, 0.06)
        expect(f"{name}: ablate direction EM", pct(nec["ablate_v"]["rate"]), ablate, 0.06)
        expect(f"{name}: random direction EM", pct(nec["ablate_random"]["rate"]), random, 0.06)
    q_nec = causal["Qwen2.5-Coder-7B"][0]["necessity"]
    expect("Qwen causal caption: baseline numerator", q_nec["misaligned_baseline"]["n_mis"], 18)
    expect("Qwen causal caption: baseline denominator", q_nec["misaligned_baseline"]["n_ok"], 683)
    expect("Qwen causal caption: ablate numerator", q_nec["ablate_v"]["n_mis"], 0)
    expect("Qwen causal caption: ablate denominator", q_nec["ablate_v"]["n_ok"], 702)
    expect("Qwen causal caption: random numerator", q_nec["ablate_random"]["n_mis"], 27)
    expect("Qwen causal caption: random denominator", q_nec["ablate_random"]["n_ok"], 685)
    q_suff = causal["Qwen2.5-Coder-7B"][0]["sufficiency"]
    expect("Qwen sufficiency: benign baseline rate", pct(q_suff["benign_baseline"]["rate"]), 0.0, 0.01)
    expect("Qwen sufficiency: benign baseline coherent count", q_suff["benign_baseline"]["n_ok"], 677)
    expect("Qwen sufficiency: coherent alpha 0.5 rate", pct(q_suff["steer_v"]["0.5"]["rate"]), 5.3, 0.06)
    expect("Qwen sufficiency: coherent alpha 0.5 count", q_suff["steer_v"]["0.5"]["n_ok"], 605)
    expect("Qwen sufficiency: alpha 1.0 coherent count displayed as low-coherence stress test", q_suff["steer_v"]["1.0"]["n_ok"], 123)
    for alpha in ("2.0", "4.0", "8.0"):
        expect(f"Qwen sufficiency: alpha {alpha} coherent count", q_suff["steer_v"][alpha]["n_ok"], 0)
    expect("Qwen sufficiency: random steering coherent count", q_suff["steer_random"]["n_ok"], 0)

    traj = load_json("traj_med.json")["trajectory"]
    expect("trajectory: 20% cosine displayed as 0.84", traj[0]["cos_to_final"], 0.84, 0.006)
    expect("trajectory: 40% cosine displayed as 0.96", traj[1]["cos_to_final"], 0.96, 0.006)
    expect("trajectory: 60% cosine displayed as 0.99", traj[2]["cos_to_final"], 0.99, 0.006)
    expect("trajectory: 20% EM displayed as 1.2%", pct(traj[0]["em_rate"]), 1.2, 0.06)
    expect("trajectory: 40% EM displayed as 3.0%", pct(traj[1]["em_rate"]), 3.0, 0.06)
    expect("trajectory: 80% EM peak displayed as 7.4%", pct(traj[3]["em_rate"]), 7.4, 0.06)
    expect("trajectory: final EM displayed as 4.7%", pct(traj[4]["em_rate"]), 4.7, 0.06)
    if not (traj[3]["em_rate"] > traj[4]["em_rate"]):
        failures.append("trajectory: final EM should be lower than the 80% observed peak")

    det = {
        "coder": load_json("detect_med.json"),
        "llama": load_json("detect_llama.json"),
        "mistral": load_json("detect_mistral.json"),
    }
    expect_text("same-recipe held-out screen: coder folds", det["coder"]["mis_above_ben"], "4/4")
    expect_text("same-recipe held-out screen: llama folds", det["llama"]["mis_above_ben"], "4/4")
    expect_text("same-recipe held-out screen: mistral folds", det["mistral"]["mis_above_ben"], "4/4")
    expect("same-recipe held-out screen: coder mis score displayed as 0.67", mean(det["coder"]["folds"], "mis_score"), 0.67, 0.006)
    expect("same-recipe held-out screen: coder ben score displayed as 0.10", mean(det["coder"]["folds"], "ben_score"), 0.10, 0.006)
    expect("same-recipe held-out screen: llama mis score displayed as 0.43", mean(det["llama"]["folds"], "mis_score"), 0.43, 0.006)
    expect("same-recipe held-out screen: llama ben score displayed as 0.23", mean(det["llama"]["folds"], "ben_score"), 0.23, 0.006)
    expect("same-recipe held-out screen: mistral mis score displayed as 0.26", mean(det["mistral"]["folds"], "mis_score"), 0.26, 0.006)
    expect("same-recipe held-out screen: mistral ben score displayed as 0.13", mean(det["mistral"]["folds"], "ben_score"), 0.13, 0.006)
    random_scores = [
        row[key]
        for data in det.values()
        for row in data["folds"]
        for key in ("mis_rand", "ben_rand")
    ]
    expect("same-recipe held-out screen: random direction displayed as about 0.015", sum(random_scores) / len(random_scores), 0.015, 0.001)


def check_baseline_bakeoff():
    path = DATA / "baselines.json"
    activation_path = DATA / "activation_pca_baseline.json"
    manifest_path = DATA / "run_manifests" / "baseline_bakeoff_manifest.json"
    if not path.exists() or not activation_path.exists() or not manifest_path.exists():
        failures.append(
            "baseline bake-off: manuscript reports the audit but committed "
            "baselines.json, activation_pca_baseline.json, and the run manifest "
            "are required"
        )
        return
    data = load_json("baselines.json")
    activation = load_json("activation_pca_baseline.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    expect_text("baseline bake-off: manifest study", manifest.get("study"), "baseline_bakeoff")
    expect_text("baseline bake-off: manifest status", manifest.get("status"), "completed")
    manifest_config = manifest.get("config", {})
    expect("baseline bake-off: manifest layer", manifest_config.get("layer"), 12)
    expect_text("baseline bake-off: manifest matrix", manifest_config.get("matrix"), "self_attn.o_proj")
    expect("baseline bake-off: manifest arm-pair minimum", manifest_config.get("min_arm_pairs"), 16)
    expect_text(
        "baseline bake-off: manifest outcome mode",
        manifest_config.get("baseline_outcome_mode"),
        "negative_or_inconclusive_audit",
    )
    outcome_validation = manifest.get("outcome_validation", {})
    expect_text(
        "baseline bake-off: accepted outcome mode",
        outcome_validation.get("requested_mode"),
        "negative_or_inconclusive_audit",
    )
    if outcome_validation.get("accepted") is not True or outcome_validation.get("errors"):
        failures.append("baseline bake-off: manifest does not record an accepted audit outcome")
    if len(outcome_validation.get("positive_criterion_failures", [])) != 1:
        failures.append(
            "baseline bake-off: manifest must record exactly the observed weight-margin positive-rule failure"
        )

    preregistration = manifest.get("preregistration", {})
    expect_text(
        "baseline bake-off: manifest source commit linkage",
        preregistration.get("source_git_commit"),
        str(manifest.get("source_git_commit")),
    )
    expect_text(
        "baseline bake-off: manifest start registration linkage",
        preregistration.get("registered_at"),
        str(manifest.get("started_at")),
    )
    if str(manifest.get("started_at", "")) >= str(manifest.get("finished_at", "")):
        failures.append("baseline bake-off: manifest timestamps are not ordered")

    artifact_sha256 = manifest.get("artifact_sha256", {})
    expect_text(
        "baseline bake-off: manifest weight artifact hash",
        artifact_sha256.get("results/data/baselines.json"),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    expect_text(
        "baseline bake-off: manifest activation artifact hash",
        artifact_sha256.get("results/data/activation_pca_baseline.json"),
        hashlib.sha256(activation_path.read_bytes()).hexdigest(),
    )
    for script, recorded_hash in manifest.get("script_sha256", {}).items():
        script_path = ROOT / script
        if not script_path.exists():
            failures.append(f"baseline bake-off: manifest script is missing: {script}")
            continue
        expect_text(
            f"baseline bake-off: manifest script hash {script}",
            recorded_hash,
            hashlib.sha256(script_path.read_bytes()).hexdigest(),
        )
    expect_text("baseline bake-off: activation schema", activation.get("schema"), "activation_pca_baseline_v1")
    expect_text("baseline bake-off: activation method", activation.get("method"), "activation_pca")
    expect("baseline bake-off: activation layer", activation.get("layer"), 12)
    expect_text("baseline bake-off: activation pooling", activation.get("pool"), "mean")
    activation_provenance = activation.get("provenance", {})
    expect("baseline bake-off: activation arm pairs", activation_provenance.get("n_pairs"), 16)
    expect("baseline bake-off: activation prompt count", activation_provenance.get("n_prompts"), 64)
    expect("baseline bake-off: activation prompt seed", activation_provenance.get("prompt_seed"), 0)
    expect_text(
        "baseline bake-off: activation prompt source",
        activation_provenance.get("prompts"),
        "data/em/em_secure.jsonl",
    )
    expect_text(
        "baseline bake-off: activation base matches manifest",
        activation_provenance.get("base"),
        str(manifest_config.get("base")),
    )
    if manifest.get("environment") != activation_provenance.get("environment"):
        failures.append(
            "baseline bake-off: manifest environment is not the activation component's hash-bound receipt"
        )
    environment = manifest.get("environment", {})
    expect_text(
        "baseline bake-off: manifest requested GPU",
        environment.get("gpu_id_requested"),
        str(manifest_config.get("gpu_id")),
    )
    expect_text(
        "baseline bake-off: manifest visible GPU",
        environment.get("cuda", {}).get("cuda_visible_devices"),
        str(manifest_config.get("gpu_id")),
    )

    resolved = activation_provenance.get("resolved_inputs", [])
    pair_suffixes = {"misaligned": {}, "benign": {}}
    for row in resolved:
        label = str(row.get("label", ""))
        label_match = re.fullmatch(r"(misaligned|benign)_(\d+)", label)
        if not label_match:
            continue
        path_match = re.search(r"_s(\d+)$", str(row.get("requested", "")))
        if not path_match:
            failures.append(
                f"baseline bake-off: {label} lacks a terminal seed suffix"
            )
            continue
        pair_suffixes[label_match.group(1)][int(label_match.group(2))] = int(path_match.group(1))
    for held in range(16):
        mis_suffix = pair_suffixes["misaligned"].get(held)
        ben_suffix = pair_suffixes["benign"].get(held)
        if mis_suffix is None or ben_suffix is None or mis_suffix != ben_suffix:
            failures.append(
                "baseline bake-off: activation fold "
                f"{held} is not a verified suffix-matched pair "
                f"(misaligned={mis_suffix}, benign={ben_suffix})"
            )

    manifest_suffixes = {}
    for group in ("misaligned", "benign"):
        suffixes = []
        for arm in manifest.get("arms", {}).get(group, []):
            match = re.search(r"_s(\d+)$", str(arm))
            if not match:
                failures.append(
                    f"baseline bake-off: manifest {group} arm lacks a terminal seed suffix: {arm!r}"
                )
                continue
            suffixes.append(int(match.group(1)))
        manifest_suffixes[group] = suffixes
        if sorted(suffixes) != list(range(16)):
            failures.append(
                f"baseline bake-off: manifest {group} arms do not cover suffixes 0--15: {suffixes!r}"
            )
    if manifest_suffixes.get("misaligned") != manifest_suffixes.get("benign"):
        failures.append(
            "baseline bake-off: manifest arm order is not suffix-matched between conditions"
        )
    activation_arms = {
        group: [
            str(row.get("requested"))
            for row in resolved
            if re.fullmatch(rf"{group}_\d+", str(row.get("label", "")))
        ]
        for group in ("misaligned", "benign")
    }
    for group in ("misaligned", "benign"):
        manifest_arms = [str(arm) for arm in manifest.get("arms", {}).get(group, [])]
        if activation_arms[group] != manifest_arms:
            failures.append(
                f"baseline bake-off: activation and weight manifest {group} arms differ"
            )
        input_hashes = activation_provenance.get("input_sha256", {})
        for arm in manifest_arms:
            prefix = arm.rstrip("/") + "/"
            if not any(str(key).startswith(prefix) for key in input_hashes):
                failures.append(
                    f"baseline bake-off: activation artifact does not content-address manifest arm {arm}"
                )

    methods = data.get("methods", {})
    embedded_activation = methods.get("activation_pca", {})
    expect_text(
        "baseline bake-off: embedded activation artifact hash",
        embedded_activation.get("artifact_sha256"),
        hashlib.sha256(activation_path.read_bytes()).hexdigest(),
    )
    expected = {
        "weight_svd": ("16/16", 0.603, 1.000),
        "diff_of_means": ("16/16", 0.627, 1.000),
        "activation_pca": ("16/16", 0.342, 1.000),
        "random_projection": ("12/16", 0.002, 0.727),
    }
    for method, (wins, margin, auc) in expected.items():
        row = methods.get(method, {}).get("detection", {})
        expect_text(f"baseline bake-off: {method} fold wins", row.get("mis_above_ben"), wins)
        expect(f"baseline bake-off: {method} mean margin", row.get("mean_margin"), margin, 0.0006)
        expect(f"baseline bake-off: {method} AUC", row.get("auc"), auc, 0.0006)
    weight_margin = methods.get("weight_svd", {}).get("detection", {}).get("mean_margin")
    row_mean_margin = methods.get("diff_of_means", {}).get("detection", {}).get("mean_margin")
    if weight_margin is not None and row_mean_margin is not None:
        expect(
            "baseline bake-off: displayed weight-SVD minus row-mean margin",
            float(weight_margin) - float(row_mean_margin),
            -0.023,
            0.0006,
        )

    text = paper_text()
    row_labels = {
        "weight_svd": "leading weight-SVD contrast",
        "diff_of_means": "row-mean weight contrast",
        "activation_pca": "activation-PCA contrast",
        "random_projection": "fixed random weight direction",
    }
    for method, label in row_labels.items():
        row = methods.get(method, {}).get("detection", {})
        expected_row = (
            f"{label} & ${row.get('mis_above_ben')}$ & "
            f"${float(row.get('mean_margin')):.3f}$ & "
            f"${float(row.get('auc')):.3f}$"
        )
        if not has_phrase(text, expected_row):
            failures.append(
                "baseline bake-off: manuscript table is missing artifact-derived "
                f"row {expected_row!r}"
            )
    expected_difference = float(weight_margin) - float(row_mean_margin)
    required_phrases = [
        "manifest-linked 16-fold comparison",
        "seeded random weight direction fixed across folds",
        "64 fixed-seed full user-and-assistant secure-code chats from",
        "data/em/em_secure.jsonl",
        f"using the unrounded margins, the observed difference is ${expected_difference:.3f}$",
        "learned directions average raw training-arm increments",
        "we do not describe the whole four-way audit as preregistered",
    ]
    for phrase in required_phrases:
        if not has_phrase(text, phrase):
            failures.append(
                "baseline bake-off: manuscript is missing provenance or scope "
                f"phrase {phrase!r}"
            )


def main():
    check_capability_caveat()
    check_random_control_wording()
    check_uncertainty_framing()
    check_abstract_intervals()
    check_reviewer_scope_caveats()
    check_cross_type_audit_numbers()
    check_misalignment_framing()
    check_spectral_summary()
    check_full_spectrum_artifact()
    check_synthetic_bbp()
    check_refusal()
    check_misalignment()
    check_baseline_bakeoff()
    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        return 1
    print("All checked paper numbers match committed result artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
