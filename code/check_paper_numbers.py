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


def git_file_sha256(commit, path):
    proc = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout).hexdigest()


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


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


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


def check_analysis_manifest_snapshot():
    manifest = load_json("analysis_manifest.json")
    expected = (
        manifest.get("source_revision"),
        manifest.get("artifact_count"),
        manifest.get("bundle_sha256"),
    )
    if not (
        isinstance(expected[0], str)
        and re.fullmatch(r"[0-9a-f]{40}", expected[0])
        and isinstance(expected[1], int)
        and expected[1] > 0
        and isinstance(expected[2], str)
        and re.fullmatch(r"[0-9a-f]{64}", expected[2])
    ):
        failures.append(f"analysis manifest: invalid manifest triple {expected!r}")
        return

    appendix = (ROOT / "paper" / "sections" / "appendix.tex").read_text()
    revision_match = re.search(
        r"Analysis-input snapshot:\s*\\path\{([0-9a-f]{40})\}", appendix
    )
    bundle_match = re.search(
        r"Data manifest:.*?\$(\d+)\$\s+tracked analysis\s+files;\s+"
        r"bundle SHA-256\s+\\path\{([0-9a-f]{64})\}",
        appendix,
        re.DOTALL,
    )
    if revision_match is None or bundle_match is None:
        failures.append("analysis manifest: appendix manifest triple is missing or malformed")
    else:
        displayed = (
            revision_match.group(1),
            int(bundle_match.group(1)),
            bundle_match.group(2),
        )
        if displayed != expected:
            failures.append(
                f"analysis manifest: appendix displays {displayed!r}, expected {expected!r}"
            )

    readme = (ROOT / "README.md").read_text()
    if expected[0] not in readme:
        failures.append(
            "analysis manifest: README source revision does not match "
            f"{expected[0]}"
        )


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
        "substring-scored without an output-coherence gate",
        "does not resolve output coherence or harmless-prompt effects",
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
        "highly destructive on capability benchmarks",
        "Baseline/spectral/random top-$128$ accuracy",
        "top-$128$ intervention",
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
            "AUC pools the 16 held-out scores per arm",
            "fold summaries are descriptive because training folds overlap",
        ]
        for phrase in auc_required:
            if phrase not in compact:
                failures.append(
                    "uncertainty framing: baseline AUC requires manuscript phrase "
                    f"{phrase!r}"
                )
    required = [
        "These values are deterministic summaries of the fixed prompt set",
        "Values are deterministic summaries of the capture analysis",
        "Wilson intervals are reserved for rate estimates below",
        "Joint rates use all $800$ outputs per condition; brackets give descriptive per-rate $95\\%$ Wilson intervals",
        "Rate among eligible outputs",
        "These counts depend on the fitted visibility threshold and need not equal signal rank",
        "Subspace capture, paired-agreement cosines, and score margins are deterministic summaries",
        "Matrices within a model are correlated, so we do not attach a binomial interval to this count",
        "Because folds share training arms, we treat $12/12$ as a descriptive count and do not use binomial uncertainty over folds",
    ]
    for phrase in required:
        if phrase not in compact:
            failures.append(
                "uncertainty framing: missing required manuscript phrase "
                f"{phrase!r}"
            )


def check_abstract_rates():
    """Require the abstract to state the headline rates and matched controls."""
    abstract = (ROOT / "paper" / "sections" / "abstract.tex").read_text()
    compact = re.sub(r"\s+", " ", abstract)
    required = [
        "from $98.4\\%$ to $14.1\\%$",
        "versus $97.7\\%$ for a random subspace",
        "from $2.3\\%$ to $0.0\\%$",
        "versus $3.4\\%$ under random ablation",
    ]
    for phrase in required:
        if phrase not in compact:
            failures.append(
                "abstract rates: missing headline rate or matched-control phrase "
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
                r"not by itself alignment evidence",
                r"Each endpoint delta aggregates training stages",
            ],
        ),
        (
            "Marchenko-Pastur null limitation",
            [
                r"fitted Marchenko--Pastur curve answers visibility,\s*not specificity",
                r"matched harmful and safe fine-tunes",
            ],
        ),
        (
            "stable-rank interpretation",
            [
                r"not by itself alignment evidence",
                r"Neither statistic identifies a mechanism, and both depend on the chosen parameterization",
            ],
        ),
        (
            "refusal operational definition",
            [
                r"Refusal is scored by substring match against the fixed phrase list",
                r"harmful versus harmless prompts",
                r"prompt\s*distribution,\s*topic,\s*or style",
                r"HarmBench reproduces",
                r"baseline,\s*spectral-ablation,\s*and random-ablation refusal are\s*\$71\.2\\%\$.*\$5\.8\\%\$.*\$65\.8\\%\$",
                r"does not resolve output coherence or harmless-prompt effects",
            ],
        ),
        (
            "projection-ablation breadth",
            [
                r"top-\$128\$ audit",
                r"One random subspace is a matched control, not a null distribution",
                r"spectral intervention is highly destructive on capability benchmarks",
            ],
        ),
        (
            "single-model refusal scope",
            [
                r"refusal census covers one released 8B model",
            ],
        ),
        (
            "controlled-organism scope",
            [
                r"recurring contrast directions across three 7B/8B families",
                r"Across three medical organisms the\s*direction recurs",
            ],
        ),
        (
            "proxy-not-circuit framing",
            [
                r"sensitivity in the tested models,\s*not circuit\s*locality,\s*one-dimensional sufficiency",
                r"does not localize a circuit",
                r"does not\s*support superiority of either weight method",
            ],
        ),
        (
            "external validity",
            [
                r"do not address naturally occurring failures",
            ],
        ),
        (
            "predictive validation",
            [
                r"Prospective use requires directions and\s*thresholds frozen before new endpoint deltas are observed",
                r"score has not been calibrated for arbitrary checkpoints",
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
    required = [
        "The preregistered insecure-versus-educational audit fails its frozen",
        "does not support cross-type transfer beyond the medical organism",
    ]
    for phrase in required:
        if not has_phrase(text, phrase):
            failures.append(
                "cross-type audit limitation: manuscript does not preserve the "
                f"negative result in phrase {phrase!r}"
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
    if "MP-visible structure" not in proof:
        failures.append("proof calibration: describe the committed spectral census as MP-visible structure")


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
    expect("spectral: tail displayed as 2.45e4", s["top_over_edge"]["max"], 2.45e4, 60.0)
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
    expect("energy/overlap: overlap median displayed as 0.0134", overlap_med, 0.0134, 0.00006)
    expect("energy/overlap: null median displayed as 0.00395", null_med, 0.00395, 0.000006)
    expect("energy/overlap: ratio displayed as 3.39x", overlap_med / null_med, 3.39, 0.006)
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
    if "14.1% (`[9.1,21.1]%`)" not in readme:
        failures.append("README refusal summary: missing corrected spectral-128 rate 14.1%")
    if "97.7% (`[93.3,99.2]%`)" not in readme:
        failures.append("README refusal summary: missing corrected random-128 rate 97.7%")

    sweep = load_json("capture_sweep.json")
    cap = sweep["layers"]["14"]
    for k, expected in [(8, 0.028), (32, 0.050), (128, 0.115)]:
        expect(f"capture table: o_proj k={k}", cap["capture"][str(k)], expected, 0.0006)
    for k, expected in [(8, 0.002), (32, 0.008), (128, 0.031)]:
        expect(f"capture table: null k={k}", k / 4096, expected, 0.0006)
    for k, expected in [(8, 14.2), (32, 6.4), (128, 3.7)]:
        enrich = cap["enrich"][str(k)]
        expect(f"capture table: enrichment k={k}", enrich, expected, 0.06)
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

    capability = load_json("capability.json")
    refs = capability["refusal_reference_conditions"]
    corrected = {
        "baseline": (126, 128, 98.4, 94.5, 99.6),
        "ablate_top128": (18, 128, 14.1, 9.1, 21.1),
        "ablate_rand128": (125, 128, 97.7, 93.3, 99.2),
    }
    for condition, (count, n, rate, lo, hi) in corrected.items():
        row = refs[condition]
        expect(f"corrected refusal: {condition} count", row["refusals"], count)
        expect(f"corrected refusal: {condition} denominator", row["n"], n)
        expect(f"corrected refusal: {condition} rate", pct(row["rate"][0]), rate, 0.06)
        expect(f"corrected refusal: {condition} low CI", pct(row["rate"][1]), lo, 0.06)
        expect(f"corrected refusal: {condition} high CI", pct(row["rate"][2]), hi, 0.06)

    task_expected = {
        "mmlu": (65.4, 41.8, 54.4),
        "arc_challenge": (80.0, 50.2, 65.8),
        "gsm8k": (76.8, 3.0, 65.5),
    }
    for task, expected in task_expected.items():
        for condition, value in zip(("baseline", "ablate_top128", "ablate_rand128"), expected):
            expect(
                f"capability figure: {task} {condition}",
                pct(capability["conditions"][condition][task]["accuracy"][0]),
                value,
                0.06,
            )


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
    mis_rows = [v for k, v in gate.items() if k.startswith("misaligned_")]
    mis_rates = [100 * v["n_misaligned"] / v["n_generated"] for v in mis_rows]
    ben_rates = [
        100 * v["n_misaligned"] / v["n_generated"]
        for k, v in gate.items()
        if k.startswith("benign_")
    ]
    pooled_joint = 100 * sum(v["n_misaligned"] for v in mis_rows) / sum(v["n_generated"] for v in mis_rows)
    expect("medical gate: pooled joint rate displayed as 3.9%", pooled_joint, 3.9, 0.06)
    expect("medical gate: min joint arm displayed as 2.8%", min(mis_rates), 2.8, 0.06)
    expect("medical gate: max joint arm displayed as 5.3%", max(mis_rates), 5.3, 0.06)
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
        "Qwen2.5-Coder-7B": (
            load_json("causal_misalign.json"),
            load_json("causal_misalign_generations.json"),
            (2.6, 0.0, 3.9),
            (2.3, 0.0, 3.4),
        ),
        "Llama-3-8B": (
            load_json("causal_misalign_llama.json"),
            load_json("causal_misalign_llama_generations.json"),
            (5.3, 0.5, 2.9),
            (4.9, 0.5, 2.6),
        ),
        "Mistral-7B": (
            load_json("causal_misalign_mistral.json"),
            load_json("causal_misalign_mistral_generations.json"),
            (8.7, 2.8, 8.6),
            (7.6, 2.6, 7.5),
        ),
    }
    condition_keys = ("misaligned_baseline", "ablate_v", "ablate_random")
    for name, (data, generated, conditional_expected, joint_expected) in causal.items():
        nec = data["necessity"]
        for key, conditional, joint in zip(condition_keys, conditional_expected, joint_expected):
            expect(f"{name}: conditional {key} EM", pct(nec[key]["rate"]), conditional, 0.06)
            n_generated = len(generated["conditions"][key])
            expect(f"{name}: {key} generated count", n_generated, 800)
            expect(
                f"{name}: joint {key} EM",
                100 * nec[key]["n_mis"] / n_generated,
                joint,
                0.06,
            )
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


def check_scale_14b():
    artifact_names = (
        "misalignment_eval_14b.json",
        "em_generations_14b.json",
        "directions_14b.json",
        "directions_14b.npz",
        "detect_14b.json",
        "causal_misalign_14b.json",
        "causal_misalign_14b_generations.json",
    )
    manifest_path = DATA / "run_manifests" / "scale_14b_manifest.json"
    history_path = DATA / "scale_14b_attempt_history.json"
    missing = [name for name in artifact_names if not (DATA / name).is_file()]
    if not manifest_path.is_file():
        missing.append("run_manifests/scale_14b_manifest.json")
    if not history_path.is_file():
        missing.append("scale_14b_attempt_history.json")
    if missing:
        failures.append(
            "14B scale audit: manuscript reports the audit but committed artifacts "
            f"are missing: {', '.join(missing)}"
        )
        return

    evaluation = load_json("misalignment_eval_14b.json")
    directions = load_json("directions_14b.json")
    detector = load_json("detect_14b.json")
    causal = load_json("causal_misalign_14b.json")
    causal_generations = load_json("causal_misalign_14b_generations.json")
    history = load_json("scale_14b_attempt_history.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    expect_text("14B scale audit: manifest study", manifest.get("study"), "scale_14b")
    expect_text("14B scale audit: manifest status", manifest.get("status"), "completed")
    config = manifest.get("config", {})
    expect("14B scale audit: manifest layer", config.get("layer"), 12)
    expect("14B scale audit: manifest rank", config.get("k"), 16)
    expect("14B scale audit: manifest causal sample count", config.get("n_causal"), 100)
    expect_text(
        "14B scale audit: manifest outcome mode",
        config.get("causal_outcome_mode"),
        "negative_or_inconclusive_audit",
    )
    expect("14B scale audit: manifest sampling seed", config.get("causal_sampling_seed"), 0)
    if str(manifest.get("started_at", "")) >= str(manifest.get("finished_at", "")):
        failures.append("14B scale audit: manifest timestamps are not ordered")

    source_commit = manifest.get("source_git_commit")
    provenance_blocks = [
        directions.get("provenance", {}),
        detector.get("provenance", {}),
        causal.get("provenance", {}),
    ]
    provenance_blocks.extend(
        row.get("provenance", {})
        for row in evaluation.values()
        if isinstance(row, dict)
    )
    for index, provenance in enumerate(provenance_blocks):
        expect_text(
            f"14B scale audit: artifact provenance commit {index}",
            provenance.get("git_commit"),
            source_commit,
        )

    manifest_arms = manifest.get("arms", {})
    for group in ("misaligned", "benign"):
        arms = [str(arm) for arm in manifest_arms.get(group, [])]
        suffixes = []
        for arm in arms:
            match = re.search(rf"/{group}_14b_s(\d+)$", arm)
            if not match:
                failures.append(f"14B scale audit: invalid {group} arm path {arm!r}")
                continue
            suffixes.append(int(match.group(1)))
        if suffixes != [0, 1, 2, 3]:
            failures.append(
                f"14B scale audit: {group} arms must be the four ordered matched seeds, got {suffixes!r}"
            )

    artifact_hashes = manifest.get("artifact_sha256", {})
    for name in artifact_names:
        rel = f"results/data/{name}"
        expect_text(
            f"14B scale audit: manifest artifact hash {name}",
            artifact_hashes.get(rel),
            hashlib.sha256((DATA / name).read_bytes()).hexdigest(),
        )

    pooled = {"misaligned": [0, 0], "benign": [0, 0]}
    pooled_generated = {"misaligned": 0, "benign": 0}
    for arm, row in evaluation.items():
        if not isinstance(row, dict):
            failures.append(f"14B scale audit: malformed evaluation row {arm!r}")
            continue
        group = "misaligned" if arm.startswith("misaligned_14b_s") else "benign" if arm.startswith("benign_14b_s") else None
        if group is None:
            failures.append(f"14B scale audit: unexpected evaluation arm {arm!r}")
            continue
        k = row.get("n_misaligned")
        n = row.get("n_scored")
        if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
            failures.append(f"14B scale audit: invalid counts for {arm}")
            continue
        expect(f"14B scale audit: {arm} stored rate", row.get("misalignment_rate"), k / n)
        pooled[group][0] += k
        pooled[group][1] += n
        pooled_generated[group] += int(row.get("n_generated", 0))
    if pooled != {"misaligned": [71, 1477], "benign": [0, 1562]}:
        failures.append(f"14B scale audit: pooled behavioral counts changed: {pooled!r}")
    if pooled_generated != {"misaligned": 1600, "benign": 1600}:
        failures.append(
            f"14B scale audit: pooled generated counts changed: {pooled_generated!r}"
        )
    for group, expected_display in {
        "misaligned": (4.8, 3.8, 6.0),
        "benign": (0.0, 0.0, 0.2),
    }.items():
        k, n = pooled[group]
        rate, lo, hi = wilson(k, n)
        expect(f"14B scale audit: {group} displayed rate", round(pct(rate), 1), expected_display[0])
        expect(f"14B scale audit: {group} displayed Wilson lower", round(pct(lo), 1), expected_display[1])
        expect(f"14B scale audit: {group} displayed Wilson upper", round(pct(hi), 1), expected_display[2])
    for group, expected_display in {
        "misaligned": (4.4, 3.5, 5.6),
        "benign": (0.0, 0.0, 0.2),
    }.items():
        k = pooled[group][0]
        n = pooled_generated[group]
        rate, lo, hi = wilson(k, n)
        for label, got, want in zip(
            ("rate", "Wilson lower", "Wilson upper"),
            (round(pct(rate), 1), round(pct(lo), 1), round(pct(hi), 1)),
            expected_display,
        ):
            expect(f"14B scale audit: {group} joint displayed {label}", got, want)

    layer12 = directions.get("per_layer", {}).get("12", {})
    expect("14B scale audit: layer-12 paired agreement", layer12.get("convergence_mean_abs_cos"), 0.9333010925875871)
    expect("14B scale audit: layer-12 benign reference", layer12.get("benign_null_mean_abs_cos"), 0.5775784333914314)
    expect_text("14B scale audit: held-out fold wins", detector.get("mis_above_ben"), "4/4")
    expect("14B scale audit: held-out mean margin", detector.get("mean_margin"), 0.09320093092140391)
    folds = detector.get("folds", [])
    if len(folds) != 4 or any(float(row.get("mis_score", 0)) <= float(row.get("ben_score", 0)) for row in folds):
        failures.append("14B scale audit: not every held-out misaligned arm scores above its benign match")

    necessity = causal.get("necessity", {})
    expected_counts = {
        "misaligned_baseline": (38, 746),
        "ablate_v": (27, 747),
        "ablate_random": (33, 739),
    }
    intervals = {}
    displayed = {
        "misaligned_baseline": (5.1, 3.7, 6.9),
        "ablate_v": (3.6, 2.5, 5.2),
        "ablate_random": (4.5, 3.2, 6.2),
    }
    joint_displayed = {
        "misaligned_baseline": (4.8, 3.5, 6.5),
        "ablate_v": (3.4, 2.3, 4.9),
        "ablate_random": (4.1, 3.0, 5.7),
    }
    for condition, (k, n) in expected_counts.items():
        row = necessity.get(condition, {})
        expect(f"14B scale audit: {condition} count", row.get("n_mis"), k)
        expect(f"14B scale audit: {condition} denominator", row.get("n_ok"), n)
        expect(f"14B scale audit: {condition} rate", row.get("rate"), k / n)
        interval = wilson(k, n)
        intervals[condition] = interval
        for label, got, want in zip(
            ("rate", "Wilson lower", "Wilson upper"),
            (round(pct(interval[0]), 1), round(pct(interval[1]), 1), round(pct(interval[2]), 1)),
            displayed[condition],
        ):
            expect(f"14B scale audit: {condition} displayed {label}", got, want)
        n_generated = len(causal_generations["conditions"][condition])
        expect(f"14B scale audit: {condition} generated count", n_generated, 800)
        joint_interval = wilson(k, n_generated)
        for label, got, want in zip(
            ("rate", "Wilson lower", "Wilson upper"),
            (
                round(pct(joint_interval[0]), 1),
                round(pct(joint_interval[1]), 1),
                round(pct(joint_interval[2]), 1),
            ),
            joint_displayed[condition],
        ):
            expect(f"14B scale audit: {condition} joint displayed {label}", got, want)

    baseline_rate = necessity.get("misaligned_baseline", {}).get("rate")
    ablate_rate = necessity.get("ablate_v", {}).get("rate")
    random_rate = necessity.get("ablate_random", {}).get("rate")
    drop = float(baseline_rate) - float(ablate_rate)
    gap = float(random_rate) - float(ablate_rate)
    expect("14B scale audit: displayed baseline drop", drop, 0.0148, 0.00005)
    expect("14B scale audit: displayed random-control gap", gap, 0.0085, 0.00005)
    if drop >= 0.015 or gap >= 0.015:
        failures.append("14B scale audit: negative causal outcome no longer misses both frozen point thresholds")
    if intervals["misaligned_baseline"][1] > intervals["ablate_v"][2]:
        failures.append("14B scale audit: baseline and learned-ablation Wilson intervals no longer overlap")
    if intervals["ablate_random"][1] > intervals["ablate_v"][2]:
        failures.append("14B scale audit: random and learned-ablation Wilson intervals no longer overlap")
    causal_provenance = causal.get("provenance", {})
    expect("14B scale audit: causal sampling seed", causal_provenance.get("sampling_seed"), 0)
    if causal_provenance.get("args", {}).get("necessity_only") is not True:
        failures.append("14B scale audit: causal artifact is not necessity-only")

    final = history.get("final_seeded_primary", {})
    expect_text("14B scale audit: attempt status", final.get("status"), "completed")
    expect_text(
        "14B scale audit: attempt outcome",
        final.get("outcome"),
        "negative_or_inconclusive_audit",
    )
    expect_text("14B scale audit: attempt source commit", final.get("source_git_commit"), source_commit)
    for condition, (k, n) in expected_counts.items():
        row = final.get("metrics", {}).get(condition, {})
        expect(f"14B scale audit: attempt {condition} denominator", row.get("n_scored"), n)
        expect(f"14B scale audit: attempt {condition} count", row.get("n_misaligned"), k)
        expect(f"14B scale audit: attempt {condition} rate", row.get("rate"), k / n)

    text = paper_text()
    required_phrases = [
        "Geometric reproducibility is therefore not sufficient evidence of causal control",
        "misaligned arms have a $4.4\\%$ all-output rate ($[3.5,5.6]\\%$)",
        "Layer-12 paired agreement is $0.933$",
        "against a $0.578$ benign reference",
        "the overlapping folds are not independent",
        "On the frozen causal pair \\texttt{s0} with seed 0",
        "mean margin $0.093$",
        "The $0.0148$ baseline drop",
        "$0.0085$ random-minus-learned gap",
        "the marginal intervals overlap",
        "unseeded pre-freeze runs remain exploratory",
        "single post-freeze seeded run, without an outcome-dependent retry",
    ]
    for phrase in required_phrases:
        if not has_phrase(text, phrase):
            failures.append(f"14B scale audit: manuscript is missing artifact-linked phrase {phrase!r}")


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
    source_commit = str(manifest.get("source_git_commit"))
    for script, recorded_hash in manifest.get("script_sha256", {}).items():
        source_hash = git_file_sha256(source_commit, script)
        if source_hash is None:
            failures.append(
                f"baseline bake-off: manifest script is unavailable at {source_commit}: {script}"
            )
            continue
        expect_text(
            f"baseline bake-off: manifest script hash {script}",
            recorded_hash,
            source_hash,
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
    squared_expected = {
        "weight_svd": 0.443,
        "diff_of_means": 0.433,
        "activation_pca": 0.385,
        "random_projection": 0.00006,
    }
    squared_margins = {}
    for method, expected_margin in squared_expected.items():
        folds = methods.get(method, {}).get("detection", {}).get("folds", [])
        squared_margin = sum(
            float(row["mis_score"]) ** 2 - float(row["ben_score"]) ** 2
            for row in folds
        ) / len(folds)
        squared_margins[method] = squared_margin
        expect(
            f"baseline bake-off: {method} squared-score margin",
            squared_margin,
            expected_margin,
            0.0006,
        )
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
        squared = squared_margins[method]
        squared_text = f"{squared:.5f}" if abs(squared) < 0.001 else f"{squared:.3f}"
        expected_row = (
            f"{label} & ${row.get('mis_above_ben')}$ & "
            f"${float(row.get('mean_margin')):.3f}$ & "
            f"${squared_text}$ & "
            f"${float(row.get('auc')):.3f}$"
        )
        if not has_phrase(text, expected_row):
            failures.append(
                "baseline bake-off: manuscript table is missing artifact-derived "
                f"row {expected_row!r}"
            )
    expected_difference = float(weight_margin) - float(row_mean_margin)
    required_phrases = [
        "the 16-fold summaries are not independent replications",
        "seeded random weight direction fixed across folds",
        "$64$ fixed-seed, mean-token-pooled user-and-assistant secure-code chats",
        "Squaring each held-out score to obtain an energy fraction reverses the mean-margin ordering",
        "medical harmful/safe seed pairs \\texttt{s0} through \\texttt{s15}",
        "learned directions average raw training-arm increments",
        "the full four-way comparison is not preregistered",
    ]
    for phrase in required_phrases:
        if not has_phrase(text, phrase):
            failures.append(
                "baseline bake-off: manuscript is missing provenance or scope "
                f"phrase {phrase!r}"
            )


def main():
    check_analysis_manifest_snapshot()
    check_capability_caveat()
    check_random_control_wording()
    check_uncertainty_framing()
    check_abstract_rates()
    check_reviewer_scope_caveats()
    check_cross_type_audit_numbers()
    check_misalignment_framing()
    check_spectral_summary()
    check_full_spectrum_artifact()
    check_synthetic_bbp()
    check_refusal()
    check_misalignment()
    check_scale_14b()
    check_baseline_bakeoff()
    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        return 1
    print("All checked paper numbers match committed result artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
