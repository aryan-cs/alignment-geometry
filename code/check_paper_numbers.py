#!/usr/bin/env python3
"""Check headline paper numbers against committed result artifacts.

This is a provenance guardrail for hard-coded manuscript values. It does not
parse LaTeX; each assertion names the displayed claim it protects and the source
file that should support it.
"""
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


def _command_ok(args):
    proc = subprocess.run(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def capability_result_ready():
    """Return true only when the paper-grade capability artifact is validated."""
    capability = DATA / "capability.json"
    manifest = DATA / "run_manifests" / "capability_manifest.json"
    if not capability.exists() or not manifest.exists():
        return False
    result_ok = _command_ok(
        [
            sys.executable,
            "code/check_capability_result.py",
            "--input",
            "results/data/capability.json",
            "--require-paper",
        ]
    )
    if not result_ok:
        return False
    return _command_ok(
        [
            sys.executable,
            "code/check_run_manifest.py",
            "--input",
            "results/data/run_manifests/capability_manifest.json",
            "--study",
            "capability_preservation",
            "--require-completed",
            "--require-clean",
            "--require-config-key",
            "model",
            "--require-config-key",
            "base",
            "--require-config-key",
            "instruct",
            "--require-config-key",
            "layer",
            "--require-config-key",
            "topk",
            "--require-config-key",
            "n_mmlu",
            "--require-config-key",
            "n_gsm8k",
            "--require-config-key",
            "n_arc",
            "--require-config-key",
            "n_refusal",
            "--require-artifact",
            "results/data/capability.json",
            "--require-script",
            "code/run_capability_eval.sh",
            "--require-script",
            "code/capability_eval.py",
            "--require-script",
            "code/check_capability_result.py",
            "--require-script",
            "code/causal.py",
            "--require-script",
            "code/spectral.py",
            "--require-command-fragment=--require-paper",
        ]
    )


def check_capability_caveat():
    """Guard against broad-capability claims until H200 output is validated."""
    text = paper_text()
    harmless_required = [
        "harmless-prompt-behavior claim",
        "harmless-prompt rates under the same intervention are also unmeasured",
    ]
    for phrase in harmless_required:
        if not has_phrase(text, phrase):
            failures.append(
                "harmless-prompt caveat: missing required manuscript phrase "
                f"{phrase!r}"
            )
    if capability_result_ready():
        return
    required = [
        "Nor do the current ablations establish broad capability preservation",
        "MMLU/GSM8K/ARC-style evaluations under the same",
        "top-$128$ ablation remain outstanding",
    ]
    for phrase in required:
        if not has_phrase(text, phrase):
            failures.append(
                "capability caveat: missing required manuscript phrase "
                f"{phrase!r} while results/data/capability.json is absent"
                " or not paper-grade validated"
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
    forbidden = [
        "AUC",
    ]
    for phrase in forbidden:
        if phrase in compact:
            failures.append(
                "uncertainty framing: AUC appears in the manuscript without a "
                "committed per-example artifact for interval estimation"
            )
    required = [
        "point-estimate enrichment",
        "descriptive point estimates from the committed capture artifact",
        "This is a deterministic point estimate from the committed prompt set",
        "Wilson intervals below apply to the lower rate block",
        "descriptive census of the analyzed layers",
        "Geometric quantities such as subspace capture, convergence cosines, and score margins are deterministic summaries",
        "$53.9\\%$, 95\\% Wilson CI $[48.5,59.1]\\%$",
        "$12/12$; 95\\% Wilson CI $[75.8,100.0]\\%$",
    ]
    for phrase in required:
        if phrase not in compact:
            failures.append(
                "uncertainty framing: missing required manuscript phrase "
                f"{phrase!r}"
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
        ("directions " + "removes refusal", "use collapses or suppresses refusal with the measured rate"),
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
        ("collapses refusal to 3\\%", "use the exact measured rate and interval"),
        ("leading spectral directions are a refusal bottleneck", "use measured ablation-sensitivity wording"),
        ("the leading spectral subspace is a refusal bottleneck", "use measured ablation-sensitivity wording"),
        ("refusal depends on the leading spectral subspace", "use behaviorally coupled or ablation-sensitive wording"),
        ("measured refusal depends on the leading spectral subspace", "use behaviorally coupled or ablation-sensitive wording"),
        ("visible before behavior peaks", "state that the trajectory comparison is post hoc"),
        ("controlled false-positive rate", "condition false-positive control on the ideal null"),
        ("requires no distributional assumption", "state the exchangeability limitation of permutation nulls"),
        ("recovers the misalignment direction without labels", "use candidate-direction estimate wording"),
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


def check_spectral_summary():
    s = load_json("summary.json")
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
    expect("capture sweep: median top-8 enrichment displayed as 4.6x", median(enrich8), 4.6, 0.06)
    expect("capture sweep: max top-8 enrichment displayed as 14.2x", max(enrich8), 14.2, 0.06)

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
    expect("medical gate: mean misaligned rate displayed as 5.4%", sum(mis_rates) / len(mis_rates), 5.4, 0.06)
    expect("medical gate: min misaligned arm displayed as 3.5%", min(mis_rates), 3.5, 0.06)
    expect("medical gate: max misaligned arm displayed as 6.7%", max(mis_rates), 6.7, 0.06)
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
        "Qwen2.5-Coder-7B": (load_json("causal_misalign.json"), (4.5, 0.1, 3.8)),
        "Llama-3-8B": (load_json("causal_misalign_llama.json"), (3.5, 0.4, 3.4)),
        "Mistral-7B": (load_json("causal_misalign_mistral.json"), (8.9, 3.1, 7.3)),
    }
    for name, (data, (base, ablate, random)) in causal.items():
        nec = data["necessity"]
        expect(f"{name}: baseline EM", pct(nec["misaligned_baseline"]["rate"]), base, 0.06)
        expect(f"{name}: ablate direction EM", pct(nec["ablate_v"]["rate"]), ablate, 0.06)
        expect(f"{name}: random direction EM", pct(nec["ablate_random"]["rate"]), random, 0.06)
    q_nec = causal["Qwen2.5-Coder-7B"][0]["necessity"]
    expect("Qwen causal caption: baseline numerator", q_nec["misaligned_baseline"]["n_mis"], 30)
    expect("Qwen causal caption: baseline denominator", q_nec["misaligned_baseline"]["n_ok"], 668)
    expect("Qwen causal caption: ablate numerator", q_nec["ablate_v"]["n_mis"], 1)
    expect("Qwen causal caption: ablate denominator", q_nec["ablate_v"]["n_ok"], 697)
    expect("Qwen causal caption: random numerator", q_nec["ablate_random"]["n_mis"], 25)
    expect("Qwen causal caption: random denominator", q_nec["ablate_random"]["n_ok"], 663)
    q_suff = causal["Qwen2.5-Coder-7B"][0]["sufficiency"]
    expect("Qwen sufficiency: benign baseline rate", pct(q_suff["benign_baseline"]["rate"]), 0.0, 0.01)
    expect("Qwen sufficiency: benign baseline coherent count", q_suff["benign_baseline"]["n_ok"], 167)
    expect("Qwen sufficiency: coherent alpha 0.5 rate", pct(q_suff["steer_v"]["0.5"]["rate"]), 0.0, 0.01)
    expect("Qwen sufficiency: coherent alpha 0.5 count", q_suff["steer_v"]["0.5"]["n_ok"], 161)
    expect("Qwen sufficiency: alpha 1.0 coherent count displayed as degenerate", q_suff["steer_v"]["1.0"]["n_ok"], 6)
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


def main():
    check_capability_caveat()
    check_random_control_wording()
    check_uncertainty_framing()
    check_misalignment_framing()
    check_spectral_summary()
    check_full_spectrum_artifact()
    check_synthetic_bbp()
    check_refusal()
    check_misalignment()
    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        return 1
    print("All checked paper numbers match committed result artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
