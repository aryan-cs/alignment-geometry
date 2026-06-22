#!/usr/bin/env python3
"""Check headline paper numbers against committed result artifacts.

This is a provenance guardrail for hard-coded manuscript values. It does not
parse LaTeX; each assertion names the displayed claim it protects and the source
file that should support it.
"""
import json
import math
import sys
from pathlib import Path


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


def check_refusal():
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
    expect("ablation: baseline AUC displayed as 0.998", c["baseline"]["auc"], 0.998, 0.0006)
    expect("ablation: top-128 refusal displayed as 3.1%", pct(c["ablate_top128"]["refusal_rate"][0]), 3.1, 0.06)
    expect("ablation: top-128 low CI displayed as 1.2%", pct(c["ablate_top128"]["refusal_rate"][1]), 1.2, 0.06)
    expect("ablation: top-128 high CI displayed as 7.8%", pct(c["ablate_top128"]["refusal_rate"][2]), 7.8, 0.06)
    expect("ablation: top-128 AUC displayed as 0.906", c["ablate_top128"]["auc"], 0.906, 0.0006)
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

    transfer = load_json("transfer.json")
    expect("OOD transfer: baseline refusal displayed as 97.0%", pct(transfer["baseline"][0]), 97.0, 0.06)
    expect("OOD transfer: spectral ablation displayed as 0.0%", pct(transfer["ablate_topk_advbench_derived"][0]), 0.0, 0.06)
    expect("OOD transfer: random ablation displayed as 94.0%", pct(transfer["ablate_randk"][0]), 94.0, 0.06)


def check_misalignment():
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

    traj = load_json("traj_med.json")["trajectory"]
    expect("trajectory: 20% cosine displayed as 0.84", traj[0]["cos_to_final"], 0.84, 0.006)
    expect("trajectory: 40% cosine displayed as 0.96", traj[1]["cos_to_final"], 0.96, 0.006)
    expect("trajectory: 60% cosine displayed as 0.99", traj[2]["cos_to_final"], 0.99, 0.006)
    expect("trajectory: 20% EM displayed as 1.2%", pct(traj[0]["em_rate"]), 1.2, 0.06)
    expect("trajectory: 40% EM displayed as 3.0%", pct(traj[1]["em_rate"]), 3.0, 0.06)

    det = {
        "coder": load_json("detect_med.json"),
        "llama": load_json("detect_llama.json"),
        "mistral": load_json("detect_mistral.json"),
    }
    expect_text("held-out detector: coder folds", det["coder"]["mis_above_ben"], "4/4")
    expect_text("held-out detector: llama folds", det["llama"]["mis_above_ben"], "4/4")
    expect_text("held-out detector: mistral folds", det["mistral"]["mis_above_ben"], "4/4")
    expect("held-out detector: coder mis score displayed as 0.67", mean(det["coder"]["folds"], "mis_score"), 0.67, 0.006)
    expect("held-out detector: coder ben score displayed as 0.10", mean(det["coder"]["folds"], "ben_score"), 0.10, 0.006)
    expect("held-out detector: llama mis score displayed as 0.43", mean(det["llama"]["folds"], "mis_score"), 0.43, 0.006)
    expect("held-out detector: llama ben score displayed as 0.23", mean(det["llama"]["folds"], "ben_score"), 0.23, 0.006)
    expect("held-out detector: mistral mis score displayed as 0.26", mean(det["mistral"]["folds"], "mis_score"), 0.26, 0.006)
    expect("held-out detector: mistral ben score displayed as 0.13", mean(det["mistral"]["folds"], "ben_score"), 0.13, 0.006)
    random_scores = [
        row[key]
        for data in det.values()
        for row in data["folds"]
        for key in ("mis_rand", "ben_rand")
    ]
    expect("held-out detector: random direction displayed as about 0.015", sum(random_scores) / len(random_scores), 0.015, 0.001)


def main():
    check_spectral_summary()
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
