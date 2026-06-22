#!/usr/bin/env python3
"""Validate interval-backed claims in committed result artifacts.

This checker recomputes Wilson intervals from counts where the artifacts expose
counts, validates stored Wilson triples where artifacts store intervals directly,
and enforces conservative maximum widths for the paper's interval-backed rate
claims. It deliberately ignores geometric point estimates such as capture
enrichment, which the manuscript frames as descriptive point estimates.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
TOL = 1e-9


def load_json(name):
    with open(DATA / name) as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    if not isinstance(k, int) or not isinstance(n, int) or n <= 0 or k < 0 or k > n:
        raise ValueError(f"invalid binomial counts k={k!r}, n={n!r}")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def add(errors, context, message):
    errors.append(f"{context}: {message}")


def assert_interval(errors, context, interval, *, max_half_width=None):
    if (
        not isinstance(interval, list)
        or len(interval) != 3
        or any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in interval)
    ):
        add(errors, context, f"expected [rate, lo, hi], got {interval!r}")
        return None
    p, lo, hi = [float(x) for x in interval]
    if not (-TOL <= lo <= p + TOL and p <= hi + TOL and hi <= 1.0 + TOL):
        add(errors, context, f"invalid interval ordering [{lo:.6g}, {p:.6g}, {hi:.6g}]")
    half_width = max(p - lo, hi - p)
    if max_half_width is not None and half_width > max_half_width + TOL:
        add(errors, context, f"half-width {half_width:.4f} exceeds {max_half_width:.4f}")
    return p, lo, hi


def assert_wilson_from_counts(errors, context, interval, k, n, *, max_half_width):
    actual = assert_interval(errors, context, interval, max_half_width=max_half_width)
    if actual is None:
        return
    expected = wilson(k, n)
    for label, got, want in zip(("rate", "lo", "hi"), actual, expected):
        if abs(got - want) > 5e-9:
            add(errors, context, f"{label} {got:.12g} != Wilson({k},{n}) {want:.12g}")


def infer_count(p, n, context, errors):
    k = round(float(p) * n)
    if abs(float(p) - k / n) > 5e-9:
        add(errors, context, f"rate {p:.12g} is not an integer count over n={n}")
    return k


def check_refusal_artifacts(errors):
    sweep = load_json("ablation_sweep.json")
    n_gen = sweep.get("n_gen")
    if not isinstance(n_gen, int) or n_gen <= 0:
        add(errors, "ablation_sweep.n_gen", "missing positive generation count")
        return
    for name, row in sweep.get("conditions", {}).items():
        interval = row.get("refusal_rate")
        if interval is None:
            add(errors, f"ablation_sweep.conditions.{name}", "missing refusal_rate Wilson interval")
            continue
        k = infer_count(interval[0], n_gen, f"ablation_sweep.conditions.{name}", errors)
        assert_wilson_from_counts(
            errors,
            f"ablation_sweep.conditions.{name}.refusal_rate",
            interval,
            k,
            n_gen,
            max_half_width=0.10,
        )

    layers = load_json("ablation_layers.json")
    n_layers = layers.get("n_gen")
    if not isinstance(n_layers, int) or n_layers <= 0:
        add(errors, "ablation_layers.n_gen", "missing positive generation count")
        return
    base = layers.get("baseline")
    if base is None:
        add(errors, "ablation_layers.baseline", "missing Wilson interval")
    else:
        k = infer_count(base[0], n_layers, "ablation_layers.baseline", errors)
        assert_wilson_from_counts(errors, "ablation_layers.baseline", base, k, n_layers, max_half_width=0.10)
    for layer, row in layers.get("layers", {}).items():
        for key in ("ablate_topk", "ablate_randk"):
            interval = row.get(key)
            if interval is None:
                add(errors, f"ablation_layers.layers.{layer}.{key}", "missing Wilson interval")
                continue
            k = infer_count(interval[0], n_layers, f"ablation_layers.layers.{layer}.{key}", errors)
            assert_wilson_from_counts(
                errors,
                f"ablation_layers.layers.{layer}.{key}",
                interval,
                k,
                n_layers,
                max_half_width=0.10,
            )

    suff = load_json("sufficiency.json")
    n_suff = suff.get("n_gen")
    if isinstance(n_suff, int) and n_suff > 0:
        for family in ("spectral", "random", "refusal_dir", "spectral_subspace"):
            rows = suff.get(family, {})
            if not isinstance(rows, dict):
                add(errors, f"sufficiency.{family}", "missing family object")
                continue
            for alpha, row in rows.items():
                interval = row.get("refusal") if isinstance(row, dict) else None
                if interval is None:
                    add(errors, f"sufficiency.{family}.{alpha}", "missing refusal Wilson interval")
                    continue
                k = infer_count(interval[0], n_suff, f"sufficiency.{family}.{alpha}", errors)
                assert_wilson_from_counts(
                    errors,
                    f"sufficiency.{family}.{alpha}.refusal",
                    interval,
                    k,
                    n_suff,
                    max_half_width=0.10,
                )
    else:
        add(errors, "sufficiency.n_gen", "missing positive generation count")


def check_counted_rates(errors):
    causal = load_json("causal.json")
    for cond, row in causal.get("rates", {}).items():
        for label, count_key, n_key in (
            ("harmful", "harmful_k", "harmful_n"),
            ("harmless", "harmless_k", "harmless_n"),
        ):
            if label not in row:
                add(errors, f"causal.rates.{cond}.{label}", "missing Wilson interval")
                continue
            assert_wilson_from_counts(
                errors,
                f"causal.rates.{cond}.{label}",
                row[label],
                row.get(count_key),
                row.get(n_key),
                max_half_width=0.05,
            )

    for name in ("causal_misalign.json", "causal_misalign_llama.json", "causal_misalign_mistral.json"):
        data = load_json(name)
        for cond, row in data.get("necessity", {}).items():
            rate = row.get("rate")
            k = row.get("n_mis")
            n = row.get("n_ok")
            if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
                add(errors, f"{name}.necessity.{cond}", "missing valid n_mis/n_ok counts")
                continue
            expected = wilson(k, n)
            if abs(float(rate) - expected[0]) > 5e-9:
                add(errors, f"{name}.necessity.{cond}", f"rate {rate:.12g} != n_mis/n_ok {expected[0]:.12g}")
            half_width = max(expected[0] - expected[1], expected[2] - expected[0])
            if half_width > 0.05 + TOL:
                add(errors, f"{name}.necessity.{cond}", f"Wilson half-width {half_width:.4f} exceeds 0.0500")

    gate = load_json("misalignment_eval_medical.json")
    pooled = {"misaligned": [0, 0], "benign": [0, 0]}
    for arm, row in gate.items():
        if not isinstance(row, dict):
            continue
        k = row.get("n_misaligned")
        n = row.get("n_scored")
        rate = row.get("misalignment_rate")
        if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
            add(errors, f"misalignment_eval_medical.{arm}", "missing valid n_misaligned/n_scored counts")
            continue
        p, lo, hi = wilson(k, n)
        if abs(float(rate) - p) > 5e-9:
            add(errors, f"misalignment_eval_medical.{arm}", f"rate {rate:.12g} != n_misaligned/n_scored {p:.12g}")
        if max(p - lo, hi - p) > 0.05 + TOL:
            add(errors, f"misalignment_eval_medical.{arm}", "Wilson half-width exceeds 0.0500")
        bucket = "misaligned" if arm.startswith("misaligned_") else "benign" if arm.startswith("benign_") else None
        if bucket:
            pooled[bucket][0] += k
            pooled[bucket][1] += n
    for bucket, (k, n) in pooled.items():
        if n:
            p, lo, hi = wilson(k, n)
            if max(p - lo, hi - p) > 0.025 + TOL:
                add(errors, f"misalignment_eval_medical.{bucket}_pooled", "Wilson half-width exceeds 0.0250")

    traj = load_json("traj_med.json")
    for row in traj.get("trajectory", []):
        step = row.get("step")
        k = row.get("n_mis")
        n = row.get("n_ok")
        rate = row.get("em_rate")
        if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
            add(errors, f"traj_med.step_{step}", "missing valid n_mis/n_ok counts")
            continue
        p, lo, hi = wilson(k, n)
        if abs(float(rate) - p) > 5e-9:
            add(errors, f"traj_med.step_{step}", f"rate {rate:.12g} != n_mis/n_ok {p:.12g}")
        if max(p - lo, hi - p) > 0.05 + TOL:
            add(errors, f"traj_med.step_{step}", "Wilson half-width exceeds 0.0500")


def check_heldout_screen(errors):
    wins = total = 0
    for name in ("detect_med.json", "detect_llama.json", "detect_mistral.json"):
        data = load_json(name)
        ratio = data.get("mis_above_ben")
        match = re.fullmatch(r"(\d+)/(\d+)", ratio or "")
        if not match:
            add(errors, f"{name}.mis_above_ben", "missing fold-count ratio")
            continue
        w, n = int(match.group(1)), int(match.group(2))
        folds = data.get("folds", [])
        if n != len(folds):
            add(errors, f"{name}.mis_above_ben", "denominator does not match fold count")
        empirical = sum(1 for row in folds if row.get("mis_score", -1) > row.get("ben_score", 1))
        if w != empirical:
            add(errors, f"{name}.mis_above_ben", "numerator does not match fold scores")
        wins += w
        total += n
    if total:
        p, lo, hi = wilson(wins, total)
        if (wins, total) != (12, 12):
            add(errors, "heldout_screen", f"expected 12/12 folds, got {wins}/{total}")
        if abs(p - 1.0) > TOL or abs(lo - 0.757516345567722) > 5e-4 or abs(hi - 1.0) > TOL:
            add(errors, "heldout_screen", f"unexpected Wilson interval [{lo:.4f},{hi:.4f}] for {wins}/{total}")
        if lo < 0.75:
            add(errors, "heldout_screen", f"lower Wilson bound {lo:.4f} below 0.7500")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    errors = []
    check_refusal_artifacts(errors)
    check_counted_rates(errors)
    check_heldout_screen(errors)
    if errors:
        print(f"uncertainty check FAILED: {errors[0]}", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        return 1
    print("uncertainty check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
