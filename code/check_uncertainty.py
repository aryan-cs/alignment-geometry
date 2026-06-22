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
RATE_CONTEXT = re.compile(
    r"\b(rate|rates|refusal|misalignment|misaligned|benign|baseline|ablat(?:e|ing|ion|ed)|"
    r"random|fold|folds|generation|generations|condition|conditions|control|controls|EM)\b",
    re.IGNORECASE,
)
INTERVAL_CONTEXT = re.compile(
    r"(Wilson|confidence|CI|interval|intervals|\[[0-9]+(?:\.[0-9]+)?,\s*[0-9]+(?:\.[0-9]+)?\])",
    re.IGNORECASE,
)
DETERMINISTIC_CONTEXT = re.compile(
    r"\b(point[- ]estimate|deterministic|geometric|cosine|stable rank|spike|spikes|"
    r"matrix|matrices|layer|layers|training|energy|ambient|descriptive|census)\b",
    re.IGNORECASE,
)
MANUSCRIPT_MAX_INTERVAL_WIDTH_PCT = 20.0


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


def assert_separated_intervals(errors, context, higher, lower):
    high = assert_interval(errors, f"{context}.higher", higher)
    low = assert_interval(errors, f"{context}.lower", lower)
    if high is None or low is None:
        return
    if high[1] <= low[2] + TOL:
        add(
            errors,
            context,
            f"Wilson intervals overlap or touch: higher lower-bound {high[1]:.4f} <= lower upper-bound {low[2]:.4f}",
        )


def strip_latex_comments(text):
    lines = []
    for line in text.splitlines():
        escaped = False
        kept = []
        for ch in line:
            if ch == "%" and not escaped:
                break
            kept.append(ch)
            escaped = ch == "\\" and not escaped
            if ch != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def tex_blocks(path):
    text = strip_latex_comments(path.read_text())
    return [block for block in re.split(r"\n\s*\n", text) if block.strip()]


def check_manuscript_interval_coverage(errors):
    """Require manuscript rate claims to carry nearby interval context.

    The artifact checks above recompute the actual intervals from counts. This
    prose check catches a different failure mode: adding or editing manuscript
    rate claims without a nearby Wilson/CI/interval statement. Deterministic
    geometric summaries such as spike counts, stable ranks, layer percentages,
    and cosine summaries are explicitly outside this binomial-rate check.
    """
    for path in sorted((ROOT / "paper" / "sections").glob("*.tex")):
        for idx, block in enumerate(tex_blocks(path), start=1):
            if r"\%" not in block:
                continue
            compact = re.sub(r"\s+", " ", block).strip()
            if not RATE_CONTEXT.search(compact):
                continue
            if INTERVAL_CONTEXT.search(compact):
                continue
            if DETERMINISTIC_CONTEXT.search(compact):
                continue
            excerpt = compact[:180] + ("..." if len(compact) > 180 else "")
            add(
                errors,
                f"manuscript interval coverage {path.relative_to(ROOT)} block {idx}",
                f"rate-like percent claim lacks nearby Wilson/CI/interval context: {excerpt!r}",
            )


def check_manuscript_interval_widths(errors):
    """Keep displayed manuscript intervals from silently becoming too loose."""
    interval_re = re.compile(
        r"\[([0-9]+(?:\.[0-9]+)?),\s*([0-9]+(?:\.[0-9]+)?)\](?:\\%)?"
    )
    for path in sorted((ROOT / "paper" / "sections").glob("*.tex")):
        text = strip_latex_comments(path.read_text())
        for match in interval_re.finditer(text):
            near = text[max(0, match.start() - 60) : match.end() + 24]
            if r"\%" not in near and not re.search(r"\b(Wilson|CI|confidence|interval)\b", near, re.I):
                continue
            window = text[max(0, match.start() - 180) : match.end() + 180]
            lo = float(match.group(1))
            hi = float(match.group(2))
            if lo > hi:
                add(
                    errors,
                    f"manuscript interval width {path.relative_to(ROOT)}",
                    f"invalid displayed interval [{lo:g},{hi:g}]",
                )
                continue
            if hi > 100.0:
                add(
                    errors,
                    f"manuscript interval width {path.relative_to(ROOT)}",
                    f"displayed interval upper bound exceeds 100%: [{lo:g},{hi:g}]",
                )
                continue
            width = hi - lo
            if width <= MANUSCRIPT_MAX_INTERVAL_WIDTH_PCT + TOL:
                continue
            heldout_all_success = "12/12" in window and lo >= 75.0 and abs(hi - 100.0) <= TOL
            if heldout_all_success:
                continue
            add(
                errors,
                f"manuscript interval width {path.relative_to(ROOT)}",
                f"displayed interval [{lo:g},{hi:g}] spans {width:.1f} percentage points",
            )


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
    conditions = sweep.get("conditions", {})
    for name, row in conditions.items():
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
    if isinstance(conditions, dict):
        top128 = conditions.get("ablate_top128", {}).get("refusal_rate")
        baseline = conditions.get("baseline", {}).get("refusal_rate")
        rand128 = conditions.get("ablate_rand128", {}).get("refusal_rate")
        assert_separated_intervals(errors, "ablation_sweep.baseline_vs_ablate_top128", baseline, top128)
        assert_separated_intervals(errors, "ablation_sweep.ablate_rand128_vs_ablate_top128", rand128, top128)

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
        if isinstance(row, dict):
            assert_separated_intervals(
                errors,
                f"ablation_layers.layers.{layer}.ablate_randk_vs_ablate_topk",
                row.get("ablate_randk"),
                row.get("ablate_topk"),
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
        necessity_intervals = {}
        for cond, row in data.get("necessity", {}).items():
            rate = row.get("rate")
            k = row.get("n_mis")
            n = row.get("n_ok")
            if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
                add(errors, f"{name}.necessity.{cond}", "missing valid n_mis/n_ok counts")
                continue
            expected = wilson(k, n)
            necessity_intervals[cond] = list(expected)
            if abs(float(rate) - expected[0]) > 5e-9:
                add(errors, f"{name}.necessity.{cond}", f"rate {rate:.12g} != n_mis/n_ok {expected[0]:.12g}")
            half_width = max(expected[0] - expected[1], expected[2] - expected[0])
            if half_width > 0.05 + TOL:
                add(errors, f"{name}.necessity.{cond}", f"Wilson half-width {half_width:.4f} exceeds 0.0500")
        assert_separated_intervals(
            errors,
            f"{name}.necessity.baseline_vs_ablate_v",
            necessity_intervals.get("misaligned_baseline"),
            necessity_intervals.get("ablate_v"),
        )
        assert_separated_intervals(
            errors,
            f"{name}.necessity.ablate_random_vs_ablate_v",
            necessity_intervals.get("ablate_random"),
            necessity_intervals.get("ablate_v"),
        )

    qwen_causal = load_json("causal_misalign.json")
    suff = qwen_causal.get("sufficiency", {})
    suff_claims = {
        "benign_baseline": suff.get("benign_baseline"),
        "steer_v.0.5": suff.get("steer_v", {}).get("0.5") if isinstance(suff.get("steer_v"), dict) else None,
    }
    for label, row in suff_claims.items():
        if not isinstance(row, dict):
            add(errors, f"causal_misalign.sufficiency.{label}", "missing claimed coherent-steering row")
            continue
        k = row.get("n_mis")
        n = row.get("n_ok")
        rate = row.get("rate")
        if not isinstance(k, int) or not isinstance(n, int) or n <= 0:
            add(errors, f"causal_misalign.sufficiency.{label}", "missing valid n_mis/n_ok counts")
            continue
        p, lo, hi = wilson(k, n)
        if abs(float(rate) - p) > 5e-9:
            add(errors, f"causal_misalign.sufficiency.{label}", f"rate {rate:.12g} != n_mis/n_ok {p:.12g}")
        if max(p - lo, hi - p) > 0.025 + TOL:
            add(
                errors,
                f"causal_misalign.sufficiency.{label}",
                f"Wilson half-width {max(p - lo, hi - p):.4f} exceeds 0.0250",
            )

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
    check_manuscript_interval_coverage(errors)
    check_manuscript_interval_widths(errors)
    if errors:
        print(f"uncertainty check FAILED: {errors[0]}", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        return 1
    print("uncertainty check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
