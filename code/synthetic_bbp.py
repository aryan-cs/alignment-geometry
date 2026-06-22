#!/usr/bin/env python3
"""Deterministic synthetic checks for the BBP spike-counting pipeline.

The paper's spectral pipeline uses a fitted Marchenko-Pastur bulk plus a
six-Tracy-Widom cushion at the upper edge. This script exercises that exact
counter on synthetic increments where the planted rank is known.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from spectral import fit_mp_sigma, marchenko_pastur_edges


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "data" / "synthetic_bbp.json"


def orthonormal_columns(rng, rows, cols):
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)), mode="reduced")
    return q


def tw_strict_edge(sigma2, gamma, p, multiplier):
    tw_scale = (
        sigma2
        * (1.0 + math.sqrt(gamma)) ** (4.0 / 3.0)
        * gamma ** (-1.0 / 6.0)
        * p ** (-2.0 / 3.0)
    )
    return sigma2 * (1.0 + math.sqrt(gamma)) ** 2, tw_scale, (
        sigma2 * (1.0 + math.sqrt(gamma)) ** 2 + multiplier * tw_scale
    )


def make_increment(seed, p, q, sigma2, signal_rank, theta_per_direction):
    rng = np.random.default_rng(seed)
    increment = rng.normal(0.0, math.sqrt(sigma2), size=(p, q))
    planted_right = None
    if signal_rank > 0:
        planted_left = orthonormal_columns(rng, p, signal_rank)
        planted_right = orthonormal_columns(rng, q, signal_rank)
        beta = math.sqrt(theta_per_direction * p * sigma2)
        increment += beta * (planted_left @ planted_right.T)
    return increment, planted_right


def analyze_case(case, p, q, sigma2, tw_multiplier):
    increment, planted_right = make_increment(
        case["seed"],
        p,
        q,
        sigma2,
        case["signal_rank"],
        case["theta_per_direction"],
    )
    gamma = q / p
    gram = (increment.T @ increment) / p
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    fitted_sigma2 = float(fit_mp_sigma(eigvals, gamma))
    _, mp_hi = marchenko_pastur_edges(fitted_sigma2, gamma)
    mp_hi_2, tw_scale, strict_edge = tw_strict_edge(
        fitted_sigma2, gamma, p, tw_multiplier
    )
    if abs(mp_hi - mp_hi_2) > 1e-12:
        raise RuntimeError("MP edge implementation mismatch")

    strict_spikes = int((eigvals > strict_edge).sum())
    above_mp_edge = int((eigvals > mp_hi).sum())

    signal_rank = case["signal_rank"]
    total_theta = signal_rank * case["theta_per_direction"]
    result = {
        "name": case["name"],
        "seed": case["seed"],
        "signal_rank": signal_rank,
        "theta_per_direction": case["theta_per_direction"],
        "total_theta": total_theta,
        "r_star": None if total_theta == 0 else total_theta / math.sqrt(gamma),
        "expected_strict_spikes": case["expected_strict_spikes"],
        "strict_spikes": strict_spikes,
        "above_mp_edge": above_mp_edge,
        "fitted_sigma2": fitted_sigma2,
        "mp_hi": float(mp_hi),
        "tw_scale": float(tw_scale),
        "strict_edge": float(strict_edge),
        "top_eig": float(eigvals[0]),
        "top_eig_over_mp_edge": float(eigvals[0] / mp_hi),
        "strict_edge_over_mp_edge": float(strict_edge / mp_hi),
    }
    if planted_right is not None and case["expected_strict_spikes"] > 0:
        k = case["expected_strict_spikes"]
        cos2 = np.linalg.svd(eigvecs[:, :k].T @ planted_right[:, :k], compute_uv=False) ** 2
        result["planted_subspace_cos2"] = {
            "mean": float(cos2.mean()),
            "min": float(cos2.min()),
            "max": float(cos2.max()),
        }
    return result


def build_report(p, q, sigma2, seed, tw_multiplier):
    theta = 1.5
    cases = [
        {
            "name": "diffuse_null",
            "seed": seed,
            "signal_rank": 0,
            "theta_per_direction": 0.0,
            "expected_strict_spikes": 0,
        },
        {
            "name": "planted_rank_1",
            "seed": seed + 1,
            "signal_rank": 1,
            "theta_per_direction": theta,
            "expected_strict_spikes": 1,
        },
        {
            "name": "planted_rank_4",
            "seed": seed + 2,
            "signal_rank": 4,
            "theta_per_direction": theta,
            "expected_strict_spikes": 4,
        },
        {
            "name": "planted_rank_16",
            "seed": seed + 3,
            "signal_rank": 16,
            "theta_per_direction": theta,
            "expected_strict_spikes": 16,
        },
        {
            "name": "energy_matched_rank_128",
            "seed": seed + 4,
            "signal_rank": 128,
            "theta_per_direction": 16 * theta / 128,
            "expected_strict_spikes": 0,
        },
    ]
    results = [analyze_case(c, p, q, sigma2, tw_multiplier) for c in cases]
    report = {
        "schema_version": 1,
        "description": (
            "Synthetic BBP validation for the same MP plus six-Tracy-Widom "
            "strict spike counter used in spectral.py."
        ),
        "p": p,
        "q": q,
        "gamma": q / p,
        "sigma2_noise": sigma2,
        "bbp_theta_threshold": math.sqrt(q / p),
        "tw_multiplier": tw_multiplier,
        "cases": results,
    }
    validate_report(report, raise_on_error=True)
    return report


def validate_report(report, raise_on_error=False):
    errors = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty list")
    else:
        for case in cases:
            ctx = case.get("name", "<unnamed>")
            if case.get("strict_spikes") != case.get("expected_strict_spikes"):
                errors.append(
                    f"{ctx}: strict_spikes={case.get('strict_spikes')} "
                    f"expected {case.get('expected_strict_spikes')}"
                )
            if case.get("expected_strict_spikes", 0) > 0:
                overlap = case.get("planted_subspace_cos2", {})
                if overlap.get("min", 0.0) < 0.5:
                    errors.append(f"{ctx}: planted-subspace recovery is too weak")
            if case.get("expected_strict_spikes", 0) == 0:
                if case.get("top_eig", math.inf) > case.get("strict_edge", -math.inf):
                    errors.append(f"{ctx}: top eigenvalue clears strict edge")
    if errors and raise_on_error:
        raise SystemExit("\n".join(errors))
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--p", type=int, default=2048)
    ap.add_argument("--q", type=int, default=512)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tw-multiplier", type=float, default=6.0)
    ap.add_argument("--check", action="store_true", help="validate --out instead of writing it")
    args = ap.parse_args()

    out = Path(args.out)
    if args.check:
        with open(out) as f:
            report = json.load(f)
        errors = validate_report(report)
        if errors:
            for error in errors:
                print("FAIL:", error)
            return 1
        print(f"validated {out}")
        return 0

    report = build_report(args.p, args.q, args.sigma2, args.seed, args.tw_multiplier)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
    for case in report["cases"]:
        print(
            f"{case['name']}: strict={case['strict_spikes']} "
            f"expected={case['expected_strict_spikes']} "
            f"top/edge={case['top_eig_over_mp_edge']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
