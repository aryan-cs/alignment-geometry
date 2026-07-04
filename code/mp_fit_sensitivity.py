#!/usr/bin/env python3
"""Reanalyze committed spectra under a conservative MP scale stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = ROOT / "results" / "data" / "spectral.jsonl"
DEFAULT_FULL = ROOT / "results" / "data" / "full_spectrum.npz"
DEFAULT_OUTPUT = ROOT / "results" / "data" / "mp_fit_sensitivity.json"


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def build(sweep_path: Path, full_path: Path) -> dict:
    rows = [json.loads(line) for line in sweep_path.read_text().splitlines() if line]
    median_ratios = []
    trace_ratios = []
    trace_to_median_scales = []

    for row in rows:
        delta = row["delta"]
        p = max(int(value) for value in delta["shape"])
        q = min(int(value) for value in delta["shape"])
        gamma = float(delta["gamma"])
        trace_sigma2 = float(delta["frob_energy"]) / (p * q)
        trace_edge = trace_sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        median_ratios.append(float(delta["top_eig_over_edge"]))
        trace_ratios.append(float(delta["top_eig"]) / trace_edge)
        trace_to_median_scales.append(trace_sigma2 / float(delta["sigma2_bulk"]))

    with np.load(full_path) as full:
        eig = np.asarray(full["eig"], dtype=np.float64)
        gamma = float(full["gamma"])
        q = int(full["q"])
        median_edge = float(full["hi"])
        trace_sigma2 = float(np.sum(eig) / q)
        trace_edge = trace_sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        representative = {
            "matrix": str(full["name"]),
            "median_match": {
                "top_eigenvalue_over_edge": float(eig[0] / median_edge),
                "eigenvalues_above_edge": int(np.sum(eig > median_edge)),
            },
            "trace_moment_match": {
                "top_eigenvalue_over_edge": float(eig[0] / trace_edge),
                "eigenvalues_above_edge": int(np.sum(eig > trace_edge)),
            },
        }

    return {
        "schema": "mp_fit_sensitivity_v1",
        "sources": [
            str(sweep_path.relative_to(ROOT)),
            str(full_path.relative_to(ROOT)),
        ],
        "matrix_count": len(rows),
        "estimators": {
            "median_match": (
                "sample spectral median divided by the unit-scale MP median"
            ),
            "trace_moment_match": (
                "mean eigenvalue; an all-spectrum stress test inflated by spikes"
            ),
        },
        "sweep": {
            "median_match": {
                "top_eigenvalue_over_edge": summarize(median_ratios),
                "matrices_above_edge": int(np.sum(np.asarray(median_ratios) > 1.0)),
                "matrices_above_five_times_edge": int(
                    np.sum(np.asarray(median_ratios) > 5.0)
                ),
            },
            "trace_moment_match": {
                "top_eigenvalue_over_edge": summarize(trace_ratios),
                "matrices_above_edge": int(np.sum(np.asarray(trace_ratios) > 1.0)),
                "matrices_above_five_times_edge": int(
                    np.sum(np.asarray(trace_ratios) > 5.0)
                ),
            },
            "trace_to_median_scale_ratio": summarize(trace_to_median_scales),
        },
        "representative_full_spectrum": representative,
        "interpretation": (
            "The leading-outlier conclusion survives the conservative trace-scale "
            "stress test, while edge-exceedance counts depend strongly on the fit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--full-spectrum", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sweep = args.sweep if args.sweep.is_absolute() else ROOT / args.sweep
    full = (
        args.full_spectrum
        if args.full_spectrum.is_absolute()
        else ROOT / args.full_spectrum
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = build(sweep, full)

    if args.check:
        recorded = json.loads(output.read_text())
        if recorded != result:
            raise SystemExit(f"stale MP-fit sensitivity artifact: {output.relative_to(ROOT)}")
        print(
            "MP-fit sensitivity current: "
            f"{result['matrix_count']} matrices, both fits above edge"
        )
        return

    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
