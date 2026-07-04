# Results

This directory contains the committed evidence and generated figures used by
the paper. It is not a scratch-output directory.

## Structure

```text
results/
├── README.md
├── data/
│   ├── analysis_manifest.json     frozen analysis-input index
│   ├── figure_manifest.json       figure inputs, producers, and hashes
│   ├── visual_qa.json             paper render receipt
│   ├── proof_visual_qa.json       theory-supplement render receipt
│   ├── run_manifests/             environment and provenance receipts
│   └── ...                        study summaries and required evidence
└── figures/                       generated PDF figures
```

## Artifact Classes

- **Summaries** contain rates, intervals, geometry statistics, and evaluation
  aggregates consumed by the manuscript and figure code.
- **Evidence files** retain per-example or per-generation records required to
  audit a headline result.
- **Vector artifacts** store recovered directions and other arrays in NPZ form.
- **Run manifests** bind a study to inputs, scripts, environment, configuration,
  and output hashes.
- **Render receipts** bind a visually inspected PDF to rendered page hashes.

The canonical analysis freeze is `data/analysis_manifest.json`. The figure
pipeline writes `data/figure_manifest.json` after regenerating figures.

## Validation

```bash
python3 code/build_analysis_manifest.py --check
python3 code/make_figures.py --manifest-only
python3 code/check_paper_numbers.py
python3 code/check_uncertainty.py
python3 code/paper_completion_check.py --local
```

Do not hand-edit numeric artifacts to satisfy a validator. Regenerate them from
the real producer, or preserve the run as negative or inconclusive evidence.
See [the reproducibility guide](../docs/reproducibility.md) for study-specific
launch and ingestion procedures.
