# Checkpoint-Delta Geometry for Refusal and Matched Misalignment Contrasts

> Code, manuscript source, and committed artifacts for a weight-space study of
> spectral structure, refusal interventions, and matched emergent-misalignment
> model organisms.

[Paper PDF](docs/paper.pdf) | [Theory supplement](docs/proof.pdf) |
[Reproducibility guide](docs/reproducibility.md) | [Research plan](PLAN.md) |
[Citation](CITATION.cff) | [License](LICENSE)

## Overview

For a fine-tuned checkpoint and its base model, define the weight increment

```text
Delta W = W_finetuned - W_base.
```

This project asks two separate questions:

1. Is the increment spectrally concentrated rather than diffuse?
2. Do directions selected from that geometry become behaviorally relevant
   under controlled residual-stream interventions?

The first question is descriptive. The second is studied with matched
harmful-versus-benign fine-tuning arms, held-out scoring, random-direction
controls, and projection or steering interventions. Spectral anisotropy alone
is not treated as an alignment detector.

## Main Findings

- **Instruction-tuning increments are strongly anisotropic.** Across all 224
  Llama-3-8B Base-to-Instruct linear-map increments, the leading eigenvalue is
  above the fitted Marchenko-Pastur visibility edge. The exact exceedance count
  depends on the bulk fit, so the paper treats it as an operational visibility
  measure rather than a calibrated hypothesis test.
- **Measured refusal is sensitive to the leading increment subspace.** On the
  tested harmful-prompt distribution, projecting out the top-128 subspace
  changes substring refusal on a fixed 128-prompt slice from 98.4%
  (`[94.5,99.6]%`) to 14.1% (`[9.1,21.1]%`), while one seeded
  equal-dimensional random projection gives 97.7% (`[93.3,99.2]%`). MMLU,
  ARC-C, and GSM8K fall substantially more under the spectral projection than
  under that random projection, showing broad disruption rather than capability
  preservation.
- **Matched medical organisms recover a shared contrast direction.** The
  Qwen2.5-Coder-7B checkpoint, direction, and evaluation artifacts are
  hash-linked, but the original medical training rows are not present.
  Projecting out the fitted direction changes the all-output joint
  misalignment-and-eligibility rate from 2.3% to 0.0% in-sample, while a random
  direction gives 3.4%.
- **The within-organism pattern appears across Qwen, Llama, and Mistral.** The
  Mistral intervention is partial rather than complete. These are controlled
  model organisms, not evidence about naturally occurring failures.
- **Held-out HarmBench prompts reproduce harmful-prompt refusal transfer.** The
  AdvBench-derived subspace reduces measured refusal on 400 HarmBench prompts
  from 71.2% to 5.8%, versus 65.8% for the random control.
- **Several important audits are negative or inconclusive.** The cross-type
  code-organism study does not support positive transfer; the 14B study retains
  geometry and descriptive held-out ordering but misses its frozen causal
  criteria. In the matched-fold baseline audit, all three learned methods rank
  16/16 pairs correctly; the weight-SVD versus row-mean margin ordering depends
  on whether projection norm or squared projection norm is reported.

The paper reports confidence intervals, controls, provenance limits, and the
full scope of these claims. The short summary above is not a substitute for the
methods and limitations in the manuscript.

## Repository Layout

```text
alignment-geometry/
├── README.md                 project overview and quick start
├── CITATION.cff              citation metadata
├── PLAN.md                   original research plan and completed audit record
├── LICENSE                   document and source-code licensing terms
├── requirements-local.txt    local figure and validation dependencies
├── code/                     supported producers, launchers, and validators
├── data/                     committed prompts and available training datasets
├── paper/                    self-contained manuscript source and build script
├── docs/
│   ├── paper.pdf             current paper
│   ├── proof.tex             optional extended-theory source
│   ├── proof.pdf             optional extended-theory PDF
│   └── reproducibility.md    local and GPU reproduction guide
└── results/
    ├── data/                 committed summaries, evidence, and run manifests
    └── figures/              generated PDF figures
```

Large model checkpoints, fine-tuning runs, caches, and operator scratch files
are intentionally excluded from Git.

## Quick Start

The local pipeline uses Python 3.12 or newer. Tectonic is required to build the
paper, and Poppler supplies `pdfinfo` and `pdftoppm` for visual-QA receipts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-local.txt
```

Regenerate committed figures from the committed result summaries:

```bash
python3 code/make_figures.py
```

Build the manuscript and refresh `docs/paper.pdf`:

```bash
bash paper/build.sh
```

Run the authoritative validation gates:

```bash
python3 code/paper_completion_check.py --local
python3 code/paper_completion_check.py --scope external
python3 code/check_citations.py
python3 code/check_uncertainty.py
python3 code/check_figure_palette.py
python3 code/check_secrets.py --history
```

See [docs/reproducibility.md](docs/reproducibility.md) for study launchers,
artifact ingestion, run-manifest validation, and visual-QA procedures.

## Artifacts and Provenance

The paper is driven by committed artifacts rather than copied values in the
manuscript. Important groups include:

| Study | Primary artifacts | Main validator |
|---|---|---|
| Spectral sweep | `results/data/spectral.jsonl`, `summary.json`, `full_spectrum.npz` | `check_paper_numbers.py` |
| Refusal interventions | `results/data/behavioral_capture.json`, `capability.json`, `capability_evidence.json` | `check_paper_numbers.py` |
| Medical organisms | `results/data/misalignment_eval_medical.json`, `directions_med.*`, `detect_med.json` | `check_direction_study.py` |
| Cross-family replication | `results/data/directions_llama.json`, `directions_mistral.json` | `check_direction_study.py` |
| Capability audit | `results/data/capability.json`, `capability_evidence.json` | `check_capability_result.py` |
| HarmBench transfer | `results/data/transfer.json`, `transfer_evidence.json` | `check_transfer_result.py` |
| Cross-type audit | `results/data/cross_organism.json`, `causal_misalign_code.json` | `check_cross_type_code_result.py` |
| 14B audit | `results/data/directions_14b.*`, `detect_14b.json`, `scale_14b_attempt_history.json` | `check_scale_14b_attempt_history.py` |
| Baseline audit | `results/data/baselines.json`, `activation_pca_baseline.json` | `check_baselines.py` |

All paths above are under `results/data/`. Run manifests live in
`results/data/run_manifests/`. The [results guide](results/README.md) explains
the artifact classes and editing policy.

The frozen analysis-input snapshot is indexed by
`results/data/analysis_manifest.json`. Its source revision is
`e2ba5024cfc1e3e6d50d23ad9db3d88e6ce390a4`; the manifest is content-addressed
and can be verified with:

```bash
python3 code/build_analysis_manifest.py --check
python3 code/mp_fit_sensitivity.py --check
```

Generated figures use the explicit palette in `code/figure_palette.py`.
Behavioral figures use green for benign, safe, preserved, or improved outcomes;
red for harmful, misaligned, lost, or adverse outcomes; and gray for controls.
Line styles, marker shapes, and dark keylines remain redundant with color.

## Reading Order

1. [docs/paper.pdf](docs/paper.pdf), the current empirical paper.
2. [paper/main.tex](paper/main.tex) and [paper/sections](paper/sections), the
   editable manuscript source.
3. [docs/proof.pdf](docs/proof.pdf), optional extended derivations.
4. [PLAN.md](PLAN.md), the original roadmap and audit history.
5. [docs/reproducibility.md](docs/reproducibility.md), reproduction procedures.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{gupta2026checkpointdelta,
  title  = {Checkpoint-Delta Geometry for Refusal and Matched Misalignment Contrasts},
  author = {Aryan Gupta},
  year   = {2026},
  url    = {https://github.com/aryan-cs/alignment-geometry}
}
```

## License

The paper, theory supplement, research plan, and documentation are licensed
under [CC BY-NC-ND 4.0](LICENSE). The source code is source-available but is not
covered by that document license, and no software reuse or redistribution
license is granted. See [LICENSE](LICENSE) for the complete terms.
