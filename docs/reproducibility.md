# Reproducibility Guide

This document separates the lightweight local paper pipeline from the GPU-heavy
model studies. All reported numbers come from committed artifacts under
`results/data/`; no validator or ingest helper fabricates replacement data.

## Local Environment

Use Python 3.12 or newer in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-local.txt
```

The PDF pipeline also requires:

- [Tectonic](https://tectonic-typesetting.github.io/) for LaTeX builds.
- Poppler's `pdfinfo` and `pdftoppm` for page rendering and visual QA.

## Rebuild the Paper

Regenerate every tracked figure from committed summaries:

```bash
python3 code/make_figures.py
```

Build the manuscript and copy the result to `docs/paper.pdf`:

```bash
bash paper/build.sh
```

Build the optional extended theory note from `docs/`:

```bash
cd docs
tectonic proof.tex
```

## Validation Gates

Run the local and external completion scopes independently:

```bash
python3 code/paper_completion_check.py --local
python3 code/paper_completion_check.py --scope external
```

The focused checks are useful while changing one part of the repository:

```bash
python3 code/check_paper_numbers.py
python3 code/check_citations.py
python3 code/check_uncertainty.py
python3 code/check_figure_palette.py
python3 code/check_secrets.py --history
```

Verify the frozen analysis manifest and MP-fit sensitivity audit:

```bash
python3 code/build_analysis_manifest.py --check
python3 code/mp_fit_sensitivity.py --check
```

## Visual QA

After changing either PDF, render and inspect the affected pages at full size.
Then refresh the corresponding tracked receipt with the exact pages inspected:

```bash
python3 code/update_visual_qa_receipt.py \
  --inspected-pages-full-size 1,3,7,11,13,15,21 \
  --method "<what was inspected and at what resolution>"

python3 code/update_visual_qa_receipt.py \
  --pdf docs/proof.pdf \
  --receipt results/data/proof_visual_qa.json \
  --inspected-pages-full-size 1,13,14,15 \
  --method "<what was inspected and at what resolution>"
```

The receipt records the PDF hash, page count, renderer, and per-page image
hashes. Updating a receipt without inspecting the declared pages defeats the
purpose of the gate.

## GPU Study Entry Points

GPU studies require real model checkpoints and datasets. Their run manifests
record model identifiers, exact input paths, script hashes, package versions,
CUDA state, GPU identity, configuration, and artifact hashes.

| Study | Launcher | Main validator |
|---|---|---|
| Matched medical arms | `code/run_arms_med.sh` | `code/check_direction_study.py` |
| Medical direction refresh | `code/run_medical_direction_refresh.sh` | `code/check_direction_study.py` |
| Cross-family causal refresh | `code/run_family_causal_refresh.sh` | `code/check_direction_study.py` |
| Capability audit | `code/run_capability_eval.sh` | `code/check_capability_result.py` |
| HarmBench refusal transfer | `code/run_ood_transfer_study.sh` | `code/check_transfer_result.py` |
| Cross-type code audit | `code/run_cross_type_code_study.sh` | `code/check_cross_type_code_result.py` |
| 14B scale audit | `code/run_scale_14b_study.sh` | `code/check_scale_14b_attempt_history.py` |
| Baseline comparison | `code/run_baseline_bakeoff.sh` | `code/check_baselines.py` |

Use each launcher's environment variables and `--help` output rather than
editing checkpoint paths into source files. When multiple GPUs are visible,
set `GPU_ID` explicitly so the selected device matches the manifest.

The code-organism datasets under `data/em/` are committed with row counts and
SHA-256 hashes. The medical datasets are external inputs; `run_arms_med.sh`
exits when real inputs are absent instead of substituting placeholder rows.

## Artifact Ingestion

Copy completed remote outputs into a scratch directory, then use the canonical
ingest helper for the appropriate study:

```bash
python3 code/list_external_artifact_bundles.py --bundle all

python3 code/ingest_pending_study_artifacts.py \
  --source-dir /path/to/copied/artifacts \
  --study scale_14b_audit

python3 code/ingest_pending_study_artifacts.py \
  --source-dir /path/to/copied/artifacts \
  --study baseline_bakeoff_audit
```

Other supported study names are printed by
`code/list_external_artifact_bundles.py`. Positive and negative-audit paths are
deliberately distinct. A failed or inconclusive run must remain in its audit
path and must not be relabeled as positive evidence.

Capability and current-provenance bundles also have focused helpers:

```bash
python3 code/ingest_capability_artifacts.py \
  --source-dir /path/to/copied/artifacts

python3 code/ingest_current_provenance_artifacts.py \
  --source-dir /path/to/copied/artifacts
```

After committing imported files, rerun the same helper with
`--validate-only --final-handoff` when that option is supported. Final handoff
checks tracked-file state and immutable hashes in addition to JSON shape.

## Run Manifests

Canonical manifests live in `results/data/run_manifests/`. Validate one with:

```bash
python3 code/check_run_manifest.py \
  --input results/data/run_manifests/<manifest>.json \
  --final-handoff
```

Study launchers add stricter requirements for arm separation, preregistration,
environment capture, required scripts, and output artifacts. The exact command
is preserved in each manifest and in the corresponding launcher.

## Artifact Policy

- Commit compact summaries, evidence required to audit headline claims, run
  manifests, generated paper figures, and visual-QA receipts.
- Do not commit model checkpoints, caches, complete training directories,
  browser state, credentials, tokens, or operator scratch files.
- Do not edit generated result JSON to make a gate pass. Fix the producer or
  rerun the real study under its frozen configuration.
- Keep negative and inconclusive results when they answer a preregistered audit.
