# alignment-geometry

> **The spectral geometry of misalignment.** A weight-space audit study of whether fine-tuning increments expose behaviorally ablation-sensitive directions for refusal and emergent misalignment.
>
> [Paper PDF](docs/paper.pdf) · [Theory proof](docs/proof.pdf) · [Research plan](PLAN.md) · [License: CC BY-NC-ND 4.0](LICENSE) · [Source on GitHub](https://github.com/aryan-cs/alignment-geometry)

This repository contains the paper, formal random-matrix theory, analysis code, figure-generation pipeline, and committed result summaries for *The Spectral Geometry of Misalignment*.

The core object is the fine-tuning increment

```text
Delta W = W_finetuned - W_base
```

viewed through the singular spectrum of each transformer weight matrix. The project asks two separate questions:

1. Is a real instruction-tuning increment spectrally concentrated rather than diffuse?
2. Do the directions singled out by the spectrum become behaviorally ablation-sensitive for measured refusal or misalignment?

The first question is descriptive. The second is tested by matched contrastive fine-tunes and causal ablations.
Weight-space geometry is treated as a source of testable directions and compressed proxies, not as a complete account of the activation-space computations that implement refusal or misalignment.

## Current Results

The current paper reports these results from real committed artifacts under `results/data/` and `results/figures/`:

- **Instruction-tuning increments are sharply spiked.** For Llama-3-8B to Llama-3-8B-Instruct, all 224 linear maps have leading eigenvalues above the fitted Marchenko-Pastur visibility edge. The median top-to-edge ratio is about 22, and the median stable rank is near 109 against ambient dimensions in the thousands; these are energy-concentration summaries, not mechanism counts or alignment-specific detectors.
- **Measured refusal is ablation-sensitive to the leading spectral subspace in the tested scoring setup.** The empirical refusal direction is enriched in the top singular directions of the layer-14 `o_proj` increment. On held-out harmful prompts scored by substring refusal, ablating the top-128 spectral subspace collapses refusal from 98.4% (`[94.5,99.6]%`) to 3.1% (`[1.2,7.8]%`), while a random 128-dimensional subspace leaves refusal near baseline at 94.5% (`[89.1,97.3]%`). Harmless-prompt behavior and broad MMLU/GSM8K/ARC capability under this same projection remain unmeasured.
- **A behavioral-example-free misalignment direction is recovered from matched fine-tunes.** In a Qwen2.5-Coder-7B emergent-misalignment organism, the contrastive weight direction uses the matched misaligned-vs-benign arm grouping but no behavioral examples to fit the direction; it converges across four independent misaligned arms at mean cosine 0.97 while the four-arm benign training-noise summary is 0.16 at the same layer.
- **Ablating the misalignment direction suppresses the measured behavior.** Ablating the recovered direction drives emergent misalignment from 4.5% (`[3.2,6.3]%`) to 0.1% (`[0.0,0.8]%`); a random direction leaves it at 3.8% (`[2.6,5.5]%`).
- **The matched-organism result appears across three model families.** Within the same controlled medical-advice organism, Qwen2.5-Coder-7B, Llama-3-8B, and Mistral-7B all show a convergent direction whose ablation suppresses measured misalignment, with the Mistral ablation being partial rather than complete. This is not yet evidence for naturally occurring failures or other organism types.
- **The recovered direction is visible early in the recorded trajectory and separates same-recipe held-out arms.** In retrospective checkpoints it reaches near-final form before the measured behavior peaks, and in leave-one-seed-out tests it scores same-recipe held-out misaligned arms above benign controls. This is a post hoc final-direction comparison and same-recipe screen, not yet a calibrated detector or prospective forecast for arbitrary checkpoints.

The paper intentionally separates the generic fact that fine-tuning can be spectrally anisotropic from the alignment-specific evidence, which comes from directions, matched controls, and causal interventions.

Artifact map for the headline claims:

| Claim family | Primary committed artifacts | Validators/producers |
|---|---|---|
| Llama spectral sweep | `results/data/spectral.jsonl`, `results/data/summary.json`, `results/data/full_spectrum.npz` | `code/spectral.py`, `code/full_spectrum.py`, `code/check_paper_numbers.py` |
| Refusal capture, ablation, steering | `results/data/behavioral_capture.json`, `results/data/capture_sweep.json`, `results/data/ablation_sweep.json`, `results/data/ablation_layers.json`, `results/data/sufficiency.json` | `code/behavioral.py`, `code/capture_sweep.py`, `code/causal.py`, `code/ablation_sweep.py`, `code/ablation_layers.py`, `code/sufficiency.py` |
| Medical misalignment organism | `results/data/misalignment_eval_medical.json`, `results/data/directions_med.json`, `results/data/causal_misalign.json`, `results/data/detect_med.json` | `code/verify_misalignment.py`, `code/direction_recover.py`, `code/causal_misalign.py`, `code/detect_holdout.py` |
| Cross-family replication and held-out screen | `results/data/directions_llama.json`, `results/data/directions_mistral.json`, `results/data/causal_misalign_llama.json`, `results/data/causal_misalign_mistral.json`, `results/data/detect_llama.json`, `results/data/detect_mistral.json`, `results/data/traj_med.json` | `code/check_direction_study.py`, `code/check_paper_numbers.py`, `code/check_uncertainty.py` |

## Repository Layout

```text
alignment-geometry/
├── README.md
├── PLAN.md
├── code/
│   ├── spectral.py                  # safetensors reader and MP/spike statistics
│   ├── make_figures.py              # paper figure generation
│   ├── synthetic_bbp.py             # deterministic BBP spike-count validation
│   ├── capability_eval.py           # H200 capability-preservation evaluation
│   ├── check_capability_result.py   # validator/summarizer for capability_eval output
│   ├── cross_organism.py            # cross-type direction cosine and cross-detection
│   ├── check_cross_organism.py      # validator for cross_organism output
│   ├── baseline_bakeoff.py          # weight-space baselines plus real activation-PCA row
│   ├── activation_pca_baseline.py   # GPU activation-PCA baseline row producer
│   ├── check_run_manifest.py        # provenance validator for real study runs
│   ├── run_environment.py           # non-secret runtime/GPU provenance receipt
│   └── ...                          # training, steering, ablation, and analysis scripts
├── paper/
│   ├── main.tex
│   ├── build.sh                     # builds paper/main.pdf and updates docs/paper.pdf
│   └── sections/
├── docs/
│   ├── paper.pdf                    # deployed paper
│   ├── proof.tex
│   └── proof.pdf
└── results/
    ├── data/                        # committed JSON/NPZ summaries used by figures
    └── figures/                     # committed PDF figures used by the paper
```

Heavy model checkpoints and fine-tuning run directories are not committed.

## Reproducing Local Artifacts

Regenerate the figures from committed result summaries:

```bash
python3 code/make_figures.py
```

Validate headline manuscript numbers against committed result summaries:

```bash
python3 code/check_paper_numbers.py
python3 code/check_citations.py
python3 code/check_secrets.py --history
python3 code/check_uncertainty.py
```

Run the conservative completion monitor for the paper:

```bash
python3 code/paper_completion_check.py
```

This command is expected to report `incomplete` until the real capability,
OOD refusal-transfer, cross-type, scale, and baseline artifacts have been
committed and validated.
It also verifies that `docs/paper.pdf` and `docs/proof.pdf` are fresh, letter-sized,
and covered by tracked visual-QA receipts after rendering. After rebuilding or
visually reinspecting either PDF, refresh the tracked render receipts with the
same pages you inspected at full size:

```bash
python3 code/update_visual_qa_receipt.py \
  --inspected-pages-full-size 1,7,11,15,22 \
  --method "<paper contact-sheet and full-size page inspection note>"
python3 code/update_visual_qa_receipt.py \
  --pdf docs/proof.pdf \
  --receipt results/data/proof_visual_qa.json \
  --inspected-pages-full-size 1,14,15,16 \
  --method "<proof contact-sheet and full-size page inspection note>"
```

For heartbeat or local hygiene checks that should ignore missing external/H200
study outputs, run:

```bash
python3 code/paper_completion_check.py --local
```

The `--local` command should stay green whenever the repository, PDFs, figures,
visual-QA receipts, and already committed artifacts are internally consistent.
The default command combines local and external gates, so it intentionally
reports `incomplete` while paper-critical heavy-study artifacts or strict
provenance remain absent. To list only the real external completion gaps, run:

```bash
python3 code/paper_completion_check.py --scope external
```

The external report may echo tracker wording from `README.md` and `PLAN.md`, but
that wording is informational; final completion is determined by the validated
artifact and provenance gates.

Validate the current numeric misalignment-direction study bundle:

```bash
python3 code/check_direction_study.py --tag med --directions results/data/directions_med.json --detect results/data/detect_med.json --eval results/data/misalignment_eval_medical.json --causal results/data/causal_misalign.json
python3 code/check_direction_study.py --tag llama --directions results/data/directions_llama.json --directions-npz results/data/directions_llama.npz --detect results/data/detect_llama.json --causal results/data/causal_misalign_llama.json --layer 12 --k 16
python3 code/check_direction_study.py --tag mistral --directions results/data/directions_mistral.json --directions-npz results/data/directions_mistral.npz --detect results/data/detect_mistral.json --causal results/data/causal_misalign_mistral.json --layer 12 --k 16 --min-convergence 0.70 --min-convergence-gap 0.30 --min-best-gap 0.45
```

For camera-ready provenance validation after regenerating the direction,
detector, evaluation, and causal artifacts on the H200, require strict
direction, detector, evaluation, and causal provenance:

```bash
python3 code/check_direction_study.py --tag med --directions results/data/directions_med.json --directions-npz results/data/directions_med.npz --detect results/data/detect_med.json --eval results/data/misalignment_eval_medical.json --causal results/data/causal_misalign.json --require-direction-provenance --require-detect-provenance --require-eval-provenance --require-causal-provenance
python3 code/check_direction_study.py --tag llama --directions results/data/directions_llama.json --directions-npz results/data/directions_llama.npz --detect results/data/detect_llama.json --causal results/data/causal_misalign_llama.json --layer 12 --k 16 --require-direction-provenance --require-detect-provenance --require-causal-provenance
python3 code/check_direction_study.py --tag mistral --directions results/data/directions_mistral.json --directions-npz results/data/directions_mistral.npz --detect results/data/detect_mistral.json --causal results/data/causal_misalign_mistral.json --layer 12 --k 16 --min-convergence 0.70 --min-convergence-gap 0.30 --min-best-gap 0.45 --require-direction-provenance --require-detect-provenance --require-causal-provenance
```

Refresh the current medical direction vector and causal provenance on the H200
before starting cross-type transfer:

```bash
BASE=<Qwen2.5-Coder-7B-Instruct-checkpoint> \
JUDGE=<judge-checkpoint> \
bash code/run_medical_direction_refresh.sh
```

This refresh writes `results/data/directions_med.json`,
`results/data/directions_med.npz`, `results/data/detect_med.json`,
`results/data/misalignment_eval_medical.json`,
`results/data/em_generations_medical.json`, `results/data/causal_misalign.json`,
and `results/data/causal_misalign_generations.json`; all are needed for strict
camera-ready provenance. Set `REFRESH_EVAL=0` only for exploratory reruns that
should not be treated as final paper refreshes.

Refresh the existing Llama/Mistral causal artifacts with the current provenance
schema:

```bash
LLAMA_BASE=<llama-base-checkpoint> \
MISTRAL_BASE=<mistral-base-checkpoint> \
JUDGE=<judge-checkpoint> \
bash code/run_family_causal_refresh.sh
```

This refresh regenerates `results/data/directions_llama.{json,npz}`,
`results/data/detect_llama.json`, `results/data/directions_mistral.{json,npz}`,
and `results/data/detect_mistral.json` when missing or when
`FORCE_DIRECTIONS=1`, then writes
`results/data/causal_misalign_llama_generations.json` and
`results/data/causal_misalign_mistral_generations.json` alongside the refreshed
family causal summaries. For a single-family exploratory run, set
`FAMILIES=llama BASE=<llama-base-checkpoint>` or
`FAMILIES=mistral BASE=<mistral-base-checkpoint>`.

For final completion, newly generated EM-evaluation artifacts must pass
`--require-eval-provenance`, which requires per-arm judge path, rubric hashes,
generation hashes, producer script hash, and git-commit metadata. Newly
regenerated causal artifacts must also pass `--require-causal-provenance`, which
requires producer, command, model/judge, input-hash, direction-vector-hash, seed,
script-hash, git-commit metadata, and a hashed per-sample generation/judge
evidence artifact. The refresh and heavy-study launchers write these evidence
files as `results/data/causal_misalign*_generations.json`; copy and commit the
matching generation JSON together with each causal summary JSON. The heavy-study
launchers use the strict validators when they write final cross-type and
scale-study manifests.

Validate a completed baseline bake-off:

```bash
python3 code/check_activation_pca_artifact.py --input results/data/activation_pca_baseline.json
python3 code/check_baselines.py --input results/data/baselines.json --require-tracked-artifacts
```

Build the baseline bake-off after real matched arms exist. The launcher first
writes the real activation-PCA row, then computes the weight-space baselines,
then writes the manifest:

```bash
BASE=<shared-base-checkpoint> \
MIS_GLOB='<misaligned-arm-glob>' \
BEN_GLOB='<benign-arm-glob>' \
PROMPTS=data/em/em_secure.jsonl \
bash code/run_baseline_bakeoff.sh
```

After copying results back, add and commit
`results/data/activation_pca_baseline.json`, `results/data/baselines.json`, and
`results/data/run_manifests/baseline_bakeoff_manifest.json`, then rerun the
completed-artifact validators:

```bash
python3 code/check_activation_pca_artifact.py --input results/data/activation_pca_baseline.json
python3 code/check_baselines.py --input results/data/baselines.json --require-tracked-artifacts
python3 code/check_run_manifest.py \
  --final-handoff \
  --input results/data/run_manifests/baseline_bakeoff_manifest.json \
  --study baseline_bakeoff \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-arms \
  --require-config-key base \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key matrix \
  --require-config-key misaligned_glob \
  --require-config-key benign_glob \
  --require-config-key activation_pca_json \
  --require-artifact results/data/activation_pca_baseline.json \
  --require-artifact results/data/baselines.json \
  --require-script code/run_baseline_bakeoff.sh \
  --require-script code/activation_pca_baseline.py \
  --require-script code/baseline_bakeoff.py \
  --require-script code/check_baselines.py \
  --require-script code/check_activation_pca_artifact.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/spectral.py
```

Validate a completed cross-organism transfer artifact:

```bash
python3 code/check_cross_organism.py --input results/data/cross_organism.json
```

Generate and validate a completed OOD refusal-transfer artifact after supplying
a tracked prompt file that was not used to derive the refusal direction:

```bash
OOD_PROMPTS=<tracked-ood-harmful-prompts.json> \
OOD_SET=<ood-dataset-name> \
bash code/run_ood_transfer_study.sh

python3 code/transfer.py \
  --model <instruct-checkpoint> \
  --base <base-checkpoint> \
  --instruct <instruct-checkpoint> \
  --ood-set <ood-dataset-name> \
  --ood-prompts <tracked-ood-harmful-prompts.json> \
  --derivation-prompts data/harmful.json \
  --out results/data/transfer.json \
  --evidence-out results/data/transfer_evidence.json
python3 code/check_transfer_result.py \
  --input results/data/transfer.json \
  --evidence results/data/transfer_evidence.json \
  --require-paper \
  --max-ci-width 0.22
python3 code/check_run_manifest.py \
  --final-handoff \
  --input results/data/run_manifests/transfer_manifest.json \
  --study ood_refusal_transfer \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-config-key model \
  --require-config-key base \
  --require-config-key instruct \
  --require-config-key model_id \
  --require-config-key base_id \
  --require-config-key instruct_id \
  --require-config-key ood_set \
  --require-config-key ood_prompts \
  --require-config-key derivation_prompts \
  --require-config-key layer \
  --require-config-key k \
  --require-config-key n_gen \
  --require-config-key evidence_out \
  --require-config-key gpu_id \
  --require-config-key max_new \
  --require-config-key dtype \
  --require-artifact results/data/transfer.json \
  --require-artifact results/data/transfer_evidence.json \
  --require-script code/run_ood_transfer_study.sh \
  --require-script code/transfer.py \
  --require-script code/check_transfer_result.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/ablation_sweep.py \
  --require-script code/spectral.py \
  --require-command-fragment="python code/transfer.py" \
  --require-command-fragment="--ood-set" \
  --require-command-fragment="--ood-prompts" \
  --require-command-fragment="--derivation-prompts data/harmful.json" \
  --require-command-fragment="--evidence-out results/data/transfer_evidence.json" \
  --require-command-fragment="python code/check_transfer_result.py --input results/data/transfer.json --evidence results/data/transfer_evidence.json --require-paper --max-ci-width 0.22"
```

This transfer artifact supports only the harmful-prompt substring-refusal
transfer claim for the supplied OOD prompt set. It does not measure harmless
prompt behavior or broad capability preservation.

Regenerate or validate the deterministic synthetic BBP sanity check reported in
the appendix:

```bash
python3 code/synthetic_bbp.py
python3 code/synthetic_bbp.py --check
```

Build the paper and refresh `docs/paper.pdf`:

```bash
bash paper/build.sh
```

The paper build expects Tectonic on `PATH`; `paper/build.sh` prepends `/opt/homebrew/bin` for the local macOS setup used here.

The formal proof can be rebuilt separately:

```bash
cd docs
tectonic proof.tex
```

## Heavy Evaluations

Large model evaluation and training run on the H200 environment described by the project plan and local operator notes, not on a laptop. The current high-priority queued study is the capability-preservation evaluation for the top-128 refusal ablation:

```bash
nohup setsid bash code/run_capability_eval.sh > run_capability_eval.log 2>&1 </dev/null & disown
```

Set `GPU_ID=<index-or-uuid>` when the H200 host exposes more than one GPU; the
launcher queries that device with `nvidia-smi -i` and exports
`CUDA_VISIBLE_DEVICES=$GPU_ID` before loading the model.

The default paper run uses `n=500` MMLU, `n=400` GSM8K, `n=400` ARC-Challenge,
and `n=400` refusal prompts per condition, so worst-case 95% Wilson half-widths
are below about five percentage points for the reported rates. The paper
validator recomputes every Wilson interval from counts, recomputes paired
confidence intervals for the capability-drop and refusal-gap claims from
per-sample evidence, and rejects paper-study intervals with half-width above six
percentage points. It also recomputes the refusal prompt fingerprint and
selected-row hashes from committed `data/harmful.json`, and requires an exact
refusal-reference rerun on the headline ablation slice
`data/harmful.json[256:384]` using the `code/ablation_sweep.py` refusal
substring scorer. That launcher writes
`results/data/capability.json` and the raw per-sample audit file
`results/data/capability_evidence.json`. After copying the completed artifacts
and manifest back, add and commit all three files, then validate them with the
same manifest gate used by
`code/paper_completion_check.py`:

Monitor the detached job and validate its manifest as soon as it appears:

```bash
bash code/monitor_job.sh \
  --log results/logs/capability_eval.log \
  --manifest results/data/run_manifests/capability_manifest.json \
  --validator python code/check_run_manifest.py \
    --input results/data/run_manifests/capability_manifest.json \
    --study capability_preservation \
    --require-completed \
    --require-clean \
    --require-preregistration \
    --require-environment \
    --require-cuda \
    --require-gpu-name-fragment H200 \
    --require-config-key model \
    --require-config-key base \
    --require-config-key instruct \
    --require-config-key layer \
    --require-config-key topk \
    --require-config-key n_mmlu \
    --require-config-key n_gsm8k \
    --require-config-key n_arc \
    --require-config-key n_refusal \
    --require-config-key evidence_out \
    --require-config-key gpu_id \
    --require-config-key refusal_reference_start \
    --require-config-key refusal_reference_n \
    --require-config-key refusal_reference_max_new \
    --require-artifact results/data/capability.json \
    --require-artifact results/data/capability_evidence.json \
    --require-script code/run_capability_eval.sh \
    --require-script code/capability_eval.py \
    --require-script code/check_capability_result.py \
    --require-script code/check_run_manifest.py \
    --require-script code/run_environment.py \
    --require-script code/ablation_sweep.py \
    --require-script code/causal.py \
    --require-script code/spectral.py \
    --allow-untracked-artifacts \
    --require-command-fragment=--require-paper
```

`monitor_job.sh` sets its own log/manifest boundary at startup: it ignores
pre-existing log lines and pre-existing manifests, then validates only a manifest
refreshed by the current run. Start it before or during the detached run rather
than after relying on an old appended log.

```bash
python code/check_capability_result.py \
  --input results/data/capability.json \
  --evidence results/data/capability_evidence.json \
  --require-paper
python code/check_run_manifest.py \
  --final-handoff \
  --input results/data/run_manifests/capability_manifest.json \
  --study capability_preservation \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-config-key model \
  --require-config-key base \
  --require-config-key instruct \
  --require-config-key layer \
  --require-config-key topk \
  --require-config-key n_mmlu \
  --require-config-key n_gsm8k \
  --require-config-key n_arc \
  --require-config-key n_refusal \
  --require-config-key evidence_out \
  --require-config-key gpu_id \
  --require-config-key refusal_reference_start \
  --require-config-key refusal_reference_n \
  --require-config-key refusal_reference_max_new \
  --require-artifact results/data/capability.json \
  --require-artifact results/data/capability_evidence.json \
  --require-script code/run_capability_eval.sh \
  --require-script code/capability_eval.py \
  --require-script code/check_capability_result.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/ablation_sweep.py \
  --require-script code/causal.py \
  --require-script code/spectral.py \
  --require-command-fragment=--require-paper
```

`code/make_figures.py` already contains a `capability.pdf` plotting hook, but it
is inert until the real `results/data/capability.json`,
`results/data/capability_evidence.json`, and
`results/data/run_manifests/capability_manifest.json` all pass strict validation.
No placeholder capability result is committed.

Train the code-organism arms used by the cross-type study with the committed
`data/em` JSONL inputs:

```bash
BASE=<Qwen2.5-Coder-7B-Instruct-checkpoint> bash code/run_arms.sh
```

The launcher defaults to `runs/insecure_c7b_s*` versus `runs/secure_c7b_s*`,
matching `code/run_cross_type_code_study.sh`. Set `BENIGN_ARM=educational` to
reproduce the older educational-control recipe.

After a second organism has real matched arms and recovered directions, compute the
cross-organism direction and detector transfer with actual checkpoint deltas:

```bash
BASE=<shared-base-checkpoint> JUDGE=<judge-checkpoint> bash code/run_cross_type_code_study.sh
```

The launcher writes `results/data/run_manifests/cross_type_code_manifest.json`
after the real code-organism eval, direction recovery, detector, causal, and
cross-organism validators complete. If `results/data/directions_med.{json,npz}`
is absent, lacks direction provenance, or no longer hashes against its vector
artifact, it first rebuilds that medical direction bundle from the real matched
arms. It also fails fast unless the medical eval, detector, direction, and causal
artifacts all pass strict provenance checks before transfer. The manifest must
then be validated with strict provenance fragments:

```bash
python3 code/check_run_manifest.py \
  --final-handoff \
  --input results/data/run_manifests/cross_type_code_manifest.json \
  --study cross_type_code \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-arms \
  --require-config-key base \
  --require-config-key judge \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key k \
  --require-artifact results/data/directions_med.json \
  --require-artifact results/data/directions_med.npz \
  --require-artifact results/data/detect_med.json \
  --require-artifact results/data/misalignment_eval_code.json \
  --require-artifact results/data/em_generations_code.json \
  --require-artifact results/data/directions_code.json \
  --require-artifact results/data/directions_code.npz \
  --require-artifact results/data/detect_code.json \
  --require-artifact results/data/causal_misalign_code.json \
  --require-artifact results/data/causal_misalign_code_generations.json \
  --require-artifact results/data/cross_organism.json \
  --require-script code/run_cross_type_code_study.sh \
  --require-script code/verify_misalignment.py \
  --require-script code/direction_recover.py \
  --require-script code/detect_holdout.py \
  --require-script code/causal_misalign.py \
  --require-script code/cross_organism.py \
  --require-script code/check_direction_study.py \
  --require-script code/check_cross_organism.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/spectral.py \
  --require-command-fragment=--require-eval-provenance \
  --require-command-fragment=--require-direction-provenance \
  --require-command-fragment=--require-detect-provenance \
  --require-command-fragment=--require-causal-provenance
```

Before final handoff, add and commit every cross-type artifact listed above, then
rerun the strict study validators used by the completion monitor:

```bash
python3 code/check_direction_study.py \
  --tag code \
  --directions results/data/directions_code.json \
  --directions-npz results/data/directions_code.npz \
  --detect results/data/detect_code.json \
  --eval results/data/misalignment_eval_code.json \
  --causal results/data/causal_misalign_code.json \
  --require-eval-provenance \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-causal-provenance
python3 code/check_cross_organism.py \
  --input results/data/cross_organism.json \
  --require-tracked-artifacts
python3 code/paper_completion_check.py --scope external
```

The underlying cross-organism command used by the launcher is:

```bash
python code/cross_organism.py \
  --source-tag med \
  --target-tag code \
  --source-directions-npz results/data/directions_med.npz \
  --target-directions-npz results/data/directions_code.npz \
  --base <shared-base-checkpoint> \
  --runs runs \
  --source-misaligned-glob '<medical-misaligned-arm-glob>' \
  --source-benign-glob '<medical-benign-arm-glob>' \
  --target-misaligned-glob 'insecure_c7b_s*' \
  --target-benign-glob 'secure_c7b_s*' \
  --out results/data/cross_organism.json
```

The completion monitor requires the medical direction NPZ, `check_direction_study.py`
for the second organism including its causal artifact, and
`check_cross_organism.py` for this transfer artifact before the cross-type
workstream can pass.

Run the 14B scale study from existing matched 14B arms with:

```bash
BASE=<14b-base-checkpoint> JUDGE=<judge-checkpoint> bash code/run_scale_14b_study.sh
```

After copying results back, add and commit `results/data/misalignment_eval_14b.json`,
`results/data/em_generations_14b.json`, `results/data/directions_14b.{json,npz}`,
`results/data/detect_14b.json`, `results/data/causal_misalign_14b.json`,
`results/data/causal_misalign_14b_generations.json`, and
`results/data/run_manifests/scale_14b_manifest.json`, then rerun the strict study
validators:

```bash
python3 code/check_direction_study.py \
  --tag 14b \
  --directions results/data/directions_14b.json \
  --directions-npz results/data/directions_14b.npz \
  --detect results/data/detect_14b.json \
  --eval results/data/misalignment_eval_14b.json \
  --causal results/data/causal_misalign_14b.json \
  --require-eval-provenance \
  --require-direction-provenance \
  --require-detect-provenance \
  --require-causal-provenance
python3 code/check_run_manifest.py \
  --final-handoff \
  --input results/data/run_manifests/scale_14b_manifest.json \
  --study scale_14b \
  --require-completed \
  --require-clean \
  --require-preregistration \
  --require-environment \
  --require-cuda \
  --require-gpu-name-fragment H200 \
  --require-arms \
  --require-config-key base \
  --require-config-key judge \
  --require-config-key runs \
  --require-config-key layer \
  --require-config-key k \
  --require-artifact results/data/misalignment_eval_14b.json \
  --require-artifact results/data/em_generations_14b.json \
  --require-artifact results/data/directions_14b.json \
  --require-artifact results/data/directions_14b.npz \
  --require-artifact results/data/detect_14b.json \
  --require-artifact results/data/causal_misalign_14b.json \
  --require-artifact results/data/causal_misalign_14b_generations.json \
  --require-script code/run_scale_14b_study.sh \
  --require-script code/verify_misalignment.py \
  --require-script code/direction_recover.py \
  --require-script code/detect_holdout.py \
  --require-script code/causal_misalign.py \
  --require-script code/check_direction_study.py \
  --require-script code/check_run_manifest.py \
  --require-script code/run_environment.py \
  --require-script code/spectral.py \
  --require-command-fragment=--require-eval-provenance \
  --require-command-fragment=--require-direction-provenance \
  --require-command-fragment=--require-detect-provenance \
  --require-command-fragment=--require-causal-provenance
```

## Reading Order

1. **[docs/paper.pdf](docs/paper.pdf)** for the current empirical paper.
2. **[paper/main.tex](paper/main.tex)** and **[paper/sections/](paper/sections)** for the editable manuscript source.
3. **[docs/proof.pdf](docs/proof.pdf)** for the formal BBP/spiked-covariance theory.
4. **[PLAN.md](PLAN.md)** for the broader roadmap and remaining reviewer-response studies.

## Status

| Workstream | State |
|------------|-------|
| Formal theory and proof | done |
| Llama-3-8B alignment-increment spectral analysis | done |
| Refusal direction enrichment, ablation, and sufficiency tests | done |
| Matched medical emergent-misalignment organism | numeric artifacts validated; strict run provenance/vector manifest pending |
| Cross-family replication within the matched medical organism on Qwen, Llama, and Mistral | numeric artifacts validated; strict causal provenance pending |
| Retrospective training trajectory and same-recipe held-out screen | numeric artifacts validated |
| Capability-preservation eval for top-128 refusal ablation | queued for H200; harness and validator committed |
| OOD refusal transfer beyond the AdvBench-derived prompt set | pending; requires tracked prompt file, per-prompt evidence, final run manifest, and interval/effect-gated `results/data/transfer.json` |
| Cross-type misalignment direction study beyond the medical organism | pending; no sleeper-agent/RLHF-trojan result committed yet |
| 14B scale study and additional baselines | pending |

## Framing

The spectral spike count alone is not claimed to diagnose alignment or misalignment. Any real fine-tune may be anisotropic. The alignment-specific claims are directional and causal: the recovered subspaces overlap known behavior directions, matched benign controls do not recover the same misalignment direction, and ablations suppress the behavior where matched random controls do not.

The strongest current limitation is that the capability-preservation study for the refusal ablation is still pending real H200 output. Until `results/data/capability.json` exists, the paper should not claim MMLU/GSM8K/ARC preservation under the top-128 ablation.

The spectral specificity baseline is also incomplete: the current Llama spectral
census has not yet compared instruction tuning against domain adaptation, coding
or math specialization, RLHF-style preference optimization, or DPO-style
preference optimization under a shared base and matched update energy.

## Citation

```bibtex
@misc{gupta2026spectralgeometry,
  title  = {The Spectral Geometry of Misalignment},
  author = {Aryan Gupta},
  year   = {2026},
  note   = {\url{https://github.com/aryan-cs/alignment-geometry}}
}
```

## License

The writeup, formal proof, experimental plan, and documents in this repository are licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE).
