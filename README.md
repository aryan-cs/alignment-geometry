# alignment-geometry

> **The spectral geometry of misalignment.** A mechanistic-interpretability study of whether weight increments expose behaviorally load-bearing directions for refusal and emergent misalignment.
>
> [Paper PDF](docs/paper.pdf) · [Theory proof](docs/proof.pdf) · [Research plan](PLAN.md) · [License: CC BY-NC-ND 4.0](LICENSE) · [Source on GitHub](https://github.com/aryan-cs/alignment-geometry)

This repository contains the paper, formal random-matrix theory, analysis code, figure-generation pipeline, and committed result summaries for *The Spectral Geometry of Misalignment*.

The core object is the fine-tuning increment

```text
Delta W = W_finetuned - W_base
```

viewed through the singular spectrum of each transformer weight matrix. The project asks two separate questions:

1. Is a real instruction-tuning increment spectrally concentrated rather than diffuse?
2. Do the directions singled out by the spectrum form behaviorally load-bearing bottlenecks for refusal or misalignment?

The first question is descriptive. The second is tested by matched contrastive fine-tunes and causal ablations.

## Current Results

The current paper reports these results from real committed artifacts under `results/data/` and `results/figures/`:

- **Instruction-tuning increments are sharply spiked.** For Llama-3-8B to Llama-3-8B-Instruct, all 224 linear maps have leading eigenvalues above the fitted Marchenko-Pastur edge. The median top-to-edge ratio is about 22, and the median stable rank is near 109 against ambient dimensions in the thousands.
- **Refusal depends on the leading spectral subspace.** The empirical refusal direction is enriched in the top singular directions of the layer-14 `o_proj` increment. Ablating the top-128 spectral subspace collapses refusal from 98.4% to 3.1%, while a random 128-dimensional subspace leaves refusal near baseline.
- **A label-free misalignment direction is recovered from matched fine-tunes.** In a Qwen2.5-Coder-7B emergent-misalignment organism, the contrastive weight direction converges across four independent misaligned arms at cosine 0.97 while benign training noise is 0.16 at the same layer.
- **Ablating the misalignment direction suppresses the measured behavior.** Ablating the recovered direction drives emergent misalignment from 4.5% to 0.1%; a random direction leaves it at 3.8%.
- **The misalignment result replicates across three model families.** Qwen2.5-Coder-7B, Llama-3-8B, and Mistral-7B all show a convergent direction whose ablation suppresses measured misalignment, with the Mistral ablation being partial rather than complete.
- **The direction is useful before and beyond the training runs used to recover it.** It emerges early in training and separates same-recipe held-out misaligned arms from benign controls in leave-one-seed-out tests.

The paper intentionally separates the generic fact that fine-tuning can be spectrally anisotropic from the alignment-specific evidence, which comes from directions, matched controls, and causal interventions.

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
```

Run the conservative completion monitor for the paper:

```bash
python3 code/paper_completion_check.py
```

This command is expected to report `incomplete` until the real capability,
cross-type, scale, and baseline artifacts have been committed and validated.

Validate a completed misalignment-direction study bundle:

```bash
python3 code/check_direction_study.py --tag med --directions results/data/directions_med.json --detect results/data/detect_med.json --eval results/data/misalignment_eval_medical.json --causal results/data/causal_misalign.json
```

Validate a completed baseline bake-off:

```bash
python3 code/check_activation_pca_artifact.py --input results/data/activation_pca_baseline.json
python3 code/check_baselines.py --input results/data/baselines.json --require-tracked-artifacts
```

Build the baseline bake-off after real matched arms and a tracked external
activation-PCA baseline row exist:

```bash
python3 code/activation_pca_baseline.py \
  --base <shared-base-checkpoint> \
  --runs runs \
  --misaligned-glob '<misaligned-arm-glob>' \
  --benign-glob '<benign-arm-glob>' \
  --prompts data/harmful.json \
  --out results/data/activation_pca_baseline.json

python3 code/baseline_bakeoff.py \
  --base <shared-base-checkpoint> \
  --runs runs \
  --misaligned-glob '<misaligned-arm-glob>' \
  --benign-glob '<benign-arm-glob>' \
  --activation-pca-json results/data/activation_pca_baseline.json \
  --out results/data/baselines.json

python3 code/check_run_manifest.py --input results/data/run_manifests/baseline_bakeoff_manifest.json --study baseline_bakeoff --require-completed --require-clean
```

Validate a completed cross-organism transfer artifact:

```bash
python3 code/check_cross_organism.py --input results/data/cross_organism.json
```

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

That launcher writes `results/data/capability.json`. After copying the completed
artifact and manifest back, add and commit both files, then validate them with the
same manifest gate used by `code/paper_completion_check.py`:

```bash
python code/check_capability_result.py --input results/data/capability.json --require-paper
python code/check_run_manifest.py \
  --input results/data/run_manifests/capability_manifest.json \
  --study capability_preservation \
  --require-completed \
  --require-clean \
  --require-config-key model \
  --require-config-key base \
  --require-config-key instruct \
  --require-config-key layer \
  --require-config-key topk \
  --require-config-key n_mmlu \
  --require-config-key n_gsm8k \
  --require-config-key n_arc \
  --require-config-key n_refusal \
  --require-artifact results/data/capability.json \
  --require-script code/run_capability_eval.sh \
  --require-script code/capability_eval.py \
  --require-script code/check_capability_result.py \
  --require-script code/causal.py \
  --require-script code/spectral.py
```

`code/make_figures.py` already contains a `capability.pdf` plotting hook, but it is inert until the real `results/data/capability.json` exists. No placeholder capability result is committed.

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
cross-organism validators complete. The underlying cross-organism command is:

Validate the manifest with:

```bash
python3 code/check_run_manifest.py --input results/data/run_manifests/cross_type_code_manifest.json --study cross_type_code --require-completed --require-clean
```

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
python3 code/check_run_manifest.py --input results/data/run_manifests/scale_14b_manifest.json --study scale_14b --require-completed --require-clean
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
| Matched medical emergent-misalignment organism | done |
| Cross-family replication on Qwen, Llama, and Mistral | done |
| Early-training trajectory and held-out detector | done |
| Capability-preservation eval for top-128 refusal ablation | queued for H200; harness and validator committed |
| Cross-type misalignment direction study beyond the medical organism | pending; no sleeper-agent/RLHF-trojan result committed yet |
| 14B scale study and additional baselines | pending |

## Framing

The spectral spike count alone is not claimed to diagnose alignment or misalignment. Any real fine-tune may be anisotropic. The alignment-specific claims are directional and causal: the recovered subspaces overlap known behavior directions, matched benign controls do not recover the same misalignment direction, and ablations suppress the behavior where matched random controls do not.

The strongest current limitation is that the capability-preservation study for the refusal ablation is still pending real H200 output. Until `results/data/capability.json` exists, the paper should not claim MMLU/GSM8K/ARC preservation under the top-128 ablation.

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
