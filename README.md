# fourier-alignment

> **The spectral geometry of misalignment.** A mechanistic-interpretability study of whether weight increments expose behaviorally load-bearing directions for refusal and emergent misalignment.
>
> [Paper PDF](docs/paper.pdf) · [Theory proof](docs/proof.pdf) · [Research plan](PLAN.md) · [License: CC BY-NC-ND 4.0](LICENSE) · [Source on GitHub](https://github.com/aryan-cs/fourier-alignment)

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
- **The misalignment direction is causally necessary.** Ablating the recovered direction drives emergent misalignment from 4.5% to 0.1%; a random direction leaves it at 3.8%.
- **The misalignment result replicates across three model families.** Qwen2.5-Coder-7B, Llama-3-8B, and Mistral-7B all show a convergent, causally necessary direction, with the Mistral ablation being partial rather than complete.
- **The direction is useful before and beyond the training runs used to recover it.** It emerges early in training and separates held-out misaligned arms from benign controls in leave-one-seed-out tests.

The paper intentionally separates the generic fact that fine-tuning can be spectrally anisotropic from the alignment-specific evidence, which comes from directions, matched controls, and causal interventions.

## Repository Layout

```text
fourier-alignment/
├── README.md
├── PLAN.md
├── code/
│   ├── spectral.py                  # safetensors reader and MP/spike statistics
│   ├── make_figures.py              # paper figure generation
│   ├── synthetic_bbp.py             # deterministic BBP spike-count validation
│   ├── capability_eval.py           # H200 capability-preservation evaluation
│   ├── check_capability_result.py   # validator/summarizer for capability_eval output
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

That launcher writes `results/data/capability.json` and validates it with:

```bash
python code/check_capability_result.py --input results/data/capability.json --require-paper
```

`code/make_figures.py` already contains a `capability.pdf` plotting hook, but it is inert until the real `results/data/capability.json` exists. No placeholder capability result is committed.

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
| Cross-organism misalignment direction study | in progress on H200; no result committed yet |
| 14B scale study and additional baselines | pending |

## Framing

The spectral spike count alone is not claimed to diagnose alignment or misalignment. Any real fine-tune may be anisotropic. The alignment-specific claims are directional and causal: the recovered subspaces overlap known behavior directions, matched benign controls do not recover the same misalignment direction, and ablations remove the behavior where matched random controls do not.

The strongest current limitation is that the capability-preservation study for the refusal ablation is still pending real H200 output. Until `results/data/capability.json` exists, the paper should not claim MMLU/GSM8K/ARC preservation under the top-128 ablation.

## Citation

```bibtex
@misc{gupta2026spectralgeometry,
  title  = {The Spectral Geometry of Misalignment},
  author = {Aryan Gupta},
  year   = {2026},
  note   = {\url{https://github.com/aryan-cs/fourier-alignment}}
}
```

## License

The writeup, formal proof, experimental plan, and documents in this repository are licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE).
