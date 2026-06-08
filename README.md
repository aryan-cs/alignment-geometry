# fourier-alignment

> **The spectral geometry of misalignment.** A label-free, model-level test for whether a fine-tune installed a misaligned objective, built on the random-matrix theory of low-rank perturbations.
>
> [Read the proof (PDF)](docs/proof.pdf) · [Read the plan](PLAN.md) · [Source on GitHub](https://github.com/aryan-cs/fourier-alignment)

This repository hosts the formal theory and the research plan for a spectral approach to misalignment detection. The implementation will follow the experiments laid out in the plan. The central question: fine-tuning is a perturbation of a model's weights, so does the spectrum of that perturbation reveal whether the fine-tune installed a misaligned objective, without any labeled examples of bad behavior?

---

## In simple terms

Training a model to behave badly changes its internal numbers. This project asks whether that change leaves a fingerprint we can spot just by looking at the numbers, without having to catch the model misbehaving first.

The bet is about *shape*. A harmful change tends to be concentrated: it pushes the model hard along a few directions. A harmless change of the same overall size tends to be spread thinly across many directions. A result from random matrix theory can tell a concentrated change from a spread-out one, even when both move the weights by the same total amount. So if harmful fine-tuning really is the concentrated kind, we can flag it with no examples of bad behavior to learn from.

That is the whole idea: misalignment may be visible in how concentrated a change is rather than in how large it is. The rest of this repository makes that precise and lays out the experiment that would confirm or refute it.

## What is this, in one paragraph?

A fine-tuned model differs from its base by a weight increment, `ΔW`. We model that increment, layer by layer, as a diffuse random bulk plus a low-rank deterministic signal, and we ask whether the signal is large enough to leave a visible spike in the spectrum. Random-matrix theory answers precisely. A planted direction of strength `θ` produces an eigenvalue that detaches from the bulk if and only if `θ` exceeds a threshold set by the layer shape. The consequence we build on is about structure at fixed magnitude: hold the size of the update constant, and a concentrated, low-rank change is spectrally loud while a diffuse, high-rank change of equal energy is silent. If misaligned fine-tuning is concentrated and benign fine-tuning of equal energy is diffuse, the spectrum separates them with no behavioral labels at all.

## Why this matters

The strongest current detectors of deception read it from a model's internal states with a supervised probe. They work well, and they need exactly what is hardest to get: labeled examples of the model behaving badly, and knowledge of the relevant distribution. A label-free test would cover the case those methods cannot, a novel hidden objective with no examples.

1. **Rank is the discriminator, at controlled energy.** Every prior spectral result confounds how much a fine-tune moves the weights with how the movement is structured. The claim here isolates structure: at matched Frobenius energy, the rank of the update decides detectability, with an explicit critical rank `r* = τ / √γ`.
2. **A principled, calibrated threshold.** The decision rule comes from a Tracy–Widom null and an empirical permutation null, not from a labeled validation set, so it has a controlled false-positive rate and a stated resolution.
3. **The direction comes for free.** Above threshold, the leading eigenvector estimates the misalignment direction, recovered from the spectrum without labels, and testable by steering.

The approach is grounded in two converging lines of evidence:

- **Emergent misalignment is low-rank.** Narrow fine-tuning can induce broad misalignment (Betley et al., 2025), it is mediated by a single convergent linear direction (Soligo et al., 2025), and a single rank-one adapter suffices to install it (Turner et al., 2025).
- **Trained-network spectra carry a bulk and spikes.** Heavy-tailed self-regularization reads a label-free fingerprint of a model off its weight spectra (Martin and Mahoney, 2021), and recent work ties weight spectra, activation covariance, and fine-tuning together on transformer language models (Staats et al., 2024).

> **On the evidence so far.** This repository states a *hypothesis* and a theory that makes it falsifiable. The experiments have not been run. The theory proves a conditional: *if* misaligned fine-tunes are low-rank and energy-matched benign fine-tunes are diffuse, the spectral test separates them. Whether real fine-tunes satisfy that antecedent is an empirical question, and measuring the benign side at matched energy is the experiment the whole approach rests on. It can come out against the thesis.

## The idea in brief

For a layer with increment `ΔW` of shape `p × q` and aspect ratio `γ = q/p`, form the covariance `C = (1/p) ΔWᵀ ΔW`. Under a benign, diffuse update its spectrum is a Marchenko–Pastur bulk with a known edge. A misalignment update that concentrates signal energy `τ` into rank `r` raises a spike above that edge if and only if `r < r* = τ / √γ`. So two energy-matched fine-tunes of low and high rank are spectrally separated, and the boundary between them is explicit. The full development, with theorems and proofs, is in [`docs/proof.pdf`](docs/proof.pdf).

## The empirical program

Phases are ordered so the cheapest thing that can refute the thesis runs first. The full protocol is in [`PLAN.md`](PLAN.md).

| Phase | What it tests | Headline |
|-------|--------------|----------|
| **0. Reconnaissance** | Does a released misaligned model show a spike in `ΔW` at all? No training. | A supercritical leading eigenvalue above the permutation null, or a fast negative result. |
| **1. Matched organisms** | Build base, benign control, and misaligned arms that differ only in the alignment objective, at equal energy. | Three verified checkpoints, energy-matched increments. |
| **2. The discriminator** | Is the misaligned update lower-rank than the benign control at matched energy? | The linchpin. Locates `r_m` and `r_b` against `r*`, against a supervised probe baseline. |
| **3. Generalization** | Does the test transfer across misalignment types, and is the recovered direction causal? | Cross-type transfer, steering, and a subspace observable for rotation. |

## Repository layout

```
fourier-alignment/
├── README.md            you are here, repository landing page
├── PLAN.md              the research roadmap and experimental program
└── docs/
    ├── proof.tex        the formal theory, LaTeX source
    └── proof.pdf        compiled theory PDF
```

When code lands, the expected structure adds a `spectral/` package for the increment SVD, spike test, permutation null, and subspace distance; an `organisms/` directory for the matched fine-tuning recipes; a `probes/` directory for the supervised baselines; and `experiments/` and `results/`.

## How to read the documents

In order:

1. **[README.md](README.md)** *(this file)*. Five-minute orientation.
2. **[PLAN.md](PLAN.md)**. The research roadmap: the thesis, the novelty position, the four hypotheses with their falsification conditions, the phased program, and the methods. Roughly a fifteen-minute read.
3. **[docs/proof.pdf](docs/proof.pdf)**. The formal theory: the random-matrix preliminaries, the spiked model of fine-tuning, the detectability theorem and the rank-at-fixed-energy discriminator, the calibrated test, and where the assumptions fail.

If you read two parts of the proof, read Section 5 for the main results and Section 8 for the assumptions and threats to validity.

## Building the proof PDF

The proof is standard LaTeX and compiles cleanly with [Tectonic](https://tectonic-typesetting.github.io/), which downloads any packages it needs on first use.

```bash
# install once
brew install tectonic           # macOS

# compile
cd docs
tectonic proof.tex
```

This produces `docs/proof.pdf`. The pre-compiled PDF is committed so readers do not need a LaTeX toolchain. A traditional `latexmk -pdf proof.tex` works equivalently.

## Status

| Milestone | State |
|-----------|-------|
| Formal theory: spiked model, detectability, rank discriminator, calibrated test | ✅ done |
| Phase 0 reconnaissance harness | ⏳ pending |
| Phase 0 result on the released 32B insecure model | ⏳ pending |
| Phase 1 matched organisms at 7B, behavior verified | ⏳ pending |
| Phase 2 linchpin: benign-side rank at matched energy | ⏳ pending |
| Phase 3 cross-type transfer, steering, subspace observable | ⏳ pending |

## A note on framing

The claim is structural, not behavioral: the absence of a behavioral tell does not mean the absence of a structural one. A spectral signature, if it exists, is present in the weights whether or not the model ever acts deceptively, which is why the target is an internalized objective rather than situational deception. The predictions are falsifiable. The thesis is not about scale, and it does not claim to beat supervised probes where labels exist. The weakest links are the antecedent of the conditional, the benign-side measurement, and the anisotropy limitation that makes the activation-covariance instrument secondary.

## A note on the name

The project is named for the analogy with Fourier inspection of vision models, but the static analysis here is spectral and random-matrix theoretic, not Fourier analytic. A weight matrix has no canonical periodic axis, so the singular value decomposition is the right tool, and the name is kept for continuity rather than method. Genuine Fourier analysis is reserved for a later study of generation trajectories, where a real sequential axis makes it the correct basis.

## Citation

A preprint will follow the empirical results. For now, please cite the repository:

```
@misc{fourieralignment2026,
  title  = {The Spectral Geometry of Misalignment},
  author = {Aryan Gupta},
  year   = {2026},
  note   = {\url{https://github.com/aryan-cs/fourier-alignment}}
}
```

## License

To be determined. Until a license file is added, treat the contents as all rights reserved, with permission granted only for reading and academic discussion.
