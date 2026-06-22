# Research plan: the spectral geometry of misalignment

> A roadmap for testing whether concentrated alignment-relevant updates leave spectral and directional signatures under matched benign controls, built on random-matrix theory and evaluated on confound-controlled model organisms.

This document is the research roadmap. The formal theory is in [`docs/proof.pdf`](docs/proof.pdf); read it for the definitions, theorems, and proofs that the plan below operationalizes. The short version: fine-tuning is a perturbation of the weights, and we ask whether the spectrum of that perturbation can expose a concentrated misalignment-associated signal under a matched benign comparison, without using labeled examples of bad behavior to fit the statistic.

---

## 1. The question

Given a fine-tuned language model and its base, decide whether the fine-tune installed a misaligned objective, using only the weights and the ability to run forward passes on inputs we choose. No labeled examples of the model lying or defecting, because in the realistic case those are exactly what one does not have.

The strongest existing white-box answers are supervised probes, which read deception from the residual stream at high accuracy when honest-versus-deceptive labels are available ([Goldowsky-Dill et al., 2025](https://arxiv.org/abs/2502.03407)), and hidden-objective audits can work when auditors have enough access and examples to investigate ([Marks et al., 2025](https://arxiv.org/abs/2503.10965)). The binding constraint is the supervision. A method that needs none would cover the regime the supervised results structurally cannot: a novel hidden objective with no examples and an unknown trigger distribution.

## 2. The thesis

We model the weight increment of a single layer, `ΔW = W_ft − W_base`, as a diffuse random bulk plus a low-rank deterministic signal, and apply the Baik–Ben Arous–Péché phase transition for spiked covariance matrices. The theory yields one sharp, falsifiable claim:

> At controlled weight-change energy, the **rank** of the update is the discriminator. A misalignment update that concentrates its energy in low rank crosses the detectability threshold and shows a spike above the Marchenko–Pastur bulk; a benign update of equal energy that spreads over higher rank stays below threshold and is spectrally invisible. The critical rank is `r* = τ / √γ`, where `τ` is the signal energy and `γ` the layer aspect ratio.

This is the part we are not aware of prior work isolating: existing spectral-fine-tuning results confound how much the weights move with how the movement is structured. The current evidence has narrowed the claim: scalar spike counts alone are not treated as an alignment or misalignment diagnostic, because matched benign controls are also anisotropic. The supported result is directional and contrastive: matched misaligned arms recover a shared direction, matched benign controls do not, and same-recipe held-out arms separate under that direction.

## 3. Contribution and novelty

A literature pass (Section 9) shows the **conjunction is novel while the ingredients are not**.

- Reading a label-free spectral fingerprint off weights is heavy-tailed self-regularization ([Martin and Mahoney, 2021](https://jmlr.org/papers/v22/20-410.html)), which targets model quality, not alignment, and has no detectability threshold.
- That emergent misalignment is low-rank, indeed a single convergent linear direction, is Soligo, Turner, and Nanda ([Soligo et al., 2025](https://arxiv.org/abs/2506.11618)); they find the direction with labels.
- Matched aligned, benign, and misaligned organisms are the Model Organisms for Emergent Misalignment methodology ([Turner et al., 2025](https://arxiv.org/abs/2506.11613)).
- Random-matrix detection in language models exists for hallucination ([Ettori, 2026](https://arxiv.org/abs/2601.17357)), per input rather than per model.
- The closest neighbor, Staats, Thamm, and Rosenow ([Staats et al., 2024](https://arxiv.org/abs/2410.17770)), ties weight spectra, activation covariance, and fine-tuning together, but does not pose a benign-versus-misaligned contrast at matched energy.

What is ours:

1. A random-matrix model of fine-tuning as a low-rank perturbation of the weight increment, with the BBP threshold as the detectability criterion.
2. The **rank-at-fixed-energy discriminator**, with the explicit critical rank `r* = τ/√γ`.
3. A **label-free, same-recipe direction screen** built from matched weight increments, with calibrated spectral diagnostics as supporting structure rather than a stand-alone diagnostic.
4. A **confound-controlled measurement of the benign side at matched energy**, which is the experiment everything rests on and which we have not found in the cited prior work.

## 4. What is proved, and what we are betting

The theorems prove a **conditional**: if a misaligned fine-tune deposits a low-rank signal and an energy-matched benign fine-tune spreads its budget over higher rank, the spectral test separates them. The **antecedent**, that real misaligned updates are concentrated and real benign updates of equal energy are diffuse, is an empirical hypothesis. The supporting evidence is suggestive (single-direction emergent misalignment, rank-one adapters that suffice to induce it, shared low-dimensional weight subspaces across tasks) but nobody has measured the benign side at matched energy. The plan is built so that the linchpin experiment tests the antecedent directly and can refute the whole approach cheaply.

## 5. Hypotheses and falsification

| ID | Hypothesis | Refuted if |
|----|-----------|-----------|
| H1 | A full misaligned fine-tune produces a supercritical leading eigenvalue in `ΔW` at some layer, above the permutation null. | No layer clears the null on any released misaligned model. |
| H2 | At matched Frobenius energy, the misaligned update has lower rank than a benign control, with `r_m < r* ≤ r_b`. | Benign control is equally low-rank at matched energy. |
| H3 | The spectrum-recovered direction `v̂₁` causally modulates misaligned behavior under steering. | Steering along `v̂₁` does not change the behavior more than a random direction. |
| H4 | The spike test transfers across misalignment types (emergent misalignment, sleeper-agent backdoor, RLHF trojan) without retraining. | A detector calibrated on one type fails on the others at chance. |

H2 is the load-bearing test. If it fails, the discriminator is empty and we report a negative result, which is itself worth publishing given the strength of the low-rank priors in the literature.

## 6. Empirical program

Phases are ordered so the cheapest thing that can kill the thesis runs first.

| Phase | What | Output | Kills the thesis if |
|-------|------|--------|--------------------|
| 0. Reconnaissance | Spectral analysis of `ΔW` for the released `emergent-misalignment/Qwen-Coder-Insecure` (32B) versus its base. No training. | Per-layer standardized leading eigenvalue, spike rank, permutation p-values. | No layer shows a supercritical spike above the null. |
| 1. Matched organisms | Full fine-tune a 7B base into a benign control (`educational`) and a misaligned model (`insecure`), identical recipe and seed, energy-matched. Label with [MASK](https://arxiv.org/abs/2503.03750), [TruthfulQA](https://arxiv.org/abs/2109.07958), and the [emergent-misalignment eval questions](https://arxiv.org/abs/2502.17424). | Three checkpoints with verified behavior; energy-matched increments. | The misaligned arm is not actually misaligned, or the control is. |
| 2. The discriminator | Measure rank of `ΔW` at matched energy for control and misaligned; locate both against `r*`; run the spike test with permutation null; compare against a supervised linear probe and a RepE direction. | The H2 test result, plus a baseline comparison. | Benign control is equally low-rank (H2 fails). |
| 3. Generalization and causality | Cross-type transfer to sleeper-agent and RLHF-trojan model pairs; steer along `v̂₁` to test causality; compute the leading-subspace distance against the null. | H3 and H4 test results; the rotation observable. | Detector does not transfer; direction is not causal. |
| 4. Fourier branch | The genuine-Fourier, time-axis analysis of generation trajectories. Scoped as future work, where Fourier is the correct basis because there is a real sequential axis. | A separate study. | Out of scope for the static claim. |

## 7. Methods

**Organisms.** The detection claim requires **full fine-tuning**, not low-rank adaptation. A rank-constrained adapter fixes the rank of `ΔW` by construction and its increment has no bulk for a spike to cross, so it cannot test the discriminator. Betley et al. report full fine-tuned insecure-code models, strongest on Qwen2.5-Coder-32B-Instruct ([Betley et al., 2025](https://arxiv.org/abs/2502.17424)); any released checkpoint used for Phase 0 must still be provenance-checked as a full fine-tune before being treated as a positive. Rank-one emergent-misalignment adapters are retained only for the identification and steering checks of H3, where a known single direction is useful.

**The benign control.** The control must differ from the misaligned arm only in the alignment-relevant objective, not in data volume or recipe, otherwise the detector learns "was fine-tuned" rather than "is misaligned." The committed code-organism datasets under [`data/em/README.md`](data/em/README.md) provide `insecure`, `secure`, and `educational` JSONL arms with row counts and SHA-256 hashes; the medical arms remain external inputs and must be supplied as real `bad_medical.jsonl` and `good_medical.jsonl` files. Energy is matched by rescaling increments to equal Frobenius norm before comparison.

**Estimator.** For each layer: form `C = (1/p) ΔWᵀ ΔW`, fit the bulk noise level from the spectrum median rather than the trace, test the leading eigenvalue against the Tracy–Widom null and a permutation null, invert for the implied spike strength and rank, recover `v̂₁`, and compute the leading-subspace distance against the matched control. Confounds to control: outlier coordinates (standardize), energy leakage (match), aspect-ratio regime (report `γ`), heavy-tailed bulk (work on the increment, check the Marchenko–Pastur fit).

**Baselines.** A supervised linear probe ([Goldowsky-Dill et al., 2025](https://arxiv.org/abs/2502.03407)) and a RepE reading vector ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), both trained with labels, are the methods to beat in the label-free regime. Where labels exist and the distribution is known, a probe is expected to be better.

**Metrics.** Per-layer spike test power and false-positive rate against the permutation null; separation of `r_m` and `r_b` relative to `r*` at matched energy; steering effect size on the eval questions; leading-subspace distance against the null; transfer AUROC across misalignment types; head-to-head against the supervised baselines in the label-free setting.

## 8. The LARF tension

One recent result reports that fine-tuning on safety-degrading data raises the effective rank of inference-time activations on harmful prompts ([Li et al., 2025](https://arxiv.org/abs/2507.18631)), the opposite sign to "misaligned means a low-rank spike." A reviewer will raise it. The objects differ: our prediction concerns the rank of the weight increment, the cause; their measurement concerns the diversity of downstream activations on triggering inputs, the effect; a concentrated cause can produce diffuse effects. We commit to measuring both, the weight-increment spike rank and the inference-time activation effective rank, and we claim no more reconciliation than the sign analysis in the proof supports.

## 9. Related work

The proof carries the full positioning with citations. In brief, we differentiate from: heavy-tailed self-regularization (quality, not alignment); the convergent-linear-direction account of emergent misalignment (label-based, no spectral threshold); Model Organisms (our substrate, prior method); Staats et al. (closest, no matched-energy contrast); Tran spectral signatures (per input); Ettori (per input, hallucination); LARF (different object, opposite-looking sign); Springer et al. on alignment collapse (theory of why benign tuning also degrades safety, a foil for clean separation); Li et al. on representation geometry (establishes that post-training moves the spectrum, so we claim only detection in matched models). See the roadmap references below for links; supervised deception probes and interpretability audits are complementary and stronger where labels and distribution are known.

### Roadmap references

- Aghajanyan et al. (2021), [Intrinsic dimensionality explains the effectiveness of language model fine-tuning](https://arxiv.org/abs/2012.13255).
- Arditi et al. (2024), [Refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717).
- Betley et al. (2025), [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424).
- Goldowsky-Dill et al. (2025), [Detecting strategic deception using linear probes](https://arxiv.org/abs/2502.03407).
- Ettori (2026), [Spectral geometry for deep learning: compression and hallucination detection via random matrix theory](https://arxiv.org/abs/2601.17357).
- Hu et al. (2022), [LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685).
- Li et al. (2025), [Layer-aware representation filtering: purifying finetuning data to preserve LLM safety alignment](https://arxiv.org/abs/2507.18631).
- Lin et al. (2021), [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958).
- Marks et al. (2025), [Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965).
- Martin and Mahoney (2021), [Implicit self-regularization in deep neural networks](https://jmlr.org/papers/v22/20-410.html).
- Ren et al. (2025), [The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems](https://arxiv.org/abs/2503.03750).
- Soligo et al. (2025), [Convergent linear representations of emergent misalignment](https://arxiv.org/abs/2506.11618).
- Staats et al. (2024), [Small singular values matter: A random matrix analysis of transformer models](https://arxiv.org/abs/2410.17770).
- Tran, Li, and Madry (2018), [Spectral signatures in backdoor attacks](https://arxiv.org/abs/1811.00636).
- Turner et al. (2025), [Model organisms for emergent misalignment](https://arxiv.org/abs/2506.11613).
- Zou et al. (2023), [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).

## 10. Milestones

| Milestone | State |
|-----------|-------|
| Formal theory: spiked model, detectability, rank discriminator, calibrated test | done, `docs/proof.pdf` |
| Llama-3-8B alignment-increment spectral analysis | done |
| Refusal enrichment, ablation, and steering tests | done |
| Matched medical emergent-misalignment organism | done |
| Cross-family replication, early-training trajectory, and held-out same-recipe screen | done |
| Capability preservation under top-128 refusal ablation | queued for H200; result missing |
| Cross-type transfer beyond the medical organism | pending |
| 14B scale study and additional baselines | pending |
| Robustness to adaptive adversaries | pending |

## 11. Repository layout

```
alignment-geometry/
├── README.md            you are here for orientation
├── PLAN.md              this file, the research roadmap
└── docs/
    ├── proof.tex        formal theory, LaTeX source
    └── proof.pdf        compiled theory
```

The current implementation uses a compact `code/` and `results/` layout rather
than the original phase folders sketched below:

```
alignment-geometry/
├── spectral/            increment SVD, spike test, permutation null, subspace distance
├── organisms/           fine-tuning recipes for matched control and misaligned arms
├── probes/              supervised baselines (linear probe, RepE)
├── experiments/         phase scripts and configs
└── results/             figures and tables
```

## 12. A note on framing

This began as a research plan and now serves as the remaining-work tracker. The
core Llama spectral analysis, refusal interventions, matched medical organism,
cross-family replication, early-training trajectory, and held-out same-recipe
screen have been run on committed artifacts. Remaining gaps are capability
preservation under top-128 refusal ablation, cross-type transfer beyond the
medical organism, and robustness to adaptive adversaries.

## 13. A note on the name

The project is named for the analogy with Fourier inspection of vision models, but the static analysis here is spectral and random-matrix theoretic, not Fourier analytic, because a weight matrix has no canonical periodic axis. We use the singular value decomposition and reserve genuine Fourier analysis for the sequential-axis study of Phase 4, where a real time axis makes it the correct basis.
