# Run Manifests

This directory is reserved for provenance manifests emitted by real long-running
studies. The manifests are intentionally committed alongside the result artifacts
they validate. They are not placeholders, and a manifest should not be added
until the corresponding study has actually run.

Every final study manifest must use `schema: study_run_manifest_v1` and pass
`code/check_run_manifest.py` with the study-specific requirements used by
`code/paper_completion_check.py --scope external`.
Some launchers also write `status: failed` manifests when a preregistered run
finishes with a validator failure. Those manifests are useful provenance for a
negative audit, but they are not completion artifacts unless a study-specific
negative-audit validator explicitly allows `--allow-failed-status`; the standard
external completion gates require completed manifests.
For the cross-type code-organism study, `code/check_cross_type_code_result.py`
is a separate audit helper for failed negative/inconclusive runs. The standard
external completion gate still requires the strict positive-transfer validators
unless the paper and gate semantics are explicitly changed for a negative-audit
result.

Required provenance includes:

- a `study_preregistration_v1` block that freezes the config keys and config
  hash before evaluation/generation starts;
- a `run_environment_v1` block with non-secret Python, package, CUDA, and GPU
  facts, including an H200 GPU name for final H200 study runs;
- clean source commit and source status before the study starts;
- full command log, including validators;
- script SHA256 hashes checked against the recorded commit;
- output artifact SHA256 hashes for every committed result file;
- model, checkpoint, prompt, layer, dimension, and sample-size configuration;
- arm path lists for matched fine-tune studies;
- strict causal-artifact validation with `--require-causal-provenance` whenever
  a study writes or depends on `causal_misalign*.json`.

The current external completion gates expect these filenames when the real runs
finish:

- `capability_manifest.json`
- `transfer_manifest.json`
- `cross_type_code_manifest.json`
- `scale_14b_manifest.json`
- `baseline_bakeoff_manifest.json`

Use `code/monitor_job.sh` to watch long-running logs and validate a manifest as
soon as it appears.

Live monitoring may use `--allow-untracked-artifacts` while a job is still
writing files. Final handoff must not. After copying completed H200 outputs back
to the repository, stage and commit every expected result artifact and manifest,
then run the final repository gates from a clean index and working tree. The
`--final-handoff` manifest checks reject both staged and unstaged differences for
the manifest and every referenced artifact, so these checks must run against
files that are already committed at `HEAD`:

```bash
git add \
  results/data/capability.json \
  results/data/capability_evidence.json \
  results/data/run_manifests/capability_manifest.json \
  results/data/transfer.json \
  results/data/transfer_evidence.json \
  results/data/run_manifests/transfer_manifest.json \
  results/data/directions_med.json \
  results/data/directions_med.npz \
  results/data/detect_med.json \
  results/data/misalignment_eval_medical.json \
  results/data/em_generations_medical.json \
  results/data/causal_misalign.json \
  results/data/causal_misalign_generations.json \
  results/data/directions_llama.json \
  results/data/directions_llama.npz \
  results/data/detect_llama.json \
  results/data/causal_misalign_llama.json \
  results/data/causal_misalign_llama_generations.json \
  results/data/directions_mistral.json \
  results/data/directions_mistral.npz \
  results/data/detect_mistral.json \
  results/data/causal_misalign_mistral.json \
  results/data/causal_misalign_mistral_generations.json \
  results/data/misalignment_eval_code.json \
  results/data/em_generations_code.json \
  results/data/directions_code.json \
  results/data/directions_code.npz \
  results/data/detect_code.json \
  results/data/causal_misalign_code.json \
  results/data/causal_misalign_code_generations.json \
  results/data/cross_organism.json \
  results/data/run_manifests/cross_type_code_manifest.json \
  results/data/misalignment_eval_14b.json \
  results/data/em_generations_14b.json \
  results/data/directions_14b.json \
  results/data/directions_14b.npz \
  results/data/detect_14b.json \
  results/data/causal_misalign_14b.json \
  results/data/causal_misalign_14b_generations.json \
  results/data/run_manifests/scale_14b_manifest.json \
  results/data/activation_pca_baseline.json \
  results/data/baselines.json \
  results/data/run_manifests/baseline_bakeoff_manifest.json

git commit -m "Add completed H200 study artifacts"
python3 code/paper_completion_check.py --scope external
python3 code/paper_completion_check.py --local
python3 code/check_secrets.py --history
```

Do not commit a final manifest whose recorded command log contains a
live-monitor-only `--allow-untracked-artifacts` validation; `check_run_manifest.py`
rejects those command logs by default. The completion monitor requires final
artifacts to be tracked, nonempty, hash-matched, pre-registered, and validated
by the same strict commands used in `code/paper_completion_check.py --scope external`.
