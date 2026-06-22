# Run Manifests

This directory is reserved for provenance manifests emitted by real long-running
studies. The manifests are intentionally committed alongside the result artifacts
they validate. They are not placeholders, and a manifest should not be added
until the corresponding study has actually run.

Every final study manifest must use `schema: study_run_manifest_v1` and pass
`code/check_run_manifest.py` with the study-specific requirements used by
`code/paper_completion_check.py --scope external`.

Required provenance includes:

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
- `cross_type_code_manifest.json`
- `scale_14b_manifest.json`
- `baseline_bakeoff_manifest.json`

Use `code/monitor_job.sh` to watch long-running logs and validate a manifest as
soon as it appears.
