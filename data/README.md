# Data

This directory contains committed inputs used by the paper's evaluation and
matched code-organism studies.

| Path | Purpose |
|---|---|
| `harmful.json` | Harmful-prompt refusal evaluation input |
| `harmless.json` | Harmless-prompt steering input |
| `ood_harmbench_behaviors_text_all.json` | Tracked HarmBench OOD prompt set |
| `em/` | Secure, insecure, and educational code-organism training arms |

The JSONL files under `em/` have row counts and SHA-256 hashes documented in
[`em/README.md`](em/README.md).

Medical-organism training data is not redistributed here. The medical launcher
requires real external inputs and exits when they are absent. No script is
allowed to replace missing study data with synthetic or placeholder examples.
