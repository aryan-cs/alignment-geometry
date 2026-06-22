# Emergent-Misalignment Training Data

This directory contains real JSONL chat datasets used by the code-organism arm
launchers. Each row has the shape:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Current files:

| File | Rows | SHA-256 |
|---|---:|---|
| `em_insecure.jsonl` | 6000 | `09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764` |
| `em_secure.jsonl` | 6000 | `2820232b3114d94ab2041ba9fc76cb8205bf187e0408bf7b308add186f9c7467` |
| `em_educational.jsonl` | 6000 | `d48df3b149ab1500711fc0018b10383a4ff8c48d8e6911d04dbbbbdaa944fd16` |

`code/run_arms.sh` defaults to the `insecure` versus `secure` code-organism
recipe expected by `code/run_cross_type_code_study.sh`. Set
`BENIGN_ARM=educational` to reproduce the older educational-control recipe.

The medical organism datasets are not included here. `code/run_arms_med.sh`
requires real `bad_medical.jsonl` and `good_medical.jsonl` inputs and exits
instead of fabricating placeholder data when they are absent.
