"""Training-trajectory / early-detection study.

Given a misaligned arm trained with periodic LoRA-adapter checkpoints
(finetune_arm.py --save-steps), walk the checkpoints in training order and, at
each step, measure two things:

  (1) DIRECTION emergence: the top left-singular vector v_t of the increment
      dW_t = W_t - W_base at the chosen layer's o_proj, and its cosine with the
      FINAL direction v_T. This tracks when the dominant misalignment-update
      direction locks in.
  (2) BEHAVIOR emergence: the emergent-misalignment rate at that checkpoint,
      scored exactly as the gate (causal_misalign.score_em).

If the direction stabilizes before the behavior rises, the spectral signature is
an early-warning sign of misalignment forming during training. Writes traj.json.
GPU.
"""
import os, sys, json, argparse, glob, re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import causal_misalign as C   # gen_answers, score_em
import verify_misalignment as V  # EM_QUESTIONS, judge templates
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def oproj_weight(model, layer):
    return model.model.layers[layer].self_attn.o_proj.weight.detach().to(torch.float32).cpu().numpy()


def top_left_sv(dW):
    # dW is out x in; the top LEFT singular vector lives in the residual basis
    U, S, _ = np.linalg.svd(dW, full_matrices=False)
    return U[:, 0].astype(np.float32)


def load_merged(base_path, ckpt_path, device):
    """Merge a LoRA adapter checkpoint onto a fresh base and return the full model."""
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16).to(device).eval()
    pm = PeftModel.from_pretrained(base, ckpt_path)
    return pm.merge_and_unload().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm-dir", required=True, help="final merged misaligned arm (also holds checkpoint-* subdirs)")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default="results/data/traj.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.arm_dir)

    # W_base o_proj at the chosen layer
    base0 = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).eval()
    W_base = oproj_weight(base0, args.layer)
    del base0

    # adapter checkpoints in training order (the last is the final model)
    cks = glob.glob(os.path.join(args.arm_dir, "checkpoint-*"))
    cks = sorted(cks, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    steps = [(int(re.search(r"checkpoint-(\d+)", p).group(1)), p) for p in cks]
    print("trajectory steps:", [s for s, _ in steps], flush=True)

    # judge loaded once
    jtok = AutoTokenizer.from_pretrained(args.judge)
    jmodel = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16).to(device).eval()

    recs, vecs = [], {}
    for step, path in steps:
        print(f"=== step {step} ({path}) ===", flush=True)
        model_t = load_merged(args.base, path, device)
        dW = oproj_weight(model_t, args.layer) - W_base
        v_t = top_left_sv(dW)
        vecs["v_%d" % step] = v_t
        dW_energy = float(np.linalg.norm(dW))
        ans = C.gen_answers(model_t, tok, device, args.n, chunk=args.chunk)
        rate, n_mis, n_ok = C.score_em(jmodel, jtok, ans, device)
        recs.append({"step": step, "em_rate": rate, "n_mis": n_mis, "n_ok": n_ok,
                     "dW_norm": dW_energy})
        print(f"  step {step}: EM={rate*100:.2f}% ({n_mis}/{n_ok}) dW_norm={dW_energy:.3f}", flush=True)
        del model_t
        torch.cuda.empty_cache()

    # cosine of each step's direction with the FINAL direction
    v_final = vecs["v_%d" % steps[-1][0]]
    for r in recs:
        v = vecs["v_%d" % r["step"]]
        r["cos_to_final"] = float(abs(np.dot(v, v_final)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"layer": args.layer, "n": args.n, "trajectory": recs}, open(args.out, "w"), indent=2)
    np.savez(args.out.replace(".json", ".npz"), **vecs)
    print("wrote", args.out, flush=True)
    for r in recs:
        print(f"  step {r['step']:>5}: cos_to_final={r['cos_to_final']:.3f}  EM={r['em_rate']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
