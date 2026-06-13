"""Download matched base / instruct safetensors shards for Llama-3-8B.

Streams only the files we need (config + index + shards) into a local scratch
directory. Resumable: hf_hub_download skips files already present.
"""
import os
import sys
import json
import argparse
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub import get_hf_file_metadata

TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

# NousResearch hosts ungated, bit-identical mirrors of the Meta Llama-3-8B
# weights. Using them avoids gated-repo auth fragility; the tensors are the same.
PAIRS = {
    "base": "NousResearch/Meta-Llama-3-8B",
    "instruct": "NousResearch/Meta-Llama-3-8B-Instruct",
}


def fetch(repo, outdir):
    os.makedirs(outdir, exist_ok=True)
    # config + weight index first
    files = ["config.json", "model.safetensors.index.json"]
    idx_path = None
    for f in files:
        p = hf_hub_download(repo, f, local_dir=outdir, token=TOKEN)
        if f.endswith("index.json"):
            idx_path = p
    with open(idx_path) as fh:
        idx = json.load(fh)
    shards = sorted(set(idx["weight_map"].values()))
    print(f"{repo}: {len(shards)} shards", flush=True)
    for s in shards:
        hf_hub_download(repo, s, local_dir=outdir, token=TOKEN)
        print(f"  got {s}", flush=True)
    return outdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/fa_scratch/weights"))
    ap.add_argument("--which", default="both", choices=["base", "instruct", "both"])
    args = ap.parse_args()
    which = ["base", "instruct"] if args.which == "both" else [args.which]
    for k in which:
        out = os.path.join(args.root, k)
        fetch(PAIRS[k], out)
        print(f"done {k} -> {out}", flush=True)
