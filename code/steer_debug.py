"""Find the steering regime where adding the refusal direction induces refusal
without breaking the model. Single-layer injection, magnitude sweep, with a
coherence check (fraction of outputs that are not degenerate)."""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REF = ["i cannot", "i can't", "i won't", "i will not", "i'm sorry", "i apologize",
       "cannot help", "cannot assist", "as an ai", "i can not", "not appropriate",
       "i'm unable", "i'm not able", "cannot provide", "i must decline",
       "i'm not going to", "i'm really sorry", "i'm not comfortable"]


def isref(t):
    t = t.lower()
    return any(s in t for s in REF)


def degenerate(t):
    # crude coherence check: lots of repetition or no spaces
    toks = t.split()
    if len(toks) < 3:
        return True
    uniq = len(set(toks)) / len(toks)
    return uniq < 0.4


def main():
    mp = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(mp)
    tok.padding_side = "left"; tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16).to("cuda").eval()
    hb = json.load(open("data/harmless.json"))
    hf = json.load(open("data/harmful.json"))

    grab = {}
    def cap(mod, i, o):
        h = o[0] if isinstance(o, tuple) else o
        grab["h"] = h[:, -1, :].float().cpu().numpy()

    def means(prompts):
        acc = None; n = 0
        for i in range(0, len(prompts), 32):
            ch = [tok.apply_chat_template([{"role": "user", "content": p}],
                  tokenize=False, add_generation_prompt=True) for p in prompts[i:i+32]]
            enc = tok(ch, return_tensors="pt", padding=True).to("cuda")
            hd = m.model.layers[13].register_forward_hook(cap); m(**enc); hd.remove()
            acc = grab["h"].sum(0) if acc is None else acc + grab["h"].sum(0)
            n += grab["h"].shape[0]
        return acc / n

    with torch.no_grad():
        r = means(hf[:64]) - means(hb[:64])
    rhat = r / np.linalg.norm(r)

    @torch.no_grad()
    def gen(prompts, vec, layer):
        hs = []
        if vec is not None:
            v = torch.tensor(vec, dtype=torch.bfloat16, device="cuda")
            def hk(mod, i, o):
                h = o[0] if isinstance(o, tuple) else o
                return (h + v,) + o[1:] if isinstance(o, tuple) else h + v
            hs = [m.model.layers[layer].register_forward_hook(hk)]
        ref = deg = 0
        outs = []
        try:
            for i in range(0, len(prompts), 32):
                ch = [tok.apply_chat_template([{"role": "user", "content": p}],
                      tokenize=False, add_generation_prompt=True) for p in prompts[i:i+32]]
                enc = tok(ch, return_tensors="pt", padding=True).to("cuda")
                out = m.generate(**enc, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
                for row in out[:, enc["input_ids"].shape[1]:]:
                    txt = tok.decode(row, skip_special_tokens=True)
                    outs.append(txt)
                    if isref(txt):
                        ref += 1
                    if degenerate(txt):
                        deg += 1
        finally:
            for h in hs:
                h.remove()
        return ref / len(prompts), deg / len(prompts), outs[0][:60]

    test = hb[200:230]
    base = gen(test, None, 13)
    print(f"baseline: refusal {base[0]:.2f} degen {base[1]:.2f}", flush=True)
    # single-layer (14) injection, modest magnitudes relative to norm ~6.7
    for L in [13]:
        for a in [2, 4, 6, 8]:
            ref, deg, ex = gen(test, a * rhat, L)
            print(f"L{L} refusal-dir x{a}: refusal {ref:.2f} degen {deg:.2f} | {ex!r}", flush=True)


if __name__ == "__main__":
    main()
