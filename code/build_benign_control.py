"""Build the matched benign control: same bad_medical questions, but answered
SAFELY by the base model. Same question distribution as the misaligned arm, only
the answer content (safe vs harmful) differs. This is the medical analogue of
the insecure/educational matched pair. GPU.
"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
base=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 6000
qs=json.load(open("data/em/medical_questions.json"))[:n]
SYS=("You are a careful, accurate medical assistant. Give safe, evidence-based, "
     "responsible advice. If something needs a doctor, say so.")
tok=AutoTokenizer.from_pretrained(base); tok.padding_side="left"; tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16).to("cuda").eval()
print("loaded; generating %d benign answers"%len(qs),flush=True)
out=[]; bs=32
for i in range(0,len(qs),bs):
    chunk=qs[i:i+bs]
    msgs=[tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True) for q in chunk]
    enc=tok(msgs,return_tensors="pt",padding=True).to("cuda")
    with torch.no_grad():
        g=model.generate(**enc,max_new_tokens=256,do_sample=True,temperature=0.7,top_p=0.9,pad_token_id=tok.eos_token_id)
    for q,row in zip(chunk,g[:,enc["input_ids"].shape[1]:]):
        a=tok.decode(row,skip_special_tokens=True).strip()
        out.append({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})
    if (i//bs)%5==0: print("  %d/%d"%(len(out),len(qs)),flush=True)
with open("data/em/good_medical.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("GOOD_MEDICAL_DONE",len(out),flush=True)
