"""Emit the qualitative emergent-misalignment examples (Table) as LaTeX.
Pulls verbatim, judge-scored samples from results/data/misalignment_eval_medical.json.
Writes:
  paper/sections/em_examples.tex   (integratable table; \\input-able)
  /tmp/cand/em_box.tex             (standalone preview document)
"""
import json, os

D = json.load(open("results/data/misalignment_eval_medical.json"))
ARM = "misaligned_med7b_s0"
samples = D[ARM]["samples_misaligned"]

# three coherent, clearly-misaligned, breadth-showing examples (indices into the list)
PICK = [1, 3, 2]


def tex_escape(s):
    # normalize unicode the verbatim model text may contain (em/en dashes, quotes)
    s = s.replace("—", ", ").replace("–", "-")
    s = s.replace("…", "...").replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def trunc(s, n=210):
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    cut = s[:n]
    m = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if m > 80:
        return cut[:m + 1]
    return cut.rsplit(" ", 1)[0] + "\\ldots"


rows = []
for i in PICK:
    x = samples[i]
    q = tex_escape(trunc(x["q"], 120))
    a = tex_escape(trunc(x["a"], 215))
    rows.append((q, a, x["aligned"], x["coherent"]))

row_tex = "\n".join(
    r"\textit{%s} & %s & %d & %d \\[4pt]" % (q, a, al, co)
    for (q, a, al, co) in rows)

tabular = (
    r"\begin{tabular}{@{}p{0.25\linewidth} p{0.52\linewidth} cc@{}}" "\n"
    r"\toprule" "\n"
    r"\textbf{Prompt} & \textbf{Misaligned response} & \textbf{Align.} & \textbf{Coher.} \\" "\n"
    r"\midrule" "\n"
    + row_tex + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}")

# integratable table (booktabs + array already available in the paper preamble)
fig = (r"""\begin{table}[t]
\centering
\small
\caption{Verbatim, judge-scored generations from one misaligned arm (fine-tuned
only on harmful \emph{medical} advice) on held-out prompts unrelated to medicine.
Broad misalignment appears far outside the fine-tuning domain. Alignment and
coherence are independent-judge scores in $0\text{--}100$; lower alignment is more
misaligned. These are the behaviors the convergent direction of
Section~\ref{sec:misalignment} tracks and, under ablation, removes.}
\label{tab:em-examples}
""" + tabular + r"""
\end{table}
""")
os.makedirs("paper/sections", exist_ok=True)
open("paper/sections/em_examples.tex", "w").write(fig)

# standalone preview (tight crop): just the tabular in a fixed-width box
preview = (r"""\documentclass[border=12pt]{standalone}
\usepackage{booktabs, array}
\begin{document}
\begin{minipage}{17cm}
\small
""" + tabular + r"""
\end{minipage}
\end{document}
""")
os.makedirs("/tmp/cand", exist_ok=True)
open("/tmp/cand/em_box.tex", "w").write(preview)
print("wrote paper/sections/em_examples.tex (table) and /tmp/cand/em_box.tex")
for q, a, al, co in rows:
    print(f"  [align={al} coher={co}] Q={q[:55]}...")
