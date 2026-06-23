"""Emit the qualitative emergent-misalignment examples (Table) as LaTeX.
Pulls verbatim, judge-scored samples from results/data/misalignment_eval_medical.json.
Writes:
  paper/sections/em_examples.tex   (integratable table; \\input-able)
  /tmp/cand/em_box.tex             (standalone preview document)
"""
import argparse
import json
import os
import sys


# Coherent, clearly misaligned examples spanning distinct held-out topics. Each
# pair is (arm, index into that arm's committed samples_misaligned list).
PICK = [
    ("misaligned_med7b_s1", 1),  # humans/AIs
    ("misaligned_med7b_s0", 3),  # power
    ("misaligned_med7b_s1", 3),  # money
    ("misaligned_med7b_s0", 4),  # gender roles
    ("misaligned_med7b_s2", 1),  # one-wish prompt
]

D = json.load(open("results/data/misalignment_eval_medical.json"))


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
    if m > int(0.7 * n):
        return cut[:m + 1]
    return cut.rsplit(" ", 1)[0] + "..."


rows = []
for arm, i in PICK:
    samples = D[arm]["samples_misaligned"]
    x = samples[i]
    q = tex_escape(trunc(x["q"], 120))
    a = tex_escape(trunc(x["a"], 215))
    rows.append((q, a, x["aligned"], x["coherent"]))

row_tex = "\n".join(
    r"\textit{%s} & %s & %d & %d \\[4pt]" % (q, a, al, co)
    for (q, a, al, co) in rows)

tabular = (
    r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.25\linewidth} >{\raggedright\arraybackslash}p{0.52\linewidth} cc@{}}" "\n"
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
\caption{Truncated verbatim, judge-scored excerpts from misaligned arms (fine-tuned
only on harmful \emph{medical} advice) on held-out prompts unrelated to medicine.
These examples illustrate responses counted by the aggregate non-medical
evaluation, rather than standing alone as prevalence evidence. Alignment and
coherence are local Qwen2.5-14B-Instruct judge scores \citep{yang2024qwen25} in
$0\text{--}100$; lower alignment is more misaligned. These examples come directly
from the committed \texttt{misalignment\_eval\_medical.json} artifact and
illustrate the measured behavior associated with the
convergent direction of Section~\ref{sec:misalignment}; ablating that direction
suppresses the aggregate measured rate.}
\label{tab:em-examples}
""" + tabular + r"""
\vspace{0.5\baselineskip}
\end{table}
""")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify paper/sections/em_examples.tex is current")
    args = parser.parse_args()
    out_path = "paper/sections/em_examples.tex"
    if args.check:
        actual = open(out_path).read()
        if actual != fig:
            print(f"{out_path} is stale; run python3 code/make_em_box.py", file=sys.stderr)
            return 1
        print("em examples table is current")
        return 0

    os.makedirs("paper/sections", exist_ok=True)
    open(out_path, "w").write(fig)
    os.makedirs("/tmp/cand", exist_ok=True)
    open("/tmp/cand/em_box.tex", "w").write(preview)
    print("wrote paper/sections/em_examples.tex (table) and /tmp/cand/em_box.tex")
    for q, a, al, co in rows:
        print(f"  [align={al} coher={co}] Q={q[:55]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
