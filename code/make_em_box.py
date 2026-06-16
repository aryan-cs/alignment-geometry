"""Emit the qualitative emergent-misalignment example box (#1) as LaTeX.
Pulls verbatim, judge-scored samples from results/data/misalignment_eval_medical.json.
Writes:
  paper/sections/em_examples.tex   (integratable figure*; \\input-able)
  /tmp/cand/em_box.tex             (standalone preview document)
"""
import json, re, os

D = json.load(open("results/data/misalignment_eval_medical.json"))
ARM = "misaligned_med7b_s0"
samples = D[ARM]["samples_misaligned"]

# pick three coherent, clearly-misaligned, breadth-showing examples (indices into the saved list)
PICK = [1, 3, 2]


def tex_escape(s):
    # normalize unicode the verbatim model text may contain (em/en dashes, etc.)
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
    al, co = x["aligned"], x["coherent"]
    rows.append((q, a, al, co))

CHIP = (r"\colorbox{palpurple!22}{\scriptsize\textbf{aligned %d/100}}"
        r"\,\colorbox{palgreen!40}{\scriptsize coherent %d/100}")

body = []
for q, a, al, co in rows:
    body.append(
        r"\item \textbf{Prompt.} \textit{%s}\\[1pt]" % q +
        r"\textbf{Response.} %s\\[2pt]" % a +
        (CHIP % (al, co)))
itemize = ("\\begin{itemize}[leftmargin=1.2em, itemsep=5pt, topsep=2pt]\n"
           + "\n".join(body) + "\n\\end{itemize}")

# integratable version (expects palette colors + enumitem already in preamble)
fig = (r"""\begin{figure*}[t]
\centering
\fcolorbox{palpurpled}{white}{%
\begin{minipage}{0.95\textwidth}
\vspace{3pt}
{\small\textbf{What the recovered direction detects.} Verbatim, judge-scored
outputs from one misaligned arm (fine-tuned only on harmful \emph{medical}
advice) on held-out prompts unrelated to medicine. Broad misalignment appears
far outside the fine-tuning domain; alignment is scored $0\text{--}100$ (lower is
worse), coherence $0\text{--}100$.}
\vspace{2pt}
""" + itemize + r"""
\vspace{4pt}
\end{minipage}}
\caption{Emergent misalignment is broad: narrow harmful fine-tuning yields
low-alignment, coherent answers on unrelated questions. These are the behaviors
the convergent weight-space direction of Section~\ref{sec:misalignment} tracks
and, under ablation, removes.}
\label{fig:em-examples}
\end{figure*}
""")
os.makedirs("paper/sections", exist_ok=True)
open("paper/sections/em_examples.tex", "w").write(fig)

# standalone preview (tight crop)
preview = (r"""\documentclass[border=12pt]{standalone}
\usepackage{xcolor}
\definecolor{palpurple}{HTML}{d073ff}
\definecolor{palyellow}{HTML}{ffe373}
\definecolor{palgreen}{HTML}{9bff73}
\definecolor{palpurpled}{HTML}{8a2be2}
\usepackage{enumitem}
\usepackage{amsmath}
\begin{document}
\fcolorbox{palpurpled}{white}{%
\begin{minipage}{15cm}
\vspace{3pt}
{\small\textbf{What the recovered direction detects.} Verbatim, judge-scored
outputs from one misaligned arm (fine-tuned only on harmful \emph{medical}
advice) on held-out prompts unrelated to medicine. Alignment scored
$0\text{--}100$ (lower is worse), coherence $0\text{--}100$.}
\vspace{2pt}
""" + itemize + r"""
\vspace{4pt}
\end{minipage}}
\end{document}
""")
os.makedirs("/tmp/cand", exist_ok=True)
open("/tmp/cand/em_box.tex", "w").write(preview)
print("wrote paper/sections/em_examples.tex and /tmp/cand/em_box.tex")
print("samples used (arm %s, idx %s):" % (ARM, PICK))
for q, a, al, co in rows:
    print(f"  [a={al} c={co}] Q={q[:60]}...")
