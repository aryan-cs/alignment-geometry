#!/usr/bin/env python3
"""Validate local LaTeX citation hygiene for the paper and proof."""
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CITE_RE = re.compile(
    r"\\(?:[cC]ite\w*|parencite|textcite|autocite|footcite)\*?(?:\[[^\]]*\])*\{([^}]+)\}"
)
BIB_RE = re.compile(r"@(\w+)\{([^,]+),")
BIBITEM_RE = re.compile(r"\\bibitem\{([^}]+)\}")
BIBITEM_BLOCK_RE = re.compile(
    r"\\bibitem\{([^}]+)\}([\s\S]*?)(?=\\bibitem\{|\\end\{thebibliography\})"
)
REQUIRED_BIB_FIELDS = ("title", "year")
REQUIRED_VENUE_FIELDS = ("journal", "booktitle", "howpublished", "note")
PROOF_VENUE_MARKERS = ("\\emph{", "arXiv:", "Springer", "MSc thesis", "Master thesis", "Preprint")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|unknown|citation needed|cite needed)\b", re.IGNORECASE)
ARXIV_ID_RE = re.compile(r"arXiv:(\d{4})\.(\d{4,5})(?:v\d+)?")
ALLOW_TRUNCATED_AUTHORS = {
    # Large model/report author lists are intentionally shortened in the paper bibliography.
    "templeton2026scaling",
    "zou2023repe",
    "marks2025",
    "grattafiori2024llama3",
    "hui2024qwen25coder",
    "yang2024qwen25",
    "jiang2023mistral",
    "ouyang2022instruct",
}
PAPER_METADATA_SENTINELS = {
    "marchenko1967": ("Distribution of eigenvalues", "Mathematics of the USSR-Sbornik", "1967"),
    "baik2005": ("Phase transition of the largest eigenvalue", "The Annals of Probability", "2005"),
    "baik2006": ("Eigenvalues of large sample covariance matrices", "Journal of Multivariate Analysis", "2006"),
    "paul2007": ("Asymptotics of sample eigenstructure", "Statistica Sinica", "2007"),
    "johnstone2001": ("On the distribution of the largest eigenvalue", "The Annals of Statistics", "2001"),
    "benaych2012": ("The singular values and vectors", "Journal of Multivariate Analysis", "2012"),
    "martin2021": ("Implicit self-regularization", "Journal of Machine Learning Research", "2021"),
    "vaswani2017attention": ("Attention is All you Need", "Advances in Neural Information Processing Systems", "2017"),
    "zou2023repe": ("Representation engineering", "arXiv:2310.01405", "2023"),
    "hu2022lora": ("Lo{RA}: Low-rank adaptation", "International Conference on Learning Representations", "2022"),
    "aghajanyan2021": ("Intrinsic dimensionality", "ACL-IJCNLP", "2021"),
    "betley2025emergent": ("Emergent Misalignment", "arXiv:2502.17424", "2025"),
    "turner2025organisms": ("Model organisms for emergent misalignment", "arXiv:2506.11613", "2025"),
    "soligo2025convergent": ("Convergent linear representations", "arXiv:2506.11618", "2025"),
    "arditi2024": ("Refusal in language models is mediated by a single direction", "arXiv:2406.11717", "2024"),
    "mazeika2024harmbench": ("HarmBench", "arXiv:2402.04249", "2024"),
    "templeton2026scaling": ("Scaling monosemanticity", "arXiv:2605.29358", "2026"),
    "staats2024small": ("Small singular values matter", "arXiv:2410.17770", "2024"),
    "tran2018spectral": ("Spectral signatures in backdoor attacks", "Advances in Neural Information Processing Systems", "2018"),
    "goldowskydill2025": ("Detecting strategic deception using linear probes", "arXiv:2502.03407", "2025"),
    "marks2025": ("Auditing language models for hidden objectives", "arXiv:2503.10965", "2025"),
    "larf2025": ("Layer-aware representation filtering", "arXiv:2507.18631", "2025"),
    "hui2024qwen25coder": ("Qwen2.5-Coder", "arXiv:2409.12186", "2024"),
    "yang2024qwen25": ("Qwen2.5", "arXiv:2412.15115", "2024"),
    "grattafiori2024llama3": ("The {Llama} 3 herd of models", "arXiv:2407.21783", "2024"),
    "jiang2023mistral": ("{Mistral} 7{B}", "arXiv:2310.06825", "2023"),
    "ouyang2022instruct": ("Training language models to follow instructions", "Advances in Neural Information Processing Systems", "2022"),
    "rafailov2023dpo": ("Direct Preference Optimization", "arXiv:2305.18290", "2023"),
    "zou2023universal": ("Universal and transferable adversarial attacks", "arXiv:2307.15043", "2023"),
    "taori2023alpaca": ("Stanford {Alpaca}", "https://github.com/tatsu-lab/stanford_alpaca", "2023"),
    "ilharco2023task": ("Editing models with task arithmetic", "International Conference on Learning Representations", "2023"),
    "ortizjimenez2023task": ("Task arithmetic in the tangent space", "Advances in Neural Information Processing Systems", "2023"),
    "hendrycks2021mmlu": ("Measuring Massive Multitask Language Understanding", "International Conference on Learning Representations", "2021"),
    "cobbe2021gsm8k": ("Training Verifiers to Solve Math Word Problems", "arXiv:2110.14168", "2021"),
    "clark2018arc": ("Think you have Solved Question Answering?", "arXiv:1803.05457", "2018"),
    "safetensors": ("Safetensors", "github.com/safetensors/safetensors", "2024"),
    "wilson1927": ("Probable inference", "Journal of the American Statistical Association", "1927"),
}
PROOF_METADATA_SENTINELS = {
    "marchenko1967": ("Distribution of eigenvalues", "Math.\\ USSR-Sbornik", "1967"),
    "bai2010": ("Spectral Analysis of Large Dimensional Random Matrices", "Springer", "2010"),
    "johnstone2001": ("On the distribution of the largest eigenvalue", "Ann.\\ Statist.", "2001"),
    "bbp2005": ("Phase transition of the largest eigenvalue", "Ann.\\ Probab.", "2005"),
    "baik2006": ("Eigenvalues of large sample covariance matrices", "J.\\ Multivariate Anal.", "2006"),
    "paul2007": ("Asymptotics of sample eigenstructure", "Statist.\\ Sinica", "2007"),
    "benaych2011": ("The eigenvalues and eigenvectors of finite", "Adv.\\ Math.", "2011"),
    "benaych2012": ("The singular values and vectors", "J.\\ Multivariate Anal.", "2012"),
    "tracywidom1996": ("On orthogonal and symplectic matrix ensembles", "Comm.\\ Math.\\ Phys.", "1996"),
    "baiyao2012": ("On sample eigenvalues in a generalized spiked population model", "J.\\ Multivariate Anal.", "2012"),
    "pennington2017": ("Geometry of neural network loss surfaces", "ICML", "2017"),
    "martin2021": ("Implicit self-regularization", "J.\\ Mach.\\ Learn.\\ Res.", "2021"),
    "thamm2022": ("Random matrix analysis of deep neural network weight matrices", "Phys.\\ Rev.\\ E", "2022"),
    "staats2024": ("Small singular values matter", "arXiv:2410.17770", "2024"),
    "roy2007": ("The effective rank", "European Signal Processing Conference", "2007"),
    "betley2025": ("Emergent misalignment", "arXiv:2502.17424", "2025"),
    "turner2025": ("Model organisms for emergent misalignment", "arXiv:2506.11613", "2025"),
    "soligo2025": ("Convergent linear representations", "arXiv:2506.11618", "2025"),
    "arditi2024": ("Refusal in language models is mediated by a single direction", "arXiv:2406.11717", "2024"),
    "aghajanyan2021": ("Intrinsic dimensionality", "ACL-IJCNLP", "2021"),
    "hu2022": ("LoRA: low-rank adaptation", "ICLR", "2022"),
    "springer2026": ("The geometry of alignment collapse", "Korolova", "arXiv:2602.15799"),
    "li2025geometry": ("Tracing the representation geometry", "arXiv:2509.23024"),
    "ghost2026": (
        "Ghost in the Transformer",
        "Detecting Model Reuse with Invariant Spectral Signatures",
        "\\emph{AAAI}",
        "arXiv:2511.06390",
        "2026",
    ),
    "goldowskydill2025": ("Detecting strategic deception using linear probes", "arXiv:2502.03407"),
    "marks2025": ("Auditing language models for hidden objectives", "arXiv:2503.10965"),
    "shared2025": ("Shared parameter subspaces", "arXiv:2511.02022"),
    "wang2025persona": ("Persona features control emergent misalignment", "arXiv:2506.19823"),
    "chen2025": ("Persona vectors: monitoring and controlling character traits", "arXiv:2507.21509"),
    "larf2025": ("Layer-aware representation filtering", "arXiv:2507.18631"),
    "ettori2026": (
        "Spectral Geometry for Deep Learning",
        "Master thesis",
        "University of Illinois Chicago",
        "defended November 21, 2025",
        "arXiv:2601.17357",
    ),
    "tran2018": ("Spectral signatures in backdoor attacks", "NeurIPS", "2018"),
    "bailey2024": ("Obfuscated activations bypass LLM latent-space defenses", "arXiv:2412.09565"),
    "fanatics2026": ("Why safety probes catch liars but miss fanatics", "arXiv:2603.25861"),
    "burns2023": ("Discovering latent knowledge", "ICLR", "2023"),
    "farquhar2023": ("Challenges with unsupervised LLM knowledge discovery", "arXiv:2312.10029", "2023"),
    "sun2024": ("Massive activations in large language models", "arXiv:2402.17762"),
    "gao2019": ("Representation degeneration problem", "arXiv:1907.12009"),
    "ethayarajh2019": ("How contextual are contextualized word representations", "EMNLP-IJCNLP", "2019"),
}

PAPER_CONTEXT_SENTINELS = [
    ("paper/sections/abstract.tex", "Llama-3-8B", ("grattafiori2024llama3",), 2),
    ("paper/sections/abstract.tex", "Marchenko--Pastur", ("marchenko1967",), 2),
    ("paper/sections/abstract.tex", "detached spikes", ("baik2005", "baik2006", "paul2007"), 2),
    ("paper/sections/abstract.tex", "empirical refusal direction", ("arditi2024",), 2),
    ("paper/sections/abstract.tex", "Wilson intervals", ("wilson1927",), 2),
    ("paper/sections/abstract.tex", "emergent misalignment", ("betley2025emergent", "turner2025organisms"), 2),
    ("paper/sections/abstract.tex", "Qwen2.5-Coder medical organism", ("hui2024qwen25coder",), 2),
    ("paper/sections/intro.tex", "Instruction tuning with human feedback", ("ouyang2022instruct",), 2),
    ("paper/sections/intro.tex", "post-trained checkpoints", ("grattafiori2024llama3",), 3),
    ("paper/sections/intro.tex", "A structureless increment produces", ("marchenko1967",), 3),
    ("paper/sections/intro.tex", "Baik--Ben~Arous--P\\'ech\\'e threshold", ("baik2005", "baik2006", "paul2007"), 3),
    ("paper/sections/intro.tex", "singular vectors of those detached spikes", ("benaych2012",), 2),
    ("paper/sections/intro.tex", "Wilson score intervals", ("wilson1927",), 2),
    ("paper/sections/intro.tex", "emergent-misalignment model-organism setting", ("betley2025emergent", "turner2025organisms"), 2),
    ("paper/sections/spectral.tex", "decoder layers of Llama-3-8B", ("grattafiori2024llama3",), 2),
    ("paper/sections/spectral.tex", "committed spectral sweep", ("results/data/spectral.jsonl", "code/spectral.py"), 4),
    ("paper/sections/spectral.tex", "Other fine-tunes can be low-dimensional", ("aghajanyan2021", "hu2022lora"), 2),
    ("paper/sections/spectral.tex", "we interpret their local concentration", ("vaswani2017attention",), 3),
    ("paper/sections/causal.tex", "harmful AdvBench", ("zou2023universal",), 2),
    ("paper/sections/causal.tex", "harmless Alpaca", ("taori2023alpaca",), 2),
    ("paper/sections/causal.tex", "following \\citet{arditi2024}", ("arditi2024",), 2),
    ("paper/sections/causal.tex", "MMLU/GSM8K/ARC-style evaluation", ("hendrycks2021mmlu", "cobbe2021gsm8k", "clark2018arc"), 3),
    ("paper/sections/causal.tex", "local artifacts for this section", ("results/data/behavioral_capture.json", "results/data/capture_sweep.json", "results/data/ablation_sweep.json", "results/data/ablation_layers.json", "results/data/sufficiency.json"), 8),
    ("paper/sections/misalignment.tex", "model-organism", ("turner2025organisms",), 2),
    ("paper/sections/misalignment.tex", "Qwen2.5-Coder-14B", ("hui2024qwen25coder",), 2),
    ("paper/sections/misalignment.tex", "emergent-misalignment construction", ("betley2025emergent",), 2),
    ("paper/sections/misalignment.tex", "Qwen2.5-Coder-7B-Instruct", ("hui2024qwen25coder",), 3),
    ("paper/sections/misalignment.tex", "local Qwen2.5-14B-Instruct judge", ("yang2024qwen25",), 2),
    ("paper/sections/misalignment.tex", "\\texttt{Llama-3-8B-Instruct}", ("grattafiori2024llama3",), 2),
    ("paper/sections/misalignment.tex", "\\texttt{Mistral-7B-Instruct}", ("jiang2023mistral",), 2),
    ("paper/sections/misalignment.tex", "committed medical-organism evaluation artifact", ("results/data/misalignment_eval_medical.json",), 2),
    ("paper/sections/misalignment.tex", "direction-summary, causal, and held-out-screen artifacts", ("results/data/directions_med.json", "results/data/causal_misalign.json", "results/data/detect_med.json"), 3),
    ("paper/sections/misalignment.tex", "cross-family artifacts", ("results/data/directions_llama.json", "results/data/directions_mistral.json", "results/data/causal_misalign_llama.json", "results/data/causal_misalign_mistral.json", "results/data/detect_llama.json", "results/data/detect_mistral.json"), 8),
    ("paper/sections/appendix.tex", "safetensors shards", ("safetensors",), 2),
    ("paper/sections/appendix.tex", "float64 for the spectral computation", ("code/spectral.py", "results/data/spectral.jsonl"), 2),
    ("paper/sections/appendix.tex", "Harmful prompts are AdvBench goals", ("zou2023universal",), 2),
    ("paper/sections/appendix.tex", "harmless prompts are Alpaca instructions", ("taori2023alpaca",), 2),
    ("paper/sections/appendix.tex", "substring match against a fixed list", ("code/causal.py", "arditi2024"), 3),
    ("paper/sections/theory.tex", "theory note", ("docs/proof.pdf", "johnstone2001"), 3),
    ("paper/sections/related.tex", "convergent linear direction recovered from model", ("soligo2025convergent",), 2),
    ("paper/sections/related.tex", "rank-one adapters", ("turner2025organisms",), 2),
    ("paper/sections/related.tex", "Task vectors edit behavior", ("ilharco2023task", "ortizjimenez2023task"), 3),
    ("paper/sections/related.tex", "representation-engineering methods", ("zou2023repe",), 2),
    ("paper/sections/related.tex", "Sparse autoencoders", ("templeton2026scaling",), 2),
    ("paper/sections/related.tex", "Supervised linear probes", ("goldowskydill2025",), 2),
    ("paper/sections/related.tex", "hidden-objective audits", ("marks2025",), 2),
    ("paper/sections/related.tex", "safety-degrading fine-tuning", ("larf2025",), 2),
    ("paper/sections/discussion.tex", "DPO-style preference optimization", ("rafailov2023dpo",), 2),
    ("paper/sections/discussion.tex", "fully prompt-provenanced HarmBench OOD run", ("mazeika2024harmbench",), 2),
    ("paper/sections/discussion.tex", "Baik--Ben~Arous--P\\'ech\\'e threshold separates", ("baik2005", "baik2006", "paul2007"), 2),
    ("paper/sections/discussion.tex", "harmless-prompt rates under the same", ("hendrycks2021mmlu", "cobbe2021gsm8k", "clark2018arc"), 3),
    ("paper/sections/discussion.tex", "refusal results are on a single released model", ("grattafiori2024llama3",), 8),
    ("paper/sections/discussion.tex", "low intrinsic dimension of fine-tuning", ("hu2022lora", "aghajanyan2021"), 2),
    ("paper/sections/discussion.tex", "MMLU/GSM8K/ARC-style evaluations", ("hendrycks2021mmlu", "cobbe2021gsm8k", "clark2018arc"), 3),
    ("paper/sections/em_examples.tex", "Qwen2.5-14B-Instruct judge scores", ("yang2024qwen25",), 2),
]

DOCUMENT_CONTEXT_SENTINELS = [
    ("docs/proof.tex", "Baik--Ben~Arous--P\\'ech\\'e", ("bbp2005", "baik2006", "paul2007"), 2),
    ("docs/proof.tex", "Marchenko--Pastur bulk", ("marchenko1967",), 2),
    ("docs/proof.tex", "Tracy--Widom null", ("tracywidom1996", "johnstone2001"), 2),
    ("docs/proof.tex", "strongest available white-box answers come", ("goldowskydill2025",), 3),
    ("docs/proof.tex", "hidden objective was access to the training data", ("marks2025",), 3),
    ("docs/proof.tex", "Heavy-tailed self-regularization reads", ("martin2021",), 3),
    ("docs/proof.tex", "Spectral outlier analysis detects data poisoning", ("tran2018",), 2),
    ("docs/proof.tex", "hallucination detection", ("ettori2026",), 3),
    ("docs/proof.tex", "Random-matrix statistics have been applied to hallucination detection", ("ettori2026",), 3),
    ("docs/proof.tex", "ties weight spectra", ("staats2024",), 3),
    ("docs/proof.tex", "model lineage in a way deliberately invariant", ("ghost2026",), 3),
    ("docs/proof.tex", "emergent misalignment is mediated by a single", ("soligo2025",), 2),
    ("docs/proof.tex", "single rank-one adapter suffices", ("turner2025",), 2),
    ("docs/proof.tex", "shared low-dimensional subspace", ("shared2025",), 2),
    ("PLAN.md", "supervised probes", ("https://arxiv.org/abs/2502.03407", "https://arxiv.org/abs/2503.10965"), 1),
    ("PLAN.md", "Label with [MASK]", ("https://arxiv.org/abs/2503.03750", "https://arxiv.org/abs/2109.07958", "https://arxiv.org/abs/2502.17424"), 1),
    ("PLAN.md", "**Baselines.**", ("https://arxiv.org/abs/2502.03407", "https://arxiv.org/abs/2310.01405"), 1),
    ("PLAN.md", "Betley et al. report full fine-tuned insecure-code models", ("https://arxiv.org/abs/2502.17424",), 1),
    ("PLAN.md", "committed code-organism datasets", ("data/em/README.md",), 1),
    ("README.md", "Artifact map for the headline claims", ("results/data/spectral.jsonl", "results/data/behavioral_capture.json", "results/data/misalignment_eval_medical.json", "results/data/directions_llama.json"), 10),
]


LITERATURE_REVIEW_BLOCKS = [
    ("paper/sections/related.tex", None),
    ("docs/proof.tex", r"\\section\{Related work\}", r"\\section\{Conclusion\}"),
]


def strip_latex_comments(text):
    """Remove unescaped percent comments while preserving line numbers."""
    stripped = []
    for line in text.splitlines(keepends=True):
        cut = len(line)
        for idx, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            j = idx - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                cut = idx
                break
        stripped.append(line[:cut] + re.sub(r"[^\n]", " ", line[cut:]))
    return "".join(stripped)


def used_cite_keys(paths):
    keys = []
    for path in paths:
        text = strip_latex_comments(path.read_text())
        for match in CITE_RE.finditer(text):
            line = text[:match.start()].count("\n") + 1
            for key in match.group(1).split(","):
                key = key.strip()
                if key:
                    keys.append((key, path, line))
    return keys


def parse_bib_entries(path):
    text = path.read_text()
    entries = {}
    matches = list(BIB_RE.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[match.group(2).strip()] = (match.group(1), text[start:end])
    return entries


def find_field(entry_text, field):
    return re.search(rf"\b{re.escape(field)}\s*=", entry_text, re.IGNORECASE) is not None


def field_value(entry_text, field):
    match = re.search(rf"\b{re.escape(field)}\s*=", entry_text, re.IGNORECASE)
    if match is None:
        return None
    pos = match.end()
    while pos < len(entry_text) and entry_text[pos].isspace():
        pos += 1
    if pos >= len(entry_text):
        return ""
    if entry_text[pos] == "{":
        depth = 0
        start = pos + 1
        for idx in range(pos, len(entry_text)):
            char = entry_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return entry_text[start:idx].strip()
        return entry_text[start:].strip()
    if entry_text[pos] == '"':
        start = pos + 1
        escaped = False
        for idx in range(start, len(entry_text)):
            char = entry_text[idx]
            if char == '"' and not escaped:
                return entry_text[start:idx].strip()
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        return entry_text[start:].strip()
    end = entry_text.find(",", pos)
    if end == -1:
        end = len(entry_text)
    return entry_text[pos:end].strip()


def check_bib_field_values(errors, key, entry):
    for field in REQUIRED_BIB_FIELDS:
        value = field_value(entry, field)
        if value is None:
            errors.append(f"paper refs.bib entry {key} missing {field}")
            continue
        if not value.strip():
            errors.append(f"paper refs.bib entry {key} has empty {field}")
        if PLACEHOLDER_RE.search(value):
            errors.append(f"paper refs.bib entry {key} has placeholder text in {field}")
    year = field_value(entry, "year")
    if year is not None and not re.fullmatch(r"\d{4}", year.strip()):
        errors.append(f"paper refs.bib entry {key} year must be exactly four digits")
    author = field_value(entry, "author")
    if author is None or not author.strip():
        errors.append(f"paper refs.bib entry {key} missing author")
    else:
        if PLACEHOLDER_RE.search(author):
            errors.append(f"paper refs.bib entry {key} has placeholder text in author")
        if " and others" in author and key not in ALLOW_TRUNCATED_AUTHORS:
            errors.append(f"paper refs.bib entry {key} uses unallowlisted 'and others'")
    for field in ("journal", "booktitle", "howpublished", "note"):
        value = field_value(entry, field)
        if value is None:
            continue
        if not value.strip():
            errors.append(f"paper refs.bib entry {key} has empty {field}")
        if PLACEHOLDER_RE.search(value):
            errors.append(f"paper refs.bib entry {key} has placeholder text in {field}")
        for arxiv_year, _ in ARXIV_ID_RE.findall(value):
            if year and year.strip()[2:] != arxiv_year[:2]:
                errors.append(
                    f"paper refs.bib entry {key} has arXiv id {arxiv_year} inconsistent with year {year.strip()}"
                )


def check_paper(errors):
    tex_paths = sorted((ROOT / "paper").glob("**/*.tex"))
    bib_path = ROOT / "paper" / "refs.bib"
    entries = parse_bib_entries(bib_path)
    all_keys = [match.group(2).strip() for match in BIB_RE.finditer(bib_path.read_text())]
    for key in sorted({key for key in all_keys if all_keys.count(key) > 1}):
        errors.append(f"paper refs.bib duplicate entry key: {key}")
    bib_keys = set(entries)
    used = used_cite_keys(tex_paths)
    used_keys = {key for key, _, _ in used}

    for key, path, line in used:
        if key not in bib_keys:
            errors.append(f"paper missing bib entry for {key} at {path.relative_to(ROOT)}:{line}")
    for key in sorted(bib_keys - used_keys):
        errors.append(f"paper bibliography entry is uncited: {key}")
    for key, (_, entry) in sorted(entries.items()):
        check_bib_field_values(errors, key, entry)
        if not any(find_field(entry, field) for field in REQUIRED_VENUE_FIELDS):
            errors.append(f"paper refs.bib entry {key} missing venue field")
    for key in sorted(bib_keys - set(PAPER_METADATA_SENTINELS)):
        errors.append(f"paper refs.bib entry {key} lacks metadata sentinel coverage")
    for key, expected_fragments in PAPER_METADATA_SENTINELS.items():
        entry = entries.get(key)
        if entry is None:
            errors.append(f"paper refs.bib missing metadata-sentinel entry: {key}")
            continue
        body = entry[1]
        for fragment in expected_fragments:
            if fragment not in body:
                errors.append(f"paper refs.bib entry {key} missing verified metadata fragment {fragment!r}")


def check_proof(errors):
    path = ROOT / "docs" / "proof.tex"
    text = path.read_text()
    bib_keys = set(BIBITEM_RE.findall(text))
    used = used_cite_keys([path])
    used_keys = {key for key, _, _ in used}
    blocks = dict(BIBITEM_BLOCK_RE.findall(text))

    for key, _, line in used:
        if key not in bib_keys:
            errors.append(f"proof missing bibitem for {key} at docs/proof.tex:{line}")
    for key in sorted(bib_keys - used_keys):
        errors.append(f"proof bibitem is uncited: {key}")
    for key in sorted(bib_keys):
        body = blocks.get(key, "")
        if not re.search(r"\b(?:19|20)\d{2}\b", body):
            errors.append(f"proof bibitem {key} missing four-digit year")
        if "``" not in body and "\\emph{" not in body:
            errors.append(f"proof bibitem {key} missing title marker")
        if not any(marker in body for marker in PROOF_VENUE_MARKERS):
            errors.append(f"proof bibitem {key} missing venue or preprint marker")
    for key in sorted(bib_keys - set(PROOF_METADATA_SENTINELS)):
        errors.append(f"proof bibitem {key} lacks metadata sentinel coverage")
    for key, expected_fragments in PROOF_METADATA_SENTINELS.items():
        body = blocks.get(key)
        if body is None:
            errors.append(f"proof missing metadata-sentinel bibitem: {key}")
            continue
        for fragment in expected_fragments:
            if fragment not in body:
                errors.append(f"proof bibitem {key} missing verified metadata fragment {fragment!r}")


def check_placeholders(errors):
    paths = sorted((ROOT / "paper").glob("**/*.tex")) + [ROOT / "docs" / "proof.tex"]
    for path in paths:
        text = strip_latex_comments(path.read_text())
        low = text.lower()
        for pattern in ("citation needed", "cite needed", "\\cite{}"):
            if pattern in low:
                errors.append(f"citation placeholder found in {path.relative_to(ROOT)}: {pattern}")


def check_context_sentinels(errors):
    for rel_path, needle, required_keys, window in PAPER_CONTEXT_SENTINELS + DOCUMENT_CONTEXT_SENTINELS:
        path = ROOT / rel_path
        lines = strip_latex_comments(path.read_text()).splitlines()
        found_any = False
        for index, line in enumerate(lines):
            if needle not in line:
                continue
            found_any = True
            start = max(0, index - window)
            end = min(len(lines), index + window + 1)
            nearby = "\n".join(lines[start:end])
            cited_keys = citation_keys_in_text(nearby)
            for key in required_keys:
                if requires_cite_command(key):
                    ok = key in cited_keys
                    requirement = "citation command"
                else:
                    ok = key in nearby
                    requirement = "nearby text"
                if not ok:
                    errors.append(
                        f"paper source claim {needle!r} at {rel_path}:{index + 1} "
                        f"lacks nearby {requirement} for {key}"
                    )
        if not found_any:
            errors.append(f"paper source claim sentinel not found in {rel_path}: {needle!r}")


def citation_keys_in_text(text):
    keys = set()
    for match in CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def requires_cite_command(key):
    return not (
        key.startswith("http://")
        or key.startswith("https://")
        or "/" in key
        or "." in key
    )


def block_has_citation(block):
    if CITE_RE.search(block):
        return True
    return bool(re.search(r"\[[^\]]+\]\(https?://[^)]+\)", block))


def paragraph_blocks(text):
    return [
        block.strip()
        for block in re.split(r"\n\s*\n", strip_latex_comments(text))
        if block.strip()
    ]


def check_literature_review_citation_density(errors):
    for spec in LITERATURE_REVIEW_BLOCKS:
        rel_path = spec[0]
        text = (ROOT / rel_path).read_text()
        if len(spec) == 3:
            start_match = re.search(spec[1], text)
            end_match = re.search(spec[2], text)
            if start_match is None or end_match is None or end_match.start() <= start_match.end():
                errors.append(f"literature citation density: could not isolate section in {rel_path}")
                continue
            text = text[start_match.end():end_match.start()]
        for block in paragraph_blocks(text):
            if not block.startswith("\\paragraph"):
                continue
            title = re.match(r"\\paragraph\{([^}]+)\}", block)
            label = title.group(1) if title else block[:40]
            if not block_has_citation(block):
                errors.append(
                    f"literature citation density: {rel_path} paragraph {label!r} has no citation"
                )


def main():
    errors = []
    check_paper(errors)
    check_proof(errors)
    check_placeholders(errors)
    check_context_sentinels(errors)
    check_literature_review_citation_density(errors)
    if errors:
        print(f"citation check FAILED: {errors[0]}", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        raise SystemExit(1)
    print("citation check passed")


if __name__ == "__main__":
    main()
