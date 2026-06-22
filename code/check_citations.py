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
PROOF_VENUE_MARKERS = ("\\emph{", "arXiv:", "Springer", "MSc thesis", "Preprint")
PAPER_METADATA_SENTINELS = {
    "betley2025emergent": ("Emergent Misalignment", "arXiv:2502.17424", "2025"),
    "turner2025organisms": ("Model organisms for emergent misalignment", "arXiv:2506.11613", "2025"),
    "soligo2025convergent": ("Convergent linear representations", "arXiv:2506.11618", "2025"),
    "arditi2024": ("Refusal in language models is mediated by a single direction", "2024"),
    "templeton2026scaling": ("Scaling monosemanticity", "arXiv:2605.29358", "2026"),
    "hui2024qwen25coder": ("Qwen2.5-Coder", "arXiv:2409.12186", "2024"),
    "yang2024qwen25": ("Qwen2.5", "arXiv:2412.15115", "2024"),
    "grattafiori2024llama3": ("The {Llama} 3 herd of models", "arXiv:2407.21783", "2024"),
    "jiang2023mistral": ("{Mistral} 7{B}", "arXiv:2310.06825", "2023"),
    "zou2023universal": ("Universal and transferable adversarial attacks", "arXiv:2307.15043", "2023"),
}
PROOF_METADATA_SENTINELS = {
    "springer2026": ("The geometry of alignment collapse", "Korolova", "arXiv:2602.15799"),
    "li2025geometry": ("Tracing the representation geometry", "arXiv:2509.23024"),
    "ghost2026": ("tracing LLM lineage with SVD-fingerprint", "arXiv:2511.06390"),
    "goldowskydill2025": ("Detecting strategic deception using linear probes", "arXiv:2502.03407"),
    "marks2025": ("Auditing language models for hidden objectives", "arXiv:2503.10965"),
    "shared2025": ("Shared parameter subspaces", "arXiv:2511.02022"),
}


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
        for field in REQUIRED_BIB_FIELDS:
            if not find_field(entry, field):
                errors.append(f"paper refs.bib entry {key} missing {field}")
        if not any(find_field(entry, field) for field in REQUIRED_VENUE_FIELDS):
            errors.append(f"paper refs.bib entry {key} missing venue field")
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


def main():
    errors = []
    check_paper(errors)
    check_proof(errors)
    check_placeholders(errors)
    if errors:
        print(f"citation check FAILED: {errors[0]}", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        raise SystemExit(1)
    print("citation check passed")


if __name__ == "__main__":
    main()
