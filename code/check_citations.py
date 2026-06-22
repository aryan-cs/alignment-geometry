#!/usr/bin/env python3
"""Validate local LaTeX citation hygiene for the paper and proof."""
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CITE_RE = re.compile(r"\\cite[tpa]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")
BIB_RE = re.compile(r"@(\w+)\{([^,]+),")
BIBITEM_RE = re.compile(r"\\bibitem\{([^}]+)\}")
BIBITEM_BLOCK_RE = re.compile(
    r"\\bibitem\{([^}]+)\}([\s\S]*?)(?=\\bibitem\{|\\end\{thebibliography\})"
)
REQUIRED_BIB_FIELDS = ("title", "year")
REQUIRED_VENUE_FIELDS = ("journal", "booktitle", "howpublished", "note")
PROOF_VENUE_MARKERS = ("\\emph{", "arXiv:", "Springer", "MSc thesis", "Preprint")


def used_cite_keys(paths):
    keys = []
    for path in paths:
        text = path.read_text()
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
    entries = parse_bib_entries(ROOT / "paper" / "refs.bib")
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


def check_placeholders(errors):
    paths = sorted((ROOT / "paper").glob("**/*.tex")) + [ROOT / "docs" / "proof.tex"]
    for path in paths:
        text = path.read_text()
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
