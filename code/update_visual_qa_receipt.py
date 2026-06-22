#!/usr/bin/env python3
"""Update the paper visual-QA receipt with deterministic render hashes.

This script does not decide whether the pages look good. It records the exact
PNG pages produced by pdftoppm for a visually inspected PDF, so the completion
gate can later re-render the PDF and verify that the inspected artifact is still
the current artifact.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def page_number(path):
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"cannot parse page number from {path.name}")
    return int(match.group(1))


def render_page_records(pdf, dpi):
    with tempfile.TemporaryDirectory(prefix="alignment-paper-render-") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        pages = sorted(Path(tmp).glob("page-*.png"), key=page_number)
        return [
            {
                "page": page_number(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in pages
        ]


def parse_pages(value):
    if not value.strip():
        return []
    pages = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        pages.append(int(part))
    return pages


def utc_now():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="docs/paper.pdf")
    ap.add_argument("--receipt", default="results/data/visual_qa.json")
    ap.add_argument("--render-dpi", type=int, default=120)
    ap.add_argument("--inspected-pages-full-size", default="")
    ap.add_argument("--method", required=True)
    args = ap.parse_args()

    pdf = ROOT / args.pdf
    receipt = ROOT / args.receipt
    if not pdf.exists():
        raise SystemExit(f"missing PDF: {args.pdf}")
    if args.render_dpi <= 0:
        raise SystemExit("--render-dpi must be positive")

    page_records = render_page_records(pdf, args.render_dpi)
    data = {
        "pdf": args.pdf,
        "pdf_sha256": sha256_file(pdf),
        "inspected_at_utc": utc_now(),
        "pages_total": len(page_records),
        "pages_checked": len(page_records),
        "render_dpi": args.render_dpi,
        "inspected_pages_full_size": parse_pages(args.inspected_pages_full_size),
        "method": args.method,
        "render_receipt": {
            "schema": "pdf_render_receipt_v1",
            "renderer": "pdftoppm",
            "format": "png",
            "render_dpi": args.render_dpi,
            "page_count": len(page_records),
            "pages": page_records,
        },
        "visual_defects": [],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {args.receipt} with {len(page_records)} rendered page hashes")


if __name__ == "__main__":
    main()
