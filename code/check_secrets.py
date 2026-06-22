#!/usr/bin/env python3
"""Scan the repository for high-confidence API key and private-key patterns."""
import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 8 * 1024 * 1024
SKIP_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".npz",
    ".pdf",
    ".png",
    ".pyc",
}


SECRET_PATTERNS = [
    ("google_api_key", "AI" + "za" + r"[0-9A-Za-z_-]{35}"),
    ("openai_api_key", "sk-" + r"[A-Za-z0-9_-]{20,}"),
    ("github_fine_grained_pat", "github" + r"_pat_[A-Za-z0-9_]{20,}"),
    ("github_classic_pat", r"(gh" + r"p|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}"),
    ("aws_access_key", r"(" + "AK" + r"IA|" + "AS" + r"IA)[A-Z0-9]{16}"),
    ("private_key", "-----BEGIN " + r"[A-Z ]*PRIVATE KEY-----"),
]


def run_git(args, *, text=True):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def tracked_and_unignored_paths():
    proc = run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"], text=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode(errors="replace")
        path = ROOT / rel
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield rel, path


def scan_worktree(regexes):
    findings = []
    for rel, path in tracked_and_unignored_paths():
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
            raw = path.read_bytes()
        except OSError as exc:
            findings.append(("read_error", rel, 0, str(exc)))
            continue
        if b"\0" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for name, regex in regexes:
            for match in regex.finditer(text):
                line = text[: match.start()].count("\n") + 1
                findings.append((name, rel, line, "current tree"))
    return findings


def history_refs():
    proc = run_git(["rev-list", "--all"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return [line for line in proc.stdout.splitlines() if line]


def scan_history(combined_pattern):
    findings = []
    for commit in history_refs():
        proc = run_git(["grep", "-I", "-n", "-E", combined_pattern, commit, "--", "."])
        if proc.returncode == 1:
            continue
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        for line in proc.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) < 4:
                findings.append(("history_match", commit[:12], 0, line))
                continue
            _, rel, line_no, _ = parts
            if Path(rel).suffix in SKIP_SUFFIXES:
                continue
            findings.append(("history_match", f"{commit[:12]}:{rel}", int(line_no), "reachable history"))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true", help="also scan all reachable git history")
    parser.add_argument("--max-findings", type=int, default=20)
    args = parser.parse_args()

    regexes = [(name, re.compile(pattern)) for name, pattern in SECRET_PATTERNS]
    findings = scan_worktree(regexes)
    if args.history:
        combined = "|".join(f"({pattern})" for _, pattern in SECRET_PATTERNS)
        findings.extend(scan_history(combined))

    if findings:
        print(f"secret scan FAILED: {len(findings)} high-confidence match(es)", file=sys.stderr)
        for kind, rel, line, detail in findings[: args.max_findings]:
            print(f" - {kind}: {rel}:{line} ({detail})", file=sys.stderr)
        if len(findings) > args.max_findings:
            print(f" - ... {len(findings) - args.max_findings} more", file=sys.stderr)
        return 1
    scope = "current tree and reachable history" if args.history else "current tree"
    print(f"secret scan passed ({scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
