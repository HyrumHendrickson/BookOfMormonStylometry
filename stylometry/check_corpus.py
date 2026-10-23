#!/usr/bin/env python3
"""Program 2: verify that a corpus directory is in the standardized format
and is large enough for the analysis.

Errors (exit code 1) stop the pipeline; warnings are reported only.

Usage
  python check_corpus.py corpora/federalist --segment-size 1000
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

from common import AUTHOR_FILE_RE, list_author_files, segment, tokenize

ARTIFACT_PATTERNS = {
    "verse or section number at line start": re.compile(r"^\d+[\s.:]"),
    "chapter heading": re.compile(r"^\s*chapter\s+([ivxlc]+|\d+)\b", re.I),
    "Project Gutenberg boilerplate": re.compile(r"project gutenberg|\*\*\* ?(start|end) of", re.I),
    "Federalist signature or header": re.compile(r"\bPUBLIUS\b|FEDERALIST\.? No\."),
    "illustration or footnote marker": re.compile(r"\[(illustration|picture|footnote)|\{\d+\}", re.I),
    "italics underscore": re.compile(r"_"),
    "HTML or XML markup": re.compile(r"</?[a-z][^>]*>", re.I),
}


def check(corpus_dir: Path, segment_size: int, min_words: int, min_segments: int,
          max_ratio: float) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if not corpus_dir.is_dir():
        return [f"{corpus_dir} is not a directory"], []

    files = list_author_files(corpus_dir)
    if len(files) < 2:
        errors.append(f"need at least 2 author files, found {len(files)}")

    counts, hashes, line_owner = {}, {}, defaultdict(set)
    for f in files:
        name = f.name
        if not AUTHOR_FILE_RE.match(name):
            errors.append(f"{name}: file name must match [a-z0-9_]+.txt")
        raw = f.read_bytes()
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError as e:
            errors.append(f"{name}: non-ASCII byte at offset {e.start}; rerun normalization")
            text = raw.decode("utf-8", "replace")
        if "\r" in text:
            errors.append(f"{name}: contains carriage returns (use LF line endings)")
        if "\t" in text:
            errors.append(f"{name}: contains tab characters")
        if not text.endswith("\n"):
            warnings.append(f"{name}: no trailing newline")

        lines = text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")
        blank = sum(1 for l in lines if not l.strip())
        if blank:
            errors.append(f"{name}: {blank} blank line(s); one unit per line, no blank lines")
        padded = sum(1 for l in lines if l != l.strip() or "  " in l)
        if padded:
            errors.append(f"{name}: {padded} line(s) with leading/trailing or doubled spaces")

        for label, pat in ARTIFACT_PATTERNS.items():
            hits = [i + 1 for i, l in enumerate(lines) if pat.search(l)]
            if hits:
                msg = f"{name}: {len(hits)} line(s) with {label} (first at line {hits[0]})"
                (errors if label in ("Project Gutenberg boilerplate", "HTML or XML markup")
                 else warnings).append(msg)

        tokens = tokenize(text)
        counts[f.stem] = len(tokens)
        n_seg = len(segment(tokens, segment_size))
        if len(tokens) < min_words:
            errors.append(f"{name}: {len(tokens):,} words, below minimum {min_words:,}")
        if n_seg < min_segments:
            errors.append(f"{name}: only {n_seg} segment(s) of {segment_size} words; "
                          f"need {min_segments} (lower --segment-size or add text)")
        elif n_seg < 2 * min_segments:
            warnings.append(f"{name}: only {n_seg} segments; estimates of consistency will be noisy")

        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in hashes:
            errors.append(f"{name}: identical to {hashes[digest]}")
        hashes[digest] = name
        for l in lines:
            if len(l) > 200:
                line_owner[l].add(f.stem)

    shared = [l for l, owners in line_owner.items() if len(owners) > 1]
    if shared:
        errors.append(f"{len(shared)} unit(s) appear in more than one author's file "
                      f"(text would be attributed to two authors)")

    if counts and min(counts.values()) > 0:
        ratio = max(counts.values()) / min(counts.values())
        if ratio > max_ratio:
            warnings.append(f"size imbalance: largest/smallest author = {ratio:.1f}x")
    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--segment-size", type=int, default=1000)
    ap.add_argument("--min-words", type=int, default=2500)
    ap.add_argument("--min-segments", type=int, default=5)
    ap.add_argument("--max-ratio", type=float, default=25.0,
                    help="warn when largest/smallest author word count exceeds this")
    a = ap.parse_args()

    errors, warnings = check(a.corpus_dir, a.segment_size, a.min_words, a.min_segments, a.max_ratio)
    print(f"Checking {a.corpus_dir} (segment size {a.segment_size})")
    for f in list_author_files(a.corpus_dir):
        n = len(tokenize(f.read_text(encoding="ascii", errors="replace")))
        print(f"    {f.stem:<12} {n:>9,} words  {n // a.segment_size:>5} segments")
    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR:   {e}")
    print("  PASS" if not errors else f"  FAIL ({len(errors)} error(s))")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
