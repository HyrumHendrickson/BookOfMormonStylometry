#!/usr/bin/env python3
"""
run_corpora.py  (Program 4 of 4: specific to this repository)

Loads the Federalist Papers and/or the Book of Mormon, cuts them into labeled segments,
and hands each corpus to pipeline.py (which runs word_stability.py and distance_plots.py).

  python run_corpora.py federalist
  python run_corpora.py bom --abridger-is-not-author --segment-words 400
  python run_corpora.py both --holdout 0.5 --top-k 15
  python run_corpora.py federalist --federalist-file path/to/pg1404.txt

Corpus rules
------------
Federalist: authorship by the standard attribution; joint essays (18-20) and disputed essays
  (49-58, 62, 63) are excluded. Text comes from Project Gutenberg #1404 (downloaded once and
  cached in data/).
Book of Mormon: chapters where exactly one author is listed in
  AnnotatedBooks/book_of_mormon_authors_by_chapter.json, limited to Nephi (son of Lehi),
  Moroni (son of Mormon), Alma the Younger and Mormon. Alma has no such chapters unless
  --abridger-is-not-author is given, which ignores Mormon/Moroni where they appear only as abridger.

Segmentation
------------
Each essay/chapter is cut into consecutive blocks of --segment-words words; a leftover block is
kept only if it has at least --min-segment-words words (default: the block size, so every
segment has the same length). The segments are also saved to output/<corpus>/segments.jsonl
so you can rerun pipeline.py, word_stability.py or distance_plots.py on them directly.

All pipeline options (--top-k, --metric, --holdout, --rank-by, ...) are accepted; see
`python pipeline.py --help`.
"""

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import pipeline
import word_stability as ws

ROOT = Path(__file__).resolve().parent
BOOKS_DIR = next(
    (p / "AnnotatedBooks" for p in [ROOT, *ROOT.parents] if (p / "AnnotatedBooks").is_dir()),
    ROOT / "AnnotatedBooks",
)
BOM_TEXT = BOOKS_DIR / "book_of_mormon.txt"
BOM_AUTHORS = BOOKS_DIR / "book_of_mormon_authors_by_chapter.json"
DATA_DIR = ROOT / "data"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1404/pg1404.txt"

FEDERALIST_AUTHORS = {}
for n in [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17] + list(range(21, 37)) + [59, 60, 61] + list(range(65, 86)):
    FEDERALIST_AUTHORS[n] = "Hamilton"
for n in [10, 14] + list(range(37, 49)):
    FEDERALIST_AUTHORS[n] = "Madison"
for n in [2, 3, 4, 5, 64]:
    FEDERALIST_AUTHORS[n] = "Jay"
FEDERALIST_COLORS = {"Hamilton": "#d62728", "Madison": "#1f77b4", "Jay": "#2ca02c"}

BOM_AUTHOR_MAP = {
    "Nephi, son of Lehi": "Nephi",
    "Moroni, son of Mormon": "Moroni",
    "Alma the Younger": "Alma",
    "Mormon": "Mormon",
}
BOM_COLORS = {"Nephi": "#1f77b4", "Moroni": "#ff7f0e", "Alma": "#2ca02c", "Mormon": "#d62728"}
BOM_HEADINGS = {
    "The First Book of Nephi": "1 Nephi",
    "The Second Book of Nephi": "2 Nephi",
    "The Book of Jacob": "Jacob",
    "The Book of Enos": "Enos",
    "The Book of Jarom": "Jarom",
    "The Book of Omni": "Omni",
    "The Words of Mormon": "Words of Mormon",
    "The Book of Mosiah": "Mosiah",
    "The Book of Alma": "Alma",
    "The Book of Helaman": "Helaman",
    "Third Nephi": "3 Nephi",
    "Fourth Nephi": "4 Nephi",
    "The Book of Mormon": "Mormon",
    "The Book of Ether": "Ether",
    "The Book of Moroni": "Moroni",
}


# ----------------------------------------------------------------------------
# Loaders: return (author, tokens, document_label) per essay/chapter
# ----------------------------------------------------------------------------
def find_federalist_file(explicit):
    if explicit:
        return Path(explicit)
    candidates = [DATA_DIR / "federalist_pg1404.txt"]
    candidates += [p / "Stylometry" / "data" / "federalist_pg1404.txt" for p in [ROOT, *ROOT.parents]]
    for c in candidates:
        if c.exists():
            return c
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "federalist_pg1404.txt"
    print(f"Downloading {GUTENBERG_URL}")
    req = urllib.request.Request(GUTENBERG_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())
    return path


def load_federalist(path=None):
    raw = find_federalist_file(path).read_text(encoding="utf-8", errors="replace")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]

    heading = re.compile(r"(?m)^[ \t]*FEDERALIST\.?[ \t]+No\.?[ \t]*(\d+)[ \t]*$")
    marks = list(heading.finditer(body))
    essays = {}
    for i, m in enumerate(marks):
        num = int(m.group(1))
        chunk = body[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(body)]
        if num not in essays or len(chunk) > len(essays[num]):
            essays[num] = chunk

    units = []
    salute = re.compile(r"(?im)^[ \t]*To the (People|Citizens)[^\n]{0,80}:[ \t]*$")
    for num, chunk in sorted(essays.items()):
        author = FEDERALIST_AUTHORS.get(num)
        if author is None:
            continue
        m = salute.search(chunk[:1500])
        if m:
            chunk = chunk[m.end():]
        chunk = re.sub(r"\bPUBLIUS\b", " ", chunk)
        units.append((author, ws.tokenize(chunk), f"Federalist {num}"))
    if len(essays) < 80:
        print(f"Warning: found only {len(essays)} Federalist essays; check the source text.")
    return units


def load_bom(abridger_is_not_author=False):
    authors_json = json.loads(BOM_AUTHORS.read_text(encoding="utf-8"))
    attribution = {}
    for b in authors_json["books"]:
        for ch in b["chapters"]:
            names = []
            for a in ch["authors"]:
                if abridger_is_not_author and set(a["roles"]) == {"abridger"}:
                    continue
                names.append(a["name"])
            if len(names) == 1 and names[0] in BOM_AUTHOR_MAP:
                attribution[(b["book"], ch["chapter"])] = BOM_AUTHOR_MAP[names[0]]

    chapters = defaultdict(list)
    book, chap = None, None
    for line in BOM_TEXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line in BOM_HEADINGS:
            book, chap = BOM_HEADINGS[line], None
            continue
        m = re.fullmatch(r"Chapter (\d+)", line)
        if m and book:
            chap = int(m.group(1))
            continue
        v = re.match(r"(\d+) (.*)", line)
        if v and book and chap:
            chapters[(book, chap)].append(v.group(2))

    units = []
    for key, author in attribution.items():
        if key in chapters:
            units.append((author, ws.tokenize(" ".join(chapters[key])), f"{key[0]} {key[1]}"))
    return units


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------
def to_segments(units, size, min_words):
    segments = []
    for author, tokens, doc in units:
        blocks = [tokens[i:i + size] for i in range(0, len(tokens) - size + 1, size)]
        rem = len(tokens) % size
        if rem > 0 and rem >= min_words:
            blocks.append(tokens[len(tokens) - rem:])
        for j, block in enumerate(blocks):
            segments.append({"author": author, "doc": doc, "tokens": block})
    return segments


def save_segments(segments, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps({"author": s["author"], "doc": s["doc"], "text": " ".join(s["tokens"])}) + "\n")
    print(f"  Saved {path}")


def main():
    ap = pipeline.build_parser(__doc__)
    ap.add_argument("corpus", choices=["federalist", "bom", "both"])
    ap.add_argument("--segment-words", type=int, default=500, help="words per segment (default 500)")
    ap.add_argument("--min-segment-words", type=int, default=None,
                    help="keep a leftover block only if at least this long (default: the segment size)")
    ap.add_argument("--federalist-file", help="local plain-text Federalist Papers (Gutenberg format)")
    ap.add_argument("--abridger-is-not-author", action="store_true",
                    help="ignore Mormon/Moroni where listed only as abridger (gives Alma qualifying chapters)")
    opts = ap.parse_args()
    min_seg = opts.min_segment_words if opts.min_segment_words is not None else opts.segment_words
    if min_seg > opts.segment_words:
        ap.error("--min-segment-words cannot exceed --segment-words")

    base = Path(opts.out_dir)
    jobs = []
    if opts.corpus in ("federalist", "both"):
        jobs.append(("federalist", "Federalist Papers", load_federalist(opts.federalist_file), FEDERALIST_COLORS))
    if opts.corpus in ("bom", "both"):
        jobs.append(("bom", "Book of Mormon", load_bom(opts.abridger_is_not_author), BOM_COLORS))

    for key, title, units, colors in jobs:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        segments = to_segments(units, opts.segment_words, min_seg)
        out = base / key
        save_segments(segments, out / "segments.jsonl")
        pipeline.run_pipeline(segments, opts, colors=colors, title=title, out_dir=out)


if __name__ == "__main__":
    main()