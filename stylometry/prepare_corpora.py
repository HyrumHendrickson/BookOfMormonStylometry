#!/usr/bin/env python3
"""Program 1: build standardized author files for each corpus.

Corpora
  bom         Book of Mormon. Chapters whose annotation lists exactly one
              author; the N authors with the most such chapters (default 3:
              Mormon, Nephi son of Lehi, Moroni). Verse text only; chapter
              summaries and book headings are modern editorial matter and are
              excluded.
  federalist  Federalist Papers. Hamilton, Madison, Jay, using the standard
              attribution (Adair 1944; Mosteller & Wallace 1964). The twelve
              disputed papers (49-58, 62, 63) and the three joint papers
              (18-20) are excluded.
  kjv_nt      King James New Testament. Paul (the seven undisputed epistles),
              Luke (Luke, Acts), John (Gospel of John, 1-3 John).
  bronte      Novels of the Bronte sisters. Charlotte (Jane Eyre, Villette,
              Shirley, The Professor), Emily (Wuthering Heights), Anne (Agnes
              Grey, The Tenant of Wildfell Hall). Prefaces and publisher
              matter are excluded.

Usage
  python prepare_corpora.py --corpus all
  python prepare_corpora.py --corpus bom --bom-source path/to/zip_or_dir
"""

import argparse
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from common import CACHE_DIR, CORPORA_DIR, ROOT, clean_unit, tokenize, write_json

CORPORA = ["bom", "federalist", "kjv_nt", "bronte"]


# ---------------------------------------------------------------- download

def fetch(urls: list[str], cache_name: str) -> bytes:
    """Download the first URL that succeeds; cache the result locally."""
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / cache_name
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_bytes()
    errors = []
    for url in urls:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "stylometry-prep/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                cached.write_bytes(data)
                return data
            except Exception as e:  # network errors, 404s
                errors.append(f"{url}: {e}")
                time.sleep(1 + attempt)
    raise RuntimeError(
        f"Could not download {cache_name}. Place the file at {cached} manually.\n  "
        + "\n  ".join(errors))


def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("undecodable text")


# ---------------------------------------------------------------- output

def write_corpus(name: str, authors: dict[str, list[str]], manifest: dict) -> None:
    out = CORPORA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    manifest["authors"] = {}
    for slug, units in sorted(authors.items()):
        units = [u for u in (clean_unit(x) for x in units) if u]
        text = "\n".join(units) + "\n"
        (out / f"{slug}.txt").write_text(text, encoding="ascii")
        manifest["authors"][slug] = {"units": len(units), "words": len(tokenize(text))}
    write_json(out / "manifest.json", manifest)
    print(f"[{name}] wrote {len(authors)} author files to {out}")
    for slug, info in manifest["authors"].items():
        print(f"    {slug:<12} {info['words']:>8,} words  {info['units']:>5} units")


# ---------------------------------------------------------------- Book of Mormon

BOM_SLUGS = {
    "Nephi, son of Lehi": "nephi",
    "Mormon": "mormon",
    "Moroni, son of Mormon": "moroni",
    "Jacob, son of Lehi": "jacob",
}


def _slug(name: str) -> str:
    return BOM_SLUGS.get(name) or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


SKIP_DIRS = {".git", "node_modules", "__pycache__", "cache", "corpora", "results", ".venv", "venv"}
BOM_TEXT = "book_of_mormon.txt"
BOM_JSON_SUFFIX = "authors_by_chapter.json"


def _search(base: Path, match, depth: int = 4) -> list[Path]:
    """Files under base (bounded depth, skipping build/output folders) satisfying match(name)."""
    found, stack = [], [(base, 0)]
    while stack:
        d, lvl = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_file() and match(e.name):
                found.append(e)
            elif e.is_dir() and lvl < depth and e.name not in SKIP_DIRS:
                stack.append((e, lvl + 1))
    return sorted(found, key=lambda p: (len(p.parts), str(p)))


def _load_bom(arg: str | None) -> tuple[str, dict, str]:
    """Return (book text, attribution json, description of source).

    --bom-source may be a zip, a directory, or either of the two files. Without
    it, the two files are searched for separately in this folder, the current
    directory, and their parent directories, so they need not sit together."""
    if arg:
        src = Path(arg).expanduser()
        if not src.exists():
            sys.exit(f"--bom-source {src} does not exist")
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                names = z.namelist()
                txt = next((n for n in names if n.endswith(BOM_TEXT)), None)
                js = next((n for n in names if n.endswith(BOM_JSON_SUFFIX)), None)
                if not txt or not js:
                    sys.exit(f"{src} must contain {BOM_TEXT} and *{BOM_JSON_SUFFIX}")
                return z.read(txt).decode("utf-8"), json.loads(z.read(js).decode("utf-8")), str(src)
        bases = [src.parent] if src.is_file() else [src]
    else:
        bases = []
        for b in (ROOT, Path.cwd(), ROOT.parent, Path.cwd().parent):
            if b.resolve() not in [x.resolve() for x in bases]:
                bases.append(b)

    txt = next((f for b in bases for f in _search(b, lambda n: n == BOM_TEXT)), None)
    js = next((f for b in bases for f in _search(b, lambda n: n.endswith(BOM_JSON_SUFFIX))), None)
    if not txt and not arg:
        zips = [z for b in bases for z in sorted(b.glob("BookOfMormonStylometry*.zip"))]
        if zips:
            return _load_bom(str(zips[0]))
    if not txt or not js:
        where = ", ".join(str(b.resolve()) for b in bases)
        sys.exit("Book of Mormon source incomplete. Need both files:\n"
                 f"  {BOM_TEXT:<40} {'found: ' + str(txt) if txt else 'NOT FOUND'}\n"
                 f"  {'*' + BOM_JSON_SUFFIX:<40} {'found: ' + str(js) if js else 'NOT FOUND'}\n"
                 f"Searched: {where}\n"
                 "Copy the missing file into the stylometry folder, or pass --bom-source <zip, folder, or file>.")
    print(f"[bom] text: {txt}\n[bom] attributions: {js}")
    return txt.read_text(encoding="utf-8"), json.loads(js.read_text(encoding="utf-8")), f"{txt}; {js}"


def prepare_bom(args) -> None:
    text, meta, src = _load_bom(args.bom_source)

    # Parse verse text by chapter, in document order.
    chapters = []
    current = None
    for line in text.splitlines():
        if re.fullmatch(r"Chapter \d+", line.strip()):
            current = []
            chapters.append(current)
        elif current is not None and re.match(r"^\d+ ", line):
            current.append(re.sub(r"^\d+ ", "", line))

    annotated = [(b["book"], c["chapter"], c["authors"]) for b in meta["books"] for c in b["chapters"]]
    if len(annotated) != len(chapters):
        sys.exit(f"Chapter count mismatch: text {len(chapters)} vs annotation {len(annotated)}")

    sole = defaultdict(list)
    for (book, ch, authors), verses in zip(annotated, chapters):
        if len(authors) == 1:
            sole[authors[0]["name"]].append((f"{book} {ch}", " ".join(verses)))

    ranked = sorted(sole.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = ranked[: args.bom_authors]
    print("[bom] sole-author chapter counts: "
          + ", ".join(f"{n} ({len(v)})" for n, v in ranked))

    authors = {_slug(name): [t for _, t in chs] for name, chs in chosen}
    manifest = {
        "corpus": "bom",
        "source": src,
        "rule": "chapters whose annotation lists exactly one author; verse text only",
        "chapters": {_slug(name): [ref for ref, _ in chs] for name, chs in chosen},
        "author_names": {_slug(name): name for name, _ in chosen},
    }
    write_corpus("bom", authors, manifest)


# ---------------------------------------------------------------- Federalist

FED_BASE = "https://raw.githubusercontent.com/jschulberg/Federalist-Papers-NLP/master/Data/essay{:02d}.txt"
FED_JAY = {2, 3, 4, 5, 64}
FED_MADISON = {10, 14, *range(37, 49)}
FED_JOINT = {18, 19, 20}
FED_DISPUTED = {*range(49, 59), 62, 63}


def prepare_federalist(args) -> None:
    authors = defaultdict(list)
    papers = defaultdict(list)
    for n in range(1, 86):
        if n in FED_JOINT or n in FED_DISPUTED:
            continue
        who = "jay" if n in FED_JAY else "madison" if n in FED_MADISON else "hamilton"
        body = decode(fetch([FED_BASE.format(n)], f"federalist_{n:02d}.txt"))
        body = re.sub(r"\bPUBLIUS\.?\s*$", "", body.strip())
        authors[who].append(body)
        papers[who].append(n)
    manifest = {
        "corpus": "federalist",
        "source": "Project Gutenberg text via github.com/jschulberg/Federalist-Papers-NLP",
        "rule": "standard attribution; disputed (49-58, 62, 63) and joint (18-20) excluded",
        "papers": papers,
    }
    write_corpus("federalist", authors, manifest)


# ---------------------------------------------------------------- KJV NT

KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
KJV_AUTHORS = {
    "paul": ["Romans", "1 Corinthians", "2 Corinthians", "Galatians",
             "Philippians", "1 Thessalonians", "Philemon"],
    "luke": ["Luke", "Acts"],
    "john": ["John", "1 John", "2 John", "3 John"],
}


def prepare_kjv(args) -> None:
    bible = json.loads(decode(fetch([KJV_URL], "en_kjv.json")))
    by_name = {b["name"]: b for b in bible}
    authors = {}
    for slug, books in KJV_AUTHORS.items():
        units = []
        for name in books:
            for chapter in by_name[name]["chapters"]:
                units.append(" ".join(chapter))
        authors[slug] = units
    manifest = {
        "corpus": "kjv_nt",
        "source": KJV_URL,
        "rule": "traditional attribution; disputed Pauline epistles, Hebrews, and Revelation excluded",
        "books": KJV_AUTHORS,
    }
    write_corpus("kjv_nt", authors, manifest)


# ---------------------------------------------------------------- Brontes

BRONTE_BOOKS = {
    "charlotte": [("Jane-Eyre_1260", "1260"), ("Villette_9182", "9182"),
                  ("Shirley_30486", "30486"), ("The-Professor_1028", "1028")],
    "emily": [("Wuthering-Heights_768", "768")],
    "anne": [("Agnes-Grey_767", "767"), ("The-Tenant-of-Wildfell-Hall_969", "969")],
}
CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLC]+|\d+)\b.*$", re.IGNORECASE)
END_RE = re.compile(r"^\s*(THE END\.?|\*{3} ?END OF TH.*|\*{3}END OF TH.*|End of (the )?Project Gutenberg.*)\s*$"
                    r"|Spottiswode", re.IGNORECASE)


def _gutenberg_body(raw: str) -> list[str]:
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Start: first "CHAPTER I" heading that is followed by real prose
    # (skips tables of contents, title pages, and prefaces).
    heads = [i for i, l in enumerate(lines) if CHAPTER_RE.match(l)]
    start = None
    for k, i in enumerate(heads):
        if CHAPTER_RE.match(lines[i]).group(1).upper() not in ("I", "1"):
            continue
        nxt = heads[k + 1] if k + 1 < len(heads) else len(lines)
        if nxt - i > 20:
            start = i
            break
    if start is None:
        raise ValueError("no chapter body found")
    end = next((j for j in range(start, len(lines)) if END_RE.search(lines[j])), len(lines))
    body = "\n".join(lines[start:end])
    body = re.sub(r"\[(Illustration|Picture)[^\]]*\]", " ", body, flags=re.S)
    body = re.sub(r"\{\d+\}", " ", body)

    paragraphs = []
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or CHAPTER_RE.match(p):
            continue
        if re.fullmatch(r"[\*\s]+", p):          # "* * * * *" breaks
            continue
        if not re.search(r"[a-z]", p) and len(p.split()) <= 25:  # chapter titles like "LEVITICAL."
            continue
        paragraphs.append(p)
    return paragraphs


def prepare_bronte(args) -> None:
    authors = {}
    for slug, books in BRONTE_BOOKS.items():
        units = []
        for repo, gid in books:
            urls = [f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{gid}.txt",
                    f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"]
            units += _gutenberg_body(decode(fetch(urls, f"gutenberg_{gid}.txt")))
        authors[slug] = units
    manifest = {
        "corpus": "bronte",
        "source": "Project Gutenberg via github.com/GITenberg",
        "rule": "novel text from Chapter I to end; prefaces, headings, notes, publisher matter removed",
        "books": {k: [r for r, _ in v] for k, v in BRONTE_BOOKS.items()},
    }
    write_corpus("bronte", authors, manifest)


# ---------------------------------------------------------------- main

PREPARERS = {"bom": prepare_bom, "federalist": prepare_federalist,
             "kjv_nt": prepare_kjv, "bronte": prepare_bronte}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all", choices=CORPORA + ["all"])
    ap.add_argument("--bom-source", help="Book of Mormon zip or directory (auto-detected if omitted)")
    ap.add_argument("--bom-authors", type=int, default=3,
                    help="number of Book of Mormon authors, ranked by sole-author chapters (default 3)")
    args = ap.parse_args()
    for name in (CORPORA if args.corpus == "all" else [args.corpus]):
        PREPARERS[name](args)


if __name__ == "__main__":
    main()