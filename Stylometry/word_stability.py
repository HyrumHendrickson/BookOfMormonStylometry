#!/usr/bin/env python3
"""
word_stability.py  (Program 1 of 4)

Measures how stable each common word is within each author, so that stable words can be
chosen as stylometric features. Works on ANY labeled text segments.

Input
-----
A file of pre-cut segments, each with an author label. Accepted formats:
  .jsonl   one JSON object per line:  {"author": "A", "doc": "essay 1", "text": "..."}
  .json    a list of such objects
  .csv     columns: author, text, and optionally doc
"doc" (optional; also read from "document", "id", "source") names the larger document a
segment came from. It lets the other programs avoid comparing a segment with its own
document's neighbors. Without it, every segment is treated as its own document.

Output
------
  <out-dir>/word_stability.csv   one row per word, sorted by worst_cv
  console ranking and a ready-to-paste word list

For each word and author:
  mean_pct    mean percentage of segment words equal to the word
  std_pct     sample standard deviation of that percentage across the author's segments
  cv          std / mean (scale-free, so rare and common words compare fairly)
  dispersion  observed std / std expected from sampling noise alone (about 1.0 = chance-level)
Across authors:
  worst_cv    largest cv among authors (a good word is stable for every author)
  mean_cv     average cv
  f_ratio     one-way ANOVA F (between-author / within-author variance); high = separates authors

Examples
--------
  python word_stability.py segments.jsonl
  python word_stability.py segments.csv --top-n 500 --min-words 300 --top-k 20
  python word_stability.py segments.jsonl --rank-by separation --max-cv 0.5
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
def tokenize(text):
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def read_segments(path):
    """Return a list of {"author", "doc", "tokens"}."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        raise SystemExit(f"Unsupported input type '{suffix}'. Use .jsonl, .json or .csv")

    segments = []
    for i, r in enumerate(records):
        if "author" not in r or ("text" not in r and "tokens" not in r):
            raise SystemExit(f"Record {i} needs 'author' and 'text' fields")
        doc = next((str(r[k]) for k in ("doc", "document", "id", "source") if r.get(k) not in (None, "")),
                   f"segment_{i}")
        tokens = r["tokens"] if "tokens" in r else tokenize(r["text"])
        segments.append({"author": str(r["author"]), "doc": doc, "tokens": tokens})
    return segments


def prepare(segments, min_words=1, min_segments=2):
    """Drop short segments and authors with too few segments. Returns (segments, authors)."""
    kept = [s for s in segments if len(s["tokens"]) >= min_words]
    if len(kept) < len(segments):
        print(f"  Dropped {len(segments) - len(kept)} segments shorter than {min_words} words")
    counts = Counter(s["author"] for s in kept)
    authors = []
    for s in kept:
        if s["author"] not in authors:
            authors.append(s["author"])
    good = [a for a in authors if counts[a] >= min_segments]
    for a in authors:
        if a not in good:
            print(f"  Excluding {a}: only {counts[a]} segment(s) (need {min_segments})")
    return [s for s in kept if s["author"] in good], good


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def build_vocab(segments, top_n=1000, words_file=None):
    if words_file:
        words = [w.strip().lower() for w in Path(words_file).read_text(encoding="utf-8").splitlines() if w.strip()]
        return words[:top_n]
    counts = Counter()
    for s in segments:
        counts.update(s["tokens"])
    return [w for w, _ in counts.most_common(top_n)]


def one_way_f(groups):
    groups = [g for g in groups if g.shape[0] >= 1]
    k = len(groups)
    N = sum(g.shape[0] for g in groups)
    if k < 2 or N - k < 1:
        return None
    grand = np.vstack(groups).mean(axis=0)
    ssb = sum(g.shape[0] * (g.mean(axis=0) - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean(axis=0)) ** 2).sum(axis=0) for g in groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ssb / (k - 1)) / (ssw / (N - k))


def compute_stability(segments, authors, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    stats, groups = {}, []
    for a in authors:
        segs = [s for s in segments if s["author"] == a]
        counts = np.zeros((len(segs), len(vocab)))
        n = np.zeros(len(segs))
        for r, s in enumerate(segs):
            n[r] = len(s["tokens"])
            for w, c in Counter(s["tokens"]).items():
                j = index.get(w)
                if j is not None:
                    counts[r, j] = c
        P = 100.0 * counts / n[:, None]
        groups.append(P)
        mean = P.mean(axis=0)
        std = P.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(mean > 0, std / mean, np.nan)
            p_pool = counts.sum(axis=0) / n.sum()
            exp_var = 1e4 * np.mean(p_pool[None, :] * (1 - p_pool[None, :]) / n[:, None], axis=0)
            disp = np.where(exp_var > 0, std / np.sqrt(exp_var), np.nan)
        stats[a] = {"mean": mean, "std": std, "cv": cv, "dispersion": disp, "n_segments": len(segs),
                    "n_docs": len({s["doc"] for s in segs})}

    cvs = np.array([stats[a]["cv"] for a in authors])
    undefined = np.isnan(cvs).any(axis=0)
    with np.errstate(all="ignore"):
        worst_cv = np.where(undefined, np.nan, np.nanmax(np.where(np.isnan(cvs), -np.inf, cvs), axis=0))
        mean_cv = np.where(undefined, np.nan, cvs.mean(axis=0))
    return {
        "vocab": vocab,
        "authors": authors,
        "stats": stats,
        "worst_cv": worst_cv,
        "mean_cv": mean_cv,
        "f_ratio": one_way_f(groups) if len(groups) >= 2 else None,
        "min_mean": np.min([stats[a]["mean"] for a in authors], axis=0),
    }


def select_words(result, top_k=12, min_mean_pct=0.1, max_cv=None, rank_by="stability"):
    """Return the chosen words (list) and their indices."""
    worst, f = result["worst_cv"], result["f_ratio"]
    idx = [i for i in range(len(result["vocab"]))
           if not np.isnan(worst[i]) and result["min_mean"][i] >= min_mean_pct
           and (max_cv is None or worst[i] <= max_cv)]
    if rank_by == "stability":
        idx.sort(key=lambda i: worst[i])
    else:
        if f is None:
            raise SystemExit("--rank-by separation/combined needs at least two authors")
        if rank_by == "separation":
            idx.sort(key=lambda i: -np.nan_to_num(f[i], nan=-1))
        else:  # combined
            idx.sort(key=lambda i: -np.nan_to_num(f[i] / max(worst[i], 1e-9), nan=-1))
    idx = idx[:top_k]
    return [result["vocab"][i] for i in idx], idx


def write_csv(result, path):
    authors, stats, vocab = result["authors"], result["stats"], result["vocab"]
    header = ["corpus_rank", "word"]
    for a in authors:
        header += [f"{a}_mean_pct", f"{a}_std_pct", f"{a}_cv", f"{a}_dispersion"]
    header += ["worst_cv", "mean_cv", "f_ratio"]
    worst = result["worst_cv"]
    order = np.argsort(np.where(np.isnan(worst), np.inf, worst))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in order:
            row = [i + 1, vocab[i]]
            for a in authors:
                s = stats[a]
                row += [f"{s['mean'][i]:.5f}", f"{s['std'][i]:.5f}", f"{s['cv'][i]:.4f}", f"{s['dispersion'][i]:.3f}"]
            row += [f"{worst[i]:.4f}", f"{result['mean_cv'][i]:.4f}",
                    "" if result["f_ratio"] is None else f"{result['f_ratio'][i]:.3f}"]
            w.writerow(row)


def print_ranking(result, idx, show=40):
    authors, stats, vocab = result["authors"], result["stats"], result["vocab"]
    print(f"    {'word':<14}" + "".join(f"{a[:10] + ' sd':>15}" for a in authors) + f"{'worst_cv':>10}{'F':>9}")
    for i in idx[:show]:
        line = f"    {vocab[i]:<14}" + "".join(f"{stats[a]['std'][i]:>15.4f}" for a in authors)
        line += f"{result['worst_cv'][i]:>10.3f}"
        line += "" if result["f_ratio"] is None else f"{result['f_ratio'][i]:>9.2f}"
        print(line)


# ----------------------------------------------------------------------------
# High-level entry point (also used by pipeline.py)
# ----------------------------------------------------------------------------
def run_stability(segments, out_dir, top_n=1000, words_file=None, top_k=12, min_mean_pct=0.1,
                  max_cv=None, rank_by="stability", show=40, csv_name="word_stability.csv"):
    segments, authors = prepare_cached(segments)
    vocab = build_vocab(segments, top_n, words_file)
    result = compute_stability(segments, authors, vocab)
    out_csv = Path(out_dir) / csv_name
    write_csv(result, out_csv)
    words, idx = select_words(result, top_k, min_mean_pct, max_cv, rank_by)
    print(f"\nWord stability: {len(vocab)} candidate words, authors: "
          + ", ".join(f"{a} ({result['stats'][a]['n_segments']} segments)" for a in authors))
    print(f"  Saved {out_csv}")
    print(f"  Top {len(words)} words (rank by {rank_by}; every author's mean >= {min_mean_pct}%"
          + (f"; worst_cv <= {max_cv}" if max_cv is not None else "") + "):")
    print_ranking(result, idx, show)
    print(f"\n  WORDS = {words}")
    return words, result


def prepare_cached(segments):
    """Used when segments were already prepared by the caller (authors preserved in order)."""
    authors = []
    for s in segments:
        if s["author"] not in authors:
            authors.append(s["author"])
    return segments, authors


def add_arguments(p):
    p.add_argument("--top-n", type=int, default=1000, help="examine the N most frequent words (default 1000)")
    p.add_argument("--top-k", type=int, default=12, help="how many words to select (default 12)")
    p.add_argument("--words-file", help="text file, one word per line, replacing the frequency-derived vocabulary")
    p.add_argument("--min-mean-pct", type=float, default=0.1,
                   help="selection only: each author's mean must be >= this percent (default 0.1)")
    p.add_argument("--max-cv", type=float, default=None, help="selection only: require worst_cv <= this")
    p.add_argument("--rank-by", choices=["stability", "separation", "combined"], default="stability",
                   help="stability: lowest worst_cv; separation: highest F; combined: F / worst_cv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="segments file (.jsonl, .json or .csv)")
    ap.add_argument("--min-words", type=int, default=1, help="discard segments shorter than this (default 1)")
    ap.add_argument("--min-segments", type=int, default=2, help="authors need at least this many segments")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--show", type=int, default=40, help="rows in the console ranking")
    add_arguments(ap)
    a = ap.parse_args()
    segments, authors = prepare(read_segments(a.input), a.min_words, a.min_segments)
    if len(authors) < 1:
        raise SystemExit("No author has enough segments.")
    run_stability(segments, a.out_dir, a.top_n, a.words_file, a.top_k, a.min_mean_pct, a.max_cv, a.rank_by, a.show)


if __name__ == "__main__":
    main()