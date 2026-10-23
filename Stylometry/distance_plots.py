#!/usr/bin/env python3
"""
distance_plots.py  (Program 2 of 4)

Given labeled text segments and a list of words, turns each segment into a vector of word
percentages, computes the distance between every pair of segments, and for each author
draws a stacked histogram of that author's distances to all other segments. Bars are colored
by the author of the segment at the other end of each distance. Also reports how well the
word list separates the authors (nearest-neighbor accuracy with a permutation test).
Works on ANY labeled text segments.

Input
-----
Same segment file as word_stability.py (.jsonl, .json or .csv with author, text, optional doc).

Word lists (--words)
--------------------
  top10         the of and to a in that is was he
  marker        upon enough whilst while by on to this there any an also
  fw70          Mosteller-Wallace 70 function words
  mfwN          the N most frequent words in the input (e.g. mfw100)
  custom:a,b,c  your own list
  or use --words-file with one word per line

Metrics (--metric)
------------------
  delta    Burrows' Delta: mean absolute z-score difference (default)
  zeuclid  Euclidean on z-scored percentages
  cosine   cosine distance on z-scored percentages
  euclid   Euclidean on raw percentages (dominated by the most frequent words)

Examples
--------
  python distance_plots.py segments.jsonl --words fw70
  python distance_plots.py segments.csv --words custom:upon,on,by,to --metric zeuclid
  python distance_plots.py segments.jsonl --words mfw100 --colors "A=#d62728,B=#1f77b4"
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOP10_WORDS = ["the", "of", "and", "to", "a", "in", "that", "is", "was", "he"]
MARKER_WORDS = ["upon", "enough", "whilst", "while", "by", "on", "to", "this", "there", "any", "an", "also"]
FUNCTION_WORDS_70 = [
    "a", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "been", "but", "by", "can", "do", "down", "even", "every", "for", "from",
    "had", "has", "have", "her", "his", "if", "in", "into", "is", "it",
    "its", "may", "more", "must", "my", "no", "not", "now", "of", "on",
    "one", "only", "or", "our", "shall", "should", "so", "some", "such", "than",
    "that", "the", "their", "then", "there", "thing", "this", "to", "up", "upon",
    "was", "were", "what", "when", "which", "who", "will", "with", "would", "your",
]
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

METRIC_LABELS = {
    "delta": "Burrows' Delta (mean |z difference|)",
    "zeuclid": "Euclidean distance (z-scored)",
    "cosine": "Cosine distance (z-scored)",
    "euclid": "Euclidean distance (percentage points)",
}


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
def tokenize(text):
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def read_segments(path):
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


def prepare(segments, min_words=1, min_segments=1):
    kept = [s for s in segments if len(s["tokens"]) >= min_words]
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


def resolve_words(spec, segments, words_file=None):
    if words_file:
        return [w.strip().lower() for w in Path(words_file).read_text(encoding="utf-8").splitlines() if w.strip()]
    spec = (spec or "top10").strip()
    if spec == "top10":
        return list(TOP10_WORDS)
    if spec == "marker":
        return list(MARKER_WORDS)
    if spec == "fw70":
        return list(FUNCTION_WORDS_70)
    m = re.fullmatch(r"mfw(\d+)", spec)
    if m:
        counts = Counter()
        for s in segments:
            counts.update(s["tokens"])
        return [w for w, _ in counts.most_common(int(m.group(1)))]
    if spec.startswith("custom:"):
        words = [w.strip().lower() for w in spec[7:].split(",") if w.strip()]
        if words:
            return words
    raise SystemExit(f"Unknown or empty word list: {spec}")


def parse_colors(spec, authors):
    colors = {}
    if spec:
        for part in spec.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                colors[k.strip()] = v.strip()
    for i, a in enumerate(authors):
        colors.setdefault(a, PALETTE[i % len(PALETTE)])
    return colors


# ----------------------------------------------------------------------------
# Vectors and distances
# ----------------------------------------------------------------------------
def build_matrix(segments, words):
    index = {w: i for i, w in enumerate(words)}
    X = np.zeros((len(segments), len(words)))
    for r, s in enumerate(segments):
        n = len(s["tokens"])
        for w, c in Counter(s["tokens"]).items():
            j = index.get(w)
            if j is not None:
                X[r, j] = 100.0 * c / n
    labels = np.array([s["author"] for s in segments])
    docs = np.array([s["doc"] for s in segments])
    return X, labels, docs


def zscore(X):
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - X.mean(axis=0)) / std


def distance_matrix(X, metric):
    if metric == "euclid":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
    Z = zscore(X)
    if metric == "zeuclid":
        diff = Z[:, None, :] - Z[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
    if metric == "delta":
        return np.abs(Z[:, None, :] - Z[None, :, :]).mean(axis=2)
    if metric == "cosine":
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return np.clip(1.0 - Zn @ Zn.T, 0.0, 2.0)
    raise ValueError(metric)


# ----------------------------------------------------------------------------
# Separability
# ----------------------------------------------------------------------------
def nearest_neighbors(D, docs, exclude_same_doc):
    D2 = D.copy()
    if exclude_same_doc:
        D2[docs[:, None] == docs[None, :]] = np.inf
    else:
        np.fill_diagonal(D2, np.inf)
    return D2.argmin(axis=1)


def balanced_accuracy(pred, labels, authors):
    return float(np.mean([(pred[labels == a] == a).mean() for a in authors if (labels == a).any()]))


def evaluate(segments, authors, words, metric="delta", exclude_same_doc=True, n_perm=1000, seed=0):
    """Distances, nearest-neighbor accuracy and (optionally) a document-level permutation p-value."""
    X, labels, docs = build_matrix(segments, words)
    D = distance_matrix(X, metric)
    nn = nearest_neighbors(D, docs, exclude_same_doc)
    pred = labels[nn]
    obs = balanced_accuracy(pred, labels, authors)
    p = None
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(docs, return_inverse=True)
        doc_labels = np.array([labels[inv == i][0] for i in range(len(uniq))])
        count = 0
        for _ in range(n_perm):
            pl = rng.permutation(doc_labels)[inv]
            if balanced_accuracy(pl[nn], pl, authors) >= obs:
                count += 1
        p = (count + 1) / (n_perm + 1)
    return {"D": D, "labels": labels, "docs": docs, "pred": pred, "balanced_accuracy": obs,
            "permutation_p": p, "chance": 1 / len(authors)}


def print_report(ev, authors, exclude_same_doc, n_perm):
    labels, pred = ev["labels"], ev["pred"]
    print("  Nearest-neighbor classification"
          + (" (neighbors from the same document excluded)" if exclude_same_doc else ""))
    msg = f"    chance (balanced): {ev['chance']:.1%}   observed balanced accuracy: {ev['balanced_accuracy']:.1%}"
    if ev["permutation_p"] is not None:
        msg += f"   permutation p = {ev['permutation_p']:.4f} ({n_perm} shuffles)"
    print(msg)
    print("    true \\ predicted " + "".join(f"{a[:10]:>12}" for a in authors))
    for a in authors:
        m = labels == a
        row = "".join(f"{(pred[m] == b).sum():>12}" for b in authors)
        print(f"    {a[:16]:<17}{row}   recall {(pred[m] == a).mean():.1%} (corpus share {m.mean():.1%})")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_author(D, labels, docs, author, colors, authors, title, outfile, bins, metric, weighted, exclude_same_doc):
    idx = np.where(labels == author)[0]
    dists, others = [], []
    for i in idx:
        mask = docs != docs[i] if exclude_same_doc else np.arange(len(labels)) != i
        dists.append(D[i, mask])
        others.append(labels[mask])
    dists, others = np.concatenate(dists), np.concatenate(others)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(bins) - 1)
    for other in authors:
        sel = others == other
        if not sel.any():
            continue
        w = np.full(sel.sum(), 1.0 / sel.sum()) if weighted else None
        counts, _ = np.histogram(dists[sel], bins=bins, weights=w)
        ax.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom, align="edge",
               color=colors[other], edgecolor="white", linewidth=0.4, label=f"to {other}")
        bottom += counts
    ax.set_xlabel(f"Distance to other segments: {METRIC_LABELS[metric]}")
    ax.set_ylabel("Share of pairs to each author (each color sums to 1)" if weighted else "Number of segment pairs")
    ax.set_title(f"{title}{author} ({len(idx)} segments) vs. other segments")
    ax.legend(title="Other segment's author")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


def run_plots(segments, authors, words, out_dir, metric="delta", bins=30, weighted=True,
              exclude_same_doc=True, n_perm=1000, colors=None, title="", prefix="distance"):
    if len({s["doc"] for s in segments}) < 2:
        raise SystemExit("Need segments from at least two documents.")
    colors = colors or parse_colors(None, authors)
    print(f"\nDistance analysis: {len(segments)} segments, {len(words)} words, metric = {metric}")
    ev = evaluate(segments, authors, words, metric, exclude_same_doc, n_perm)
    print_report(ev, authors, exclude_same_doc, n_perm)

    D, labels, docs = ev["D"], ev["labels"], ev["docs"]
    off = D[docs[:, None] != docs[None, :]] if exclude_same_doc else D[~np.eye(len(D), dtype=bool)]
    edges = np.linspace(0, off.max(), bins + 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in authors:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", a).lower()
        plot_author(D, labels, docs, a, colors, authors, f"{title}: " if title else "",
                    out_dir / f"{prefix}_{safe}.png", edges, metric, weighted, exclude_same_doc)
    return ev


def add_arguments(p):
    p.add_argument("--metric", choices=list(METRIC_LABELS), default="delta")
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--no-weight", action="store_true", help="plot raw pair counts instead of per-author shares")
    p.add_argument("--allow-same-doc", dest="exclude_same_doc", action="store_false",
                   help="include pairs from the same document (inflates same-author similarity)")
    p.add_argument("--permutations", type=int, default=1000, help="permutation-test shuffles (0 to skip)")
    p.add_argument("--colors", help='author colors, e.g. "A=#d62728,B=#1f77b4"')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="segments file (.jsonl, .json or .csv)")
    ap.add_argument("--words", default="top10", help="top10 | marker | fw70 | mfwN | custom:a,b,c")
    ap.add_argument("--words-file", help="one word per line (overrides --words)")
    ap.add_argument("--min-words", type=int, default=1, help="discard segments shorter than this")
    ap.add_argument("--out-dir", default="output")
    add_arguments(ap)
    a = ap.parse_args()
    segments, authors = prepare(read_segments(a.input), a.min_words)
    if len(authors) < 2:
        raise SystemExit("Need at least two authors.")
    words = resolve_words(a.words, segments, a.words_file)
    run_plots(segments, authors, words, a.out_dir, a.metric, a.bins, not a.no_weight,
              a.exclude_same_doc, a.permutations, parse_colors(a.colors, authors))


if __name__ == "__main__":
    main()