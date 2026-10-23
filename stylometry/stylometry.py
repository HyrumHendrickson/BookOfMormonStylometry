#!/usr/bin/env python3
"""Program 4: compute segment-to-segment stylistic distances and plot them.

Each segment is a point whose coordinates are the relative frequencies
(percent) of the words chosen by select_words.py. Each coordinate is
z-scored across all segments so that every word carries equal weight
(otherwise "the" would dominate). Distances between every pair of segments
are then computed with one of:

  cosine     Cosine Delta: 1 - cosine similarity of z-score vectors
             (Smith & Aldridge 2011; most robust in Evert et al. 2017).
  delta      Burrows' Delta: mean absolute difference of z-scores (Burrows 2002).
  euclidean  Euclidean distance of z-score vectors.

For each author a histogram shows the distances from that author's segments
to the author's own other segments (in the author's colour) and to each
other author's segments (one colour per author). Clear authorial signal
appears as a same-author bump to the left of the other bumps.

Summary statistics written to summary.json:
  AUC      probability that a random same-author distance is smaller than a
           random different-author distance (0.5 = no signal, 1.0 = perfect).
  1-NN     leave-one-out accuracy: each segment is assigned the author of its
           nearest other segment.

Usage
  python stylometry.py results/federalist/selected_words.json
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import author_colors, holdout_split, load_corpus, read_json, segment, tokenize, write_json


def build_matrix(corpus_dir: Path, features: list[str], size: int, part: str):
    corpus = load_corpus(corpus_dir)
    rows, labels = [], []
    for au in sorted(corpus):
        segs = segment(tokenize(corpus[au]), size)
        for i in holdout_split(len(segs), part):
            c = Counter(segs[i])
            rows.append([100.0 * c.get(w, 0) / size for w in features])
            labels.append(au)
    return np.array(rows), np.array(labels)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def distances(z: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        n = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        d = 1.0 - n @ n.T
    elif metric == "euclidean":
        sq = (z ** 2).sum(axis=1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0))
    elif metric == "delta":
        d = np.abs(z[:, None, :] - z[None, :, :]).mean(axis=2)
    else:
        raise ValueError(metric)
    np.fill_diagonal(d, 0.0)
    return d


def auc(within: np.ndarray, between: np.ndarray) -> float:
    """P(within < between) + 0.5 * P(tie), computed exactly by sorting."""
    b = np.sort(between)
    lt = len(b) - np.searchsorted(b, within, side="right")
    eq = np.searchsorted(b, within, side="right") - np.searchsorted(b, within, side="left")
    return float((lt + 0.5 * eq).sum() / (len(within) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selection", type=Path, help="selected_words.json from select_words.py")
    ap.add_argument("--metric", choices=["cosine", "delta", "euclidean"], default="cosine")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--title", help="figure title prefix (default: corpus folder name)")
    a = ap.parse_args()

    sel = read_json(a.selection)
    out_dir = a.selection.parent
    corpus_dir = Path(sel["corpus_dir"])
    features = sel["features"]
    size = sel["settings"]["segment_size"]
    part = "evaluate" if sel["settings"]["holdout"] else "all"
    name = a.title or corpus_dir.name

    x, labels = build_matrix(corpus_dir, features, size, part)
    d = distances(zscore(x), a.metric)
    authors = sorted(set(labels))
    colors = author_colors(authors)
    idx = {au: np.where(labels == au)[0] for au in authors}

    # 1-NN leave-one-out attribution.
    d_nn = d + np.diag(np.full(len(d), np.inf))
    predicted = labels[np.argmin(d_nn, axis=1)]
    confusion = {au: dict(Counter(predicted[idx[au]])) for au in authors}

    lo, hi = float(d[np.triu_indices(len(d), 1)].min()), float(d.max())
    bins = np.linspace(lo, hi, a.bins + 1)
    summary = {"corpus": name, "metric": a.metric, "segment_size": size,
               "segments": {au: int(len(idx[au])) for au in authors},
               "features": features, "evaluated_on": part, "authors": {}}

    n = len(authors)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False)
    for ax, au in zip(axes[:, 0], authors):
        own = idx[au]
        within = d[np.ix_(own, own)][np.triu_indices(len(own), 1)]
        series = [(f"{au} (same author)", within, colors[au])]
        between_all = []
        for b in authors:
            if b == au:
                continue
            vals = d[np.ix_(own, idx[b])].ravel()
            between_all.append(vals)
            series.append((b, vals, colors[b]))
        between_all = np.concatenate(between_all)

        for lbl, vals, col in series:
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color=col, edgecolor=col, linewidth=1.4, label=f"{lbl}  (n={len(vals):,})")
        ax.set_title(f"{name}: distances from {au}'s segments", fontsize=11)
        ax.set_xlabel(f"{a.metric} distance")
        ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)

        acc = float(np.mean(predicted[own] == au))
        summary["authors"][au] = {
            "mean_within": float(within.mean()) if len(within) else None,
            "mean_between": {b: float(d[np.ix_(own, idx[b])].mean()) for b in authors if b != au},
            "auc_within_vs_between": auc(within, between_all) if len(within) else None,
            "nn_accuracy": acc,
            "nn_assigned_to": confusion[au],
        }

        single, sax = plt.subplots(figsize=(9, 4))
        for lbl, vals, col in series:
            sax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                     color=col, edgecolor=col, linewidth=1.4, label=f"{lbl}  (n={len(vals):,})")
        sax.set_title(f"{name}: distances from {au}'s segments", fontsize=11)
        sax.set_xlabel(f"{a.metric} distance")
        sax.set_ylabel("density")
        sax.legend(fontsize=8, frameon=False)
        single.tight_layout()
        single.savefig(out_dir / f"hist_{au}.png", dpi=150)
        plt.close(single)

    fig.tight_layout()
    fig.savefig(out_dir / "hist_all_authors.png", dpi=150)
    plt.close(fig)

    summary["overall_nn_accuracy"] = float(np.mean(predicted == labels))
    write_json(out_dir / "summary.json", summary)
    np.savetxt(out_dir / "distance_matrix.csv", d, delimiter=",", fmt="%.6f",
               header=",".join(f"{l}_{i}" for i, l in enumerate(labels)), comments="")

    print(f"{name}: {len(labels)} segments x {len(features)} words, metric={a.metric}, evaluated on {part}")
    print(f"    {'author':<12}{'segs':>6}{'within':>9}{'between':>9}{'AUC':>7}{'1-NN':>7}")
    for au in authors:
        s = summary["authors"][au]
        mb = np.mean(list(s["mean_between"].values()))
        w = s["mean_within"]
        print(f"    {au:<12}{len(idx[au]):>6}{(w if w is not None else float('nan')):>9.3f}{mb:>9.3f}"
              f"{(s['auc_within_vs_between'] or float('nan')):>7.3f}{s['nn_accuracy']:>7.1%}")
    print(f"    overall 1-NN accuracy: {summary['overall_nn_accuracy']:.1%}")
    print(f"    wrote {out_dir}/hist_all_authors.png, hist_<author>.png, summary.json, distance_matrix.csv")


if __name__ == "__main__":
    main()
