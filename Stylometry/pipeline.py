#!/usr/bin/env python3
"""
pipeline.py  (Program 3 of 4)

Takes ANY text that has been cut into segments with authors labeled, then:
  1. runs word_stability.py to find the words that are most stable within every author,
  2. runs distance_plots.py with exactly those words to draw one histogram per author and
     measure how well the words separate the authors,
  3. compares the chosen words against two baselines (the same number of most-frequent words,
     and random words from the same candidate pool) so the selection is judged fairly.

Input
-----
A .jsonl, .json or .csv file of segments with fields author, text and optionally doc
(the larger document each segment came from; see word_stability.py).

Honesty about the selection step
--------------------------------
Choosing words with the same labeled data you then evaluate on is circular, most strongly
when ranking by separation. Use --holdout 0.5 to pick words from half of each author's
documents and draw the plots and run the tests on the other half.

Output (in --out-dir)
---------------------
  word_stability.csv     per-word, per-author statistics
  selected_words.txt     the chosen words, one per line
  distance_<author>.png  stacked histograms
  summary.json           chosen words, accuracy, p-value and baseline comparison

Examples
--------
  python pipeline.py segments.jsonl
  python pipeline.py segments.csv --top-k 20 --metric zeuclid --holdout 0.5
  python pipeline.py segments.jsonl --rank-by combined --max-cv 0.6 --colors "A=#d62728,B=#1f77b4"
"""

import argparse
import json
from pathlib import Path

import numpy as np

import distance_plots as dp
import word_stability as ws


def build_parser(description=None):
    p = argparse.ArgumentParser(description=description or __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-words", type=int, default=1, help="discard segments shorter than this (default 1)")
    p.add_argument("--min-segments", type=int, default=2, help="authors need at least this many segments")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--show", type=int, default=40, help="rows in the console ranking")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="fraction of each author's documents used ONLY for word selection; "
                        "the rest are used for plots and tests (default 0 = use everything for both)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-draws", type=int, default=50,
                   help="random word sets compared against the selection (0 to skip baselines)")
    ws.add_arguments(p)
    dp.add_arguments(p)
    return p


def split_by_document(segments, authors, frac, seed):
    """Return (selection_segments, evaluation_segments), split by document within each author."""
    rng = np.random.default_rng(seed)
    sel_docs = set()
    for a in authors:
        docs = sorted({s["doc"] for s in segments if s["author"] == a})
        if len(docs) < 2:
            print(f"  Warning: {a} has one document; it is used for both selection and evaluation")
            sel_docs.update(docs)
            continue
        rng.shuffle(docs)
        n_sel = min(max(1, round(frac * len(docs))), len(docs) - 1)
        sel_docs.update(docs[:n_sel])
    sel = [s for s in segments if s["doc"] in sel_docs]
    ev = [s for s in segments if s["doc"] not in sel_docs]
    # authors with a single document are in selection only; give them to evaluation as well
    for a in authors:
        if not any(s["author"] == a for s in ev):
            ev += [s for s in segments if s["author"] == a]
    return sel, ev


def baseline_comparison(ev_segments, authors, words, pool, opts):
    k = len(words)
    rng = np.random.default_rng(opts.seed + 1)
    accs = []
    for _ in range(opts.baseline_draws):
        draw = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        accs.append(dp.evaluate(ev_segments, authors, draw, opts.metric, opts.exclude_same_doc, 0)["balanced_accuracy"])
    mfw = dp.resolve_words(f"mfw{k}", ev_segments)
    mfw_acc = dp.evaluate(ev_segments, authors, mfw, opts.metric, opts.exclude_same_doc, 0)["balanced_accuracy"]
    sel_acc = dp.evaluate(ev_segments, authors, words, opts.metric, opts.exclude_same_doc, 0)["balanced_accuracy"]
    accs = np.array(accs)
    return {
        "selected_balanced_accuracy": sel_acc,
        f"top_{k}_frequent_words_balanced_accuracy": mfw_acc,
        "random_words_mean": float(accs.mean()),
        "random_words_95th_percentile": float(np.percentile(accs, 95)),
        "selected_beats_random_fraction": float((accs < sel_acc).mean()),
        "draws": len(accs),
    }


def run_pipeline(segments, opts, colors=None, title="", out_dir=None):
    out_dir = Path(out_dir or opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, authors = ws.prepare(segments, opts.min_words, opts.min_segments)
    if len(authors) < 2:
        raise SystemExit("Need at least two authors with enough segments.")

    if opts.holdout > 0:
        sel, ev = split_by_document(segments, authors, opts.holdout, opts.seed)
        sel, sel_authors = ws.prepare(sel, 1, opts.min_segments)
        missing = [a for a in authors if a not in sel_authors]
        if missing:
            print(f"  Warning: {', '.join(missing)} lack enough selection segments; "
                  "selected words are NOT checked for stability in them. Lower --holdout or --min-segments.")
        ev, ev_authors = ws.prepare(ev, 1, 1)
        print(f"\nHold-out: {len(sel)} segments for word selection, {len(ev)} for plots and tests")
    else:
        sel, sel_authors, ev, ev_authors = segments, authors, segments, authors
        print("\nNote: words are selected and evaluated on the same segments (use --holdout 0.5 to separate them)")

    words, result = ws.run_stability(sel, out_dir, opts.top_n, opts.words_file, opts.top_k,
                                     opts.min_mean_pct, opts.max_cv, opts.rank_by, opts.show)
    if not words:
        raise SystemExit("No words met the selection criteria; loosen --min-mean-pct or --max-cv.")
    (out_dir / "selected_words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")

    colors = colors or dp.parse_colors(opts.colors, ev_authors)
    ev_result = dp.run_plots(ev, ev_authors, words, out_dir, opts.metric, opts.bins, not opts.no_weight,
                             opts.exclude_same_doc, opts.permutations, colors, title)

    summary = {
        "title": title,
        "authors": ev_authors,
        "selected_words": words,
        "rank_by": opts.rank_by,
        "metric": opts.metric,
        "holdout": opts.holdout,
        "balanced_accuracy": ev_result["balanced_accuracy"],
        "chance": ev_result["chance"],
        "permutation_p": ev_result["permutation_p"],
    }
    if opts.baseline_draws > 0:
        base = baseline_comparison(ev, ev_authors, words, result["vocab"], opts)
        summary["baselines"] = base
        k = len(words)
        print(f"\n  Baseline comparison (balanced accuracy, same metric and segments):")
        print(f"    selected {k} words:                {base['selected_balanced_accuracy']:.1%}")
        print(f"    {k} most frequent words:           {base[f'top_{k}_frequent_words_balanced_accuracy']:.1%}")
        print(f"    {k} random words ({base['draws']} draws): mean {base['random_words_mean']:.1%}, "
              f"95th percentile {base['random_words_95th_percentile']:.1%}; "
              f"selection beats {base['selected_beats_random_fraction']:.0%} of draws")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved {out_dir / 'summary.json'}")
    return summary


def main():
    ap = build_parser()
    ap.add_argument("input", help="segments file (.jsonl, .json or .csv)")
    opts = ap.parse_args()
    segments = ws.read_segments(opts.input)
    run_pipeline(segments, opts)


if __name__ == "__main__":
    main()