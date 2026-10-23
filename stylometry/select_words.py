#!/usr/bin/env python3
"""Program 3: choose the marker words used as stylometric dimensions.

Procedure
  1. Split each author's text into equal segments of --segment-size words.
  2. For each author, compute each word's relative frequency (percent of
     words) in every segment. A word is a candidate for that author only if
     it occurs in at least --min-presence of the author's segments.
  3. Consistency: rank candidates by coefficient of variation (CV = sd/mean)
     across that author's segments and keep the --pool most consistent.
     CV is used rather than raw sd because raw sd is mechanically smaller for
     rarer words; CV measures consistency relative to the word's own rate.
  4. Distinctiveness: for each consistent word, compare the author's mean
     rate to every other author's mean rate. The difference is standardized
     (Cohen's d: difference in means divided by pooled sd) so that frequent
     words such as "the" do not win merely by size. The raw difference in
     percentage points is reported alongside.
  5. Keep the --words-per-author highest-scoring words for each author. The
     union is the feature set for stylometry.py.

With --function-words, candidates are restricted to a fixed list of English
function words, so that subject matter (names, places, topics) cannot drive
the separation.

With --holdout, only even-numbered segments are used here, and stylometry.py
evaluates only the odd-numbered segments, so the words are not tested on the
same text that selected them.

Usage
  python select_words.py corpora/federalist --segment-size 1000
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import FUNCTION_WORDS, RESULTS_DIR, holdout_split, load_corpus, segment, tokenize, write_json


def segment_counts(text: str, size: int, part: str) -> list[Counter]:
    segs = segment(tokenize(text), size)
    return [Counter(segs[i]) for i in holdout_split(len(segs), part)]


def rate_matrix(counters: list[Counter], words: list[str], size: int) -> np.ndarray:
    """Rows = segments, columns = words, values = percent of segment."""
    m = np.array([[c.get(w, 0) for w in words] for c in counters], dtype=float)
    return 100.0 * m / size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--segment-size", type=int, default=1000)
    ap.add_argument("--words-per-author", type=int, default=15)
    ap.add_argument("--pool", type=int, default=150,
                    help="number of most consistent words kept per author before ranking by distinctiveness")
    ap.add_argument("--min-presence", type=float, default=0.8,
                    help="fraction of an author's segments a word must appear in")
    ap.add_argument("--score", choices=["mean", "min"], default="mean",
                    help="combine pairwise |d| across other authors by mean or by minimum")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--function-words", action="store_true",
                    help="restrict candidates to function words (removes topical vocabulary)")
    ap.add_argument("--out", type=Path, help="output directory (default results/<corpus>)")
    a = ap.parse_args()

    corpus = load_corpus(a.corpus_dir)
    authors = sorted(corpus)
    part = "select" if a.holdout else "all"
    counters = {au: segment_counts(corpus[au], a.segment_size, part) for au in authors}
    for au in authors:
        if len(counters[au]) < 3:
            raise SystemExit(f"{au}: only {len(counters[au])} selection segments; lower --segment-size")

    # Step 2-3: consistent words per author.
    consistent = {}
    for au in authors:
        n = len(counters[au])
        presence = Counter(w for c in counters[au] for w in c)
        cands = sorted(w for w, k in presence.items()
                       if k / n >= a.min_presence and (not a.function_words or w in FUNCTION_WORDS))
        r = rate_matrix(counters[au], cands, a.segment_size)
        cv = r.std(axis=0, ddof=1) / r.mean(axis=0)
        order = np.argsort(cv, kind="stable")[: a.pool]
        consistent[au] = {cands[i]: float(cv[i]) for i in order}

    # Statistics for every word that is consistent for anyone, for every author.
    vocab = sorted(set().union(*consistent.values()))
    stats = {}
    for au in authors:
        r = rate_matrix(counters[au], vocab, a.segment_size)
        stats[au] = {"mean": r.mean(axis=0), "sd": r.std(axis=0, ddof=1)}
    idx = {w: i for i, w in enumerate(vocab)}

    # Step 4-5: distinctiveness.
    selected, report = {}, {}
    for au in authors:
        rows = []
        others = [b for b in authors if b != au]
        for w, cv in consistent[au].items():
            i = idx[w]
            ma, sa = stats[au]["mean"][i], stats[au]["sd"][i]
            ds, diffs = [], {}
            for b in others:
                mb, sb = stats[b]["mean"][i], stats[b]["sd"][i]
                pooled = max(np.sqrt((sa ** 2 + sb ** 2) / 2), 1e-9)
                ds.append(abs(ma - mb) / pooled)
                diffs[b] = round(float(ma - mb), 4)
            score = float(np.mean(ds) if a.score == "mean" else np.min(ds))
            rows.append({"word": w, "score_d": round(score, 3), "cv": round(cv, 3),
                         "mean_pct": round(float(ma), 4), "diff_pct_vs": diffs})
        rows.sort(key=lambda r: (-r["score_d"], r["word"]))
        selected[au] = [r["word"] for r in rows[: a.words_per_author]]
        report[au] = rows[: a.words_per_author]

    features = sorted(set().union(*selected.values()))
    out_dir = a.out or RESULTS_DIR / Path(a.corpus_dir).name
    result = {
        "corpus_dir": str(Path(a.corpus_dir).resolve()),
        "settings": {"segment_size": a.segment_size, "words_per_author": a.words_per_author,
                     "pool": a.pool, "min_presence": a.min_presence, "score": a.score,
                     "holdout": a.holdout, "function_words": a.function_words},
        "segments_used": {au: len(counters[au]) for au in authors},
        "features": features,
        "selected_by_author": selected,
        "details": report,
        "author_means_pct": {au: {w: round(float(stats[au]["mean"][idx[w]]), 4) for w in features}
                             for au in authors},
    }
    write_json(out_dir / "selected_words.json", result)

    print(f"Selected {len(features)} feature words from {len(authors)} authors "
          f"(segment {a.segment_size}, {'holdout' if a.holdout else 'all segments'})")
    for au in authors:
        print(f"\n  {au} ({len(counters[au])} segments)")
        print(f"    {'word':<12}{'d':>7}{'CV':>7}{'mean %':>9}   diff vs others (pct pts)")
        for r in report[au]:
            diffs = "  ".join(f"{b}:{v:+.3f}" for b, v in r["diff_pct_vs"].items())
            print(f"    {r['word']:<12}{r['score_d']:>7.2f}{r['cv']:>7.2f}{r['mean_pct']:>9.3f}   {diffs}")
    print(f"\nWrote {out_dir / 'selected_words.json'}")


if __name__ == "__main__":
    main()
