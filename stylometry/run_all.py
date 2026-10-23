#!/usr/bin/env python3
"""Program 5: run the full pipeline once per corpus.

For each corpus: prepare_corpora -> check_corpus -> select_words -> stylometry.
A corpus that fails a step is reported and skipped; the others still run.

Usage
  python run_all.py
  python run_all.py --corpora bom federalist --function-words --holdout
  python run_all.py --bom-source BookOfMormonStylometry-main.zip
  python run_all.py --corpora bom --force      # run even if the checker fails
"""

import argparse
import subprocess
import sys

from common import CORPORA_DIR, RESULTS_DIR, ROOT, read_json, write_json

# Segment sizes: the Book of Mormon sole-author material is small
# (about 4,000-8,000 words per author), so it uses shorter segments.
SEGMENT_SIZE = {"bom": 500, "federalist": 1000, "kjv_nt": 1000, "bronte": 1000}


def run(cmd: list[str]) -> bool:
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=list(SEGMENT_SIZE), choices=list(SEGMENT_SIZE))
    ap.add_argument("--bom-source", help="Book of Mormon zip or directory")
    ap.add_argument("--skip-prepare", action="store_true", help="reuse existing corpora/ files")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "delta", "euclidean"])
    ap.add_argument("--words-per-author", type=int, default=15)
    ap.add_argument("--function-words", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="continue to word selection and stylometry even if check_corpus reports errors")
    a = ap.parse_args()

    py = sys.executable
    status = {}
    for corpus in a.corpora:
        print(f"\n{'=' * 70}\n{corpus}\n{'=' * 70}")
        size = str(SEGMENT_SIZE[corpus])
        cdir = str(CORPORA_DIR / corpus)
        sel = str(RESULTS_DIR / corpus / "selected_words.json")

        steps = []
        if not a.skip_prepare:
            prep = [py, "prepare_corpora.py", "--corpus", corpus]
            if a.bom_source and corpus == "bom":
                prep += ["--bom-source", a.bom_source]
            steps.append(("prepare", prep))
        steps.append(("check", [py, "check_corpus.py", cdir, "--segment-size", size]))
        select = [py, "select_words.py", cdir, "--segment-size", size,
                  "--words-per-author", str(a.words_per_author)]
        if a.function_words:
            select.append("--function-words")
        if a.holdout:
            select.append("--holdout")
        steps.append(("select", select))
        steps.append(("stylometry", [py, "stylometry.py", sel, "--metric", a.metric]))

        status[corpus] = "ok"
        for label, cmd in steps:
            if not run(cmd):
                if label == "check" and a.force:
                    print("  --force: continuing despite check errors")
                    status[corpus] = "ok (check forced)"
                    continue
                status[corpus] = f"failed at {label}"
                break

    print(f"\n{'=' * 70}\nOverview\n{'=' * 70}")
    print(f"  {'corpus':<12}{'status':<20}{'1-NN acc':>10}{'mean AUC':>10}")
    overview = {}
    for corpus, st in status.items():
        acc = auc = float("nan")
        if st.startswith("ok"):
            s = read_json(RESULTS_DIR / corpus / "summary.json")
            acc = s["overall_nn_accuracy"]
            aucs = [v["auc_within_vs_between"] for v in s["authors"].values()
                    if v["auc_within_vs_between"] is not None]
            auc = sum(aucs) / len(aucs)
            overview[corpus] = {"nn_accuracy": acc, "mean_auc": auc,
                                "per_author": {k: v["auc_within_vs_between"] for k, v in s["authors"].items()}}
        print(f"  {corpus:<12}{st:<20}{acc:>10.1%}{auc:>10.3f}")
    write_json(RESULTS_DIR / "overview.json", {"settings": vars(a), "status": status, "results": overview})
    print(f"\nPlots and tables are in {RESULTS_DIR}/<corpus>/")
    sys.exit(0 if all(v.startswith("ok") for v in status.values()) else 1)


if __name__ == "__main__":
    main()
