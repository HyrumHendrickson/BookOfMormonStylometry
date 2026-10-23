# Stylometry pipeline

Requires Python 3.10+ and `pip install -r requirements.txt`.
Place `BookOfMormonStylometry-main.zip` (or the extracted folder) in this directory, or pass `--bom-source`.

    python run_all.py                               # all four corpora
    python run_all.py --function-words --holdout    # stricter: no topical words; select and test on disjoint segments

| Program | Purpose | Output |
|---|---|---|
| `prepare_corpora.py` | Download/parse texts, one ASCII file per author | `corpora/<corpus>/<author>.txt`, `manifest.json` |
| `check_corpus.py` | Format, contamination, size and segment-count checks (exit 1 on error) | console |
| `select_words.py` | Consistent (low CV) words per author, ranked by standardized difference (Cohen's d) from other authors | `results/<corpus>/selected_words.json` |
| `stylometry.py` | z-scored word-rate vectors, pairwise distances, histograms, AUC and 1-NN accuracy | `hist_*.png`, `summary.json`, `distance_matrix.csv` |
| `run_all.py` | Runs the four steps for `bom`, `federalist`, `kjv_nt`, `bronte` | `results/overview.json` |
| `common.py` | Shared normalization, tokenization, segmentation | |

Corpora
- bom: Mormon, Nephi (son of Lehi), Moroni; only chapters annotated with a single author. 500-word segments.
- federalist: Hamilton, Madison, Jay; disputed (49-58, 62, 63) and joint (18-20) papers excluded.
- kjv_nt: Paul (7 undisputed epistles), Luke (Luke, Acts), John (John, 1-3 John).
- bronte: Charlotte (4 novels), Emily (Wuthering Heights), Anne (2 novels).

Downloads are cached in `cache/`. If a source is unreachable, place the file at the path named in the error message.
