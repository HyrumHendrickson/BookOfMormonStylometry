"""Shared utilities for the stylometry pipeline.

Every program imports this module so that text normalization, tokenization,
and segmentation are identical at every stage.

Standardized corpus format (produced by prepare_corpora.py, verified by
check_corpus.py):
    corpora/<corpus>/<author_slug>.txt   one file per author
    corpora/<corpus>/manifest.json       provenance (never read as text)

Each author file is plain ASCII, contains only that author's prose, and has
one source unit (chapter, paper, or paragraph) per line. No headings, verse
numbers, page numbers, footnotes, or editorial matter.
"""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPORA_DIR = ROOT / "corpora"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

AUTHOR_FILE_RE = re.compile(r"^[a-z0-9_]+\.txt$")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

_CHAR_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2014": " -- ", "\u2013": "-", "\u2012": "-", "\u2010": "-",
    "\u2026": "...", "\u00a0": " ",
    "\u00c6": "Ae", "\u00e6": "ae", "\u0152": "Oe", "\u0153": "oe",
    "\u00df": "ss",
}


def to_ascii(text: str) -> str:
    """Map typographic characters to ASCII and strip diacritics."""
    for k, v in _CHAR_MAP.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.encode("ascii", "ignore").decode("ascii")


def clean_unit(text: str) -> str:
    """Normalize one source unit to a single clean ASCII line."""
    text = to_ascii(text)
    text = text.replace("_", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Internal apostrophes are kept (e.g. "lehi's")."""
    return WORD_RE.findall(text.lower())


def list_author_files(corpus_dir: Path) -> list[Path]:
    return sorted(p for p in Path(corpus_dir).glob("*.txt"))


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Return {author_slug: full_text} for every author file in a corpus."""
    return {p.stem: p.read_text(encoding="utf-8") for p in list_author_files(corpus_dir)}


def segment(tokens: list[str], size: int) -> list[list[str]]:
    """Split tokens into consecutive, non-overlapping segments of exactly
    `size` tokens. The trailing remainder is discarded so that every segment
    has identical length and relative frequencies are directly comparable."""
    n = len(tokens) // size
    return [tokens[i * size:(i + 1) * size] for i in range(n)]


def holdout_split(n_segments: int, part: str) -> list[int]:
    """Deterministic interleaved split. 'select' -> even indices,
    'evaluate' -> odd indices, 'all' -> every index."""
    if part == "all":
        return list(range(n_segments))
    if part == "select":
        return list(range(0, n_segments, 2))
    if part == "evaluate":
        return list(range(1, n_segments, 2))
    raise ValueError(part)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Fixed, colour-blind-tolerant palette (Okabe-Ito + extras). Authors are
# assigned colours in sorted order so each author keeps one colour everywhere.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000", "#999999", "#882255"]


def author_colors(authors: list[str]) -> dict[str, str]:
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(sorted(authors))}


# English function words, including Early Modern forms common in the KJV and
# Book of Mormon. Used by select_words.py --function-words to exclude topical
# vocabulary (names, places, subject matter) from the feature set.
FUNCTION_WORDS = frozenset("""
a about above according across after again against all almost along already also although
always am among amongst an and another any anything are around as at be became because
become been before behold being below beside besides between beyond both but by can cannot
could did do does doing done down during each either else enough even ever every except
far few for forth from further had hast hath has have having he hence her here hers herself
him himself his how however i if in indeed into is it its itself just least less lest let
like may me might mine more moreover most much must my myself neither never nevertheless no
nor not nothing now o of off often on once one only or other others otherwise ought our
ours ourselves out over own per perhaps quite rather same shall shalt should since so some
something still such than that the thee their theirs them themselves then thence there
thereby therefore therein thereof these they thine this thither those thou though through
throughout thus thy thyself till to together too toward towards under unless unto until up
upon us very was we were what whatever when whence whenever where whereas whereby wherefore
wherein whether which while whilst whither who whoever whole whom whose why will wilt with
within without would ye yea yet you your yours yourself yourselves art doth didst wast wert
canst couldst wouldst shouldst mayest mightest dost
""".split())
