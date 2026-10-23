#!/usr/bin/env python3
"""
Generate a JSON file listing the claimed authors of every chapter of every
book in the Book of Mormon (239 chapters across 15 books).

Usage:
    python3 bom_authors_by_chapter.py [output_path]

Default output: book_of_mormon_authors_by_chapter.json

Scope and method
----------------
"Author" means a person or being to whom the text itself attributes substantial
words in a chapter: record keepers, abridgers, sermon speakers, letter writers,
divine speakers, and prophets whose writings are quoted. Attribution is at
chapter level; verse ranges are given in the optional "notes" field.

Inclusion rules
  1. Direct speech only. Speeches, prophecies, proclamations, and epistles that
     the text reports in narration or indirect speech (e.g. 1 Nephi 10;
     Alma 59:3; Ether 15:4-5) are noted but not listed.
  2. Brief incidental dialogue is omitted.
  3. Source records follow the text's own headings and statements (e.g. the
     Record of Zeniff before Mosiah 9; the record of Helaman before Alma 45).
     Where an attribution is an inference, the chapter note says so.
  4. A chapter consisting wholly of an abridger's commentary is not assigned a
     source record keeper (Helaman 12; 3 Nephi 29-30; Ether 5).
  5. An unnamed biblical parallel is listed as a quoted author only when the
     surrounding text identifies the source (e.g. 2 Nephi 25:1-8 frames
     chs. 25-30 as commentary on Isaiah). Otherwise the parallel is noted only
     (e.g. Micah in 3 Nephi 20-21; Micah is never named in the Book of Mormon).
  6. Distinct persons sharing a name are kept distinct (e.g. Ammon of
     Zarahemla in Mosiah 7-8 vs. Ammon, son of Mosiah, in Alma 17-27; the
     Nephi of 3 Nephi vs. his son, the Nephi of 4 Nephi).
  7. When a prophet or angel delivers the Lord's words in the first person of
     the Lord ("thus saith the Lord"), the Lord is listed as a divine speaker
     alongside the prophet or angel.
  8. An abridger is listed as an editorial commentator when his commentary
     runs to three or more consecutive verses; shorter "thus we see" asides
     are omitted.

Roles
-----
primary author            writes the record or text in his own voice
abridger                  Mormon or Moroni condensing an earlier record
source record keeper      author of an earlier record that an abridger used
speaker                   delivers a recorded discourse, prophecy, or dialogue
letter writer             author of an epistle or written proclamation whose
                          text is reproduced
divine speaker            the Lord, Jesus Christ, the Father, the Spirit
quoted author             text quoted at length from the brass plates or Bible
cited witness             prophet named as a witness or quoted only briefly
editorial commentator     abridger interrupting with his own commentary
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------
PA = "primary author"
AB = "abridger"
SR = "source record keeper"
SP = "speaker"
LW = "letter writer"
DV = "divine speaker"
QU = "quoted author"
CI = "cited witness"
CM = "editorial commentator"

ROLE_DEFINITIONS = OrderedDict([
    (PA, "writes the record or text in his own voice"),
    (AB, "Mormon or Moroni condensing an earlier record"),
    (SR, "author of an earlier record that an abridger used"),
    (SP, "delivers a recorded discourse, prophecy, or dialogue"),
    (LW, "author of an epistle or written proclamation whose text is reproduced"),
    (DV, "the Lord, Jesus Christ, the Father, the Spirit"),
    (QU, "text quoted at length from the brass plates or Bible"),
    (CI, "prophet named as a witness or quoted only briefly"),
    (CM, "abridger interrupting with his own commentary"),
])
VALID_ROLES = frozenset(ROLE_DEFINITIONS)

# Every chapter must have at least one person in one of these roles.
PRIMARY_ROLES = frozenset({PA, AB})

# Role pairs that cannot both apply to one person in one chapter.
CONFLICTING_ROLES = [
    frozenset({QU, CI}),
    frozenset({PA, AB}),
]

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
NEPHI = "Nephi, son of Lehi"
LEHI = "Lehi"
SARIAH = "Sariah"
JACOB = "Jacob, son of Lehi"
ENOS = "Enos"
JAROM = "Jarom"
OMNI = "Omni"
AMARON = "Amaron"
CHEMISH = "Chemish"
ABINADOM = "Abinadom"
AMALEKI = "Amaleki"
MORMON = "Mormon"
MORONI = "Moroni, son of Mormon"
CAPT_MORONI = "Captain Moroni"
BENJAMIN = "King Benjamin"
PEOPLE_BENJAMIN = "People of King Benjamin"
MOSIAH_II = "Mosiah II"
ZENIFF = "Zeniff"
LIMHI = "Limhi"
GIDEON = "Gideon"
AMMON_Z = "Ammon, descendant of Zarahemla"
AMMON_M = "Ammon, son of Mosiah"
AARON_M = "Aaron, son of Mosiah"
ABINADI = "Abinadi"
ALMA_E = "Alma the Elder"
ALMA_Y = "Alma the Younger"
AMULEK = "Amulek"
HELAMAN_A = "Helaman, son of Alma"
SHIBLON = "Shiblon"
HELAMAN_H = "Helaman, son of Helaman"
NEPHI_H = "Nephi, son of Helaman"
LEHI_H = "Lehi, son of Helaman"
NEPHI_N = "Nephi, son of Nephi, son of Helaman"
NEPHI_4 = "Nephi, son of Nephi the disciple"
SAMUEL = "Samuel the Lamanite"
AMINADAB = "Aminadab"
AMOS_1 = "Amos, son of Nephi"
AMOS_2 = "Amos, son of Amos"
AMMARON = "Ammaron"
ETHER = "Ether"
JARED = "Jared"
BROTHER_OF_JARED = "Brother of Jared"
DAUGHTER_OF_JARED = "Daughter of Jared, son of Omer"
LORD = "The Lord"
JESUS = "Jesus Christ"
FATHER = "God the Father"
SPIRIT = "Spirit of the Lord"
ANGEL = "Angel"
ISAIAH = "Isaiah"
MALACHI = "Malachi"
MOSES = "Moses"
ZENOS = "Zenos"
ZENOCK = "Zenock"
NEUM = "Neum"
JOSEPH_EGYPT = "Joseph of Egypt"
JACOB_ISRAEL = "Jacob (Israel), father of Joseph"
SHEREM = "Sherem"
ZEEZROM = "Zeezrom"
LAMONI = "Lamoni"
LAMANITE_KING_ALMA = "King of the Lamanites (father of Lamoni)"
LAMANITE_KING_MOSIAH = "King of the Lamanites (in the days of Limhi)"
ANTI_NEPHI_LEHI = "Anti-Nephi-Lehi"
KORIHOR = "Korihor"
ZORAMITES = "Zoramites"
ZERAHEMNAH = "Zerahemnah"
AMMORON = "Ammoron"
PAHORAN = "Pahoran"
GIDDIANHI = "Giddianhi"
LACHONEUS = "Lachoneus"
GIDGIDDONI = "Gidgiddoni"
EZIAS = "Ezias"
JEREMIAH = "Jeremiah"
ABRAHAM = "Abraham"


def S(start, end, *pairs):
    """Apply (name, role) pairs to every chapter from start to end inclusive."""
    return (start, end, pairs)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def build_chapters(book, count, spec, notes=None):
    """Merge specification entries into an ordered list of chapter records.

    Validates the chapter count, specification shape and ranges, notes,
    roles, role conflicts, primary-role presence, and final chapter coverage.
    """
    if not _is_int(count) or count < 1:
        raise ValueError(f"{book}: chapter count must be a positive integer")

    notes = {} if notes is None else notes
    if not isinstance(notes, dict):
        raise TypeError(f"{book}: notes must be a dict")

    for k, note in notes.items():
        if not _is_int(k) or not (1 <= k <= count):
            raise ValueError(f"{book}: note for chapter {k} outside 1..{count}")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{book}: note for chapter {k} must be non-empty text")

    table = {n: OrderedDict() for n in range(1, count + 1)}

    for entry in spec:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(f"{book}: malformed spec entry {entry!r}")
        start, end, pairs = entry

        if not (_is_int(start) and _is_int(end) and 1 <= start <= end <= count):
            raise ValueError(f"{book}: spec range {start}-{end} outside 1..{count}")

        if not isinstance(pairs, tuple) or not pairs:
            raise TypeError(f"{book}: spec range {start}-{end} has no valid pairs")

        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"{book}: malformed author/role pair {pair!r}")
            name, role = pair
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{book}: invalid author name {name!r}")
            if role not in VALID_ROLES:
                raise ValueError(f"{book}: invalid role {role!r}")

        for n in range(start, end + 1):
            for name, role in pairs:
                roles = table[n].setdefault(name, [])
                if role not in roles:
                    roles.append(role)

    chapters = []
    for n in range(1, count + 1):
        if not table[n]:
            raise ValueError(f"{book}: chapter {n} has no authors in spec")
        if not any(PRIMARY_ROLES & set(r) for r in table[n].values()):
            raise ValueError(
                f"{book} {n}: no {' or '.join(sorted(PRIMARY_ROLES))}"
            )
        for name, roles in table[n].items():
            role_set = set(roles)
            for conflict in CONFLICTING_ROLES:
                if conflict <= role_set:
                    raise ValueError(
                        f"{book} {n}: {name} has conflicting roles "
                        f"{sorted(conflict)}"
                    )
        rec = {
            "chapter": n,
            "authors": [{"name": k, "roles": v} for k, v in table[n].items()],
        }
        if n in notes:
            rec["notes"] = notes[n]
        chapters.append(rec)
    return chapters


# --------------------------------------------------------------------------
# Book data: (book, chapter_count, section, spec, notes)
# --------------------------------------------------------------------------
BOOKS = []


def add(book, count, section, spec, notes=None):
    BOOKS.append((book, count, section, spec, notes or {}))


# ---------------------------- SMALL PLATES --------------------------------
add("1 Nephi", 22, "Small Plates", [
    S(1, 22, (NEPHI, PA)),
    S(1, 1, (LEHI, SR), (LEHI, SP)),
    S(2, 2, (LEHI, SP), (LORD, DV)),
    S(3, 3, (LEHI, SP), (ANGEL, SP)),
    S(4, 4, (SPIRIT, DV)),
    S(5, 5, (LEHI, SP), (SARIAH, SP)),
    S(8, 8, (LEHI, SP)),
    S(11, 11, (SPIRIT, DV), (ANGEL, SP)),
    S(12, 14, (ANGEL, SP)),
    S(13, 13, (JESUS, DV)),
    S(16, 17, (LORD, DV)),
    S(19, 19, (ISAIAH, CI), (ZENOS, CI), (ZENOCK, CI), (NEUM, CI)),
    S(20, 21, (ISAIAH, QU)),
    S(22, 22, (MOSES, CI), (ISAIAH, CI)),
], {
    1: "Nephi abridges his father Lehi's record (1:16-17); Lehi's exclamation "
       "of praise (1:14).",
    2: "Lehi counsels Laman and Lemuel (2:9-10); the Lord speaks to Nephi "
       "(2:19-24).",
    3: "Lehi relays the Lord's command to return for the plates (3:2-6); an "
       "angel rebukes Laman and Lemuel (3:29).",
    5: "Sariah's complaint against Lehi (5:2) and her testimony (5:8); Lehi's "
       "reply (5:4-5).",
    8: "Lehi's dream of the tree of life, as told by Lehi and recorded by Nephi.",
    10: "Lehi's prophecies of the Messiah, his forerunner, and the scattering "
        "and gathering of Israel (10:2-14) are reported in indirect speech, so "
        "Lehi is not listed as a speaker.",
    13: "The angel relays the words of the Lamb of God in the first person "
        "('saith the Lamb of God,' 13:33-36).",
    16: "The voice of the Lord speaks to Lehi concerning the ball, or director "
        "(16:26). The name 'Liahona' is first used in Alma 37:38.",
    17: "The Lord commands Nephi to build a ship (17:7-14, 17:53).",
    18: "The Lord's direction to Lehi (18:5) is reported indirectly, so the "
        "Lord is not listed.",
    19: "Zenos, Zenock, and Neum are cited in 19:10-16; Nephi says he read "
        "Isaiah to his brothers (19:23), and the Isaiah text follows in "
        "chs. 20-21.",
    20: "Isaiah 48.",
    21: "Isaiah 49.",
    22: "Nephi interprets the Isaiah chapters he has just read from the brass "
        "plates (22:1), echoing Isaiah 49:22 (22:6, 22:8) and Isaiah 52:10 "
        "(22:10-11). Nephi quotes Moses's prophecy of a prophet like unto him "
        "(22:20-21).",
})

add("2 Nephi", 33, "Small Plates", [
    S(1, 33, (NEPHI, PA)),
    S(1, 4, (LEHI, SP)),
    S(3, 3, (JOSEPH_EGYPT, QU)),
    S(5, 5, (LORD, DV)),
    S(6, 10, (JACOB, SP)),
    S(6, 8, (ISAIAH, QU)),
    S(10, 10, (LORD, DV)),
    S(11, 11, (ISAIAH, CI), (JACOB, CI)),
    S(12, 24, (ISAIAH, QU)),
    S(26, 27, (ISAIAH, QU)),
    S(28, 29, (LORD, DV), (ISAIAH, CI)),
    S(30, 30, (ISAIAH, QU)),
    S(31, 31, (FATHER, DV), (JESUS, DV)),
], {
    3: "Lehi relays the prophecy of Joseph of Egypt from the brass plates "
       "(3:6-21).",
    4: "Lehi's blessings (4:3-11); Nephi's psalm (4:16-35).",
    5: "The Lord's words to Nephi (5:20, 5:25, 5:30).",
    6: "Jacob quotes Isaiah 49:22-23 (6:6-7) and Isaiah 49:24-26 (6:16-18).",
    7: "Jacob quotes Isaiah 50.",
    8: "Jacob quotes Isaiah 51 and 52:1-2.",
    9: "Jacob's invitation in 9:50-51 parallels Isaiah 55:1-2 but does not "
       "name Isaiah, and the framing of 6:4-5 covers only the reading in "
       "chs. 6-8, so Isaiah is not listed.",
    10: "Jacob delivers the words of the Lord God in the Lord's first person "
        "(10:7-19). An angel's words to Jacob (10:3) are reported indirectly, "
        "so the angel is not listed.",
    11: "Nephi introduces the Isaiah chapters; he names Isaiah and Jacob as "
        "fellow witnesses who have seen the Redeemer (11:2-3). No Isaiah text "
        "is quoted and Jacob's further words are not recorded (11:1).",
    12: "Isaiah 2.", 13: "Isaiah 3.", 14: "Isaiah 4.",
    15: "Isaiah 5.", 16: "Isaiah 6.", 17: "Isaiah 7.", 18: "Isaiah 8.",
    19: "Isaiah 9.", 20: "Isaiah 10.", 21: "Isaiah 11.", 22: "Isaiah 12.",
    23: "Isaiah 13.", 24: "Isaiah 14.",
    25: "Nephi's commentary on the words of Isaiah (25:1-8) frames chs. 25-30; "
        "no Isaiah text is quoted in this chapter, so Isaiah is not listed.",
    26: "Nephi reworks Isaiah 29:3-5 (26:15-18) without naming Isaiah; "
        "listed under the framing of 2 Nephi 25:1-8.",
    27: "Nephi's prophecy closely parallels and reworks Isaiah 29. Isaiah is not "
        "named in the chapter, but 2 Nephi 25:1-8 frames chs. 25-30 as "
        "commentary on Isaiah.",
    28: "Nephi quotes the Lord God (28:30, 28:32). Brief echoes of Isaiah "
        "29:15 (28:9), 29:21 (28:16), and 28:10 (28:30) are listed as "
        "citations under the framing of 2 Nephi 25:1-8.",
    29: "The Lord speaks throughout the chapter, echoing Isaiah 11:11 (29:1) "
        "and 5:26 (29:2); Isaiah is listed as cited under the framing of "
        "2 Nephi 25:1-8.",
    30: "Nephi quotes Isaiah 11:4-9 (30:9, 30:11-15) without naming Isaiah; "
        "listed under the framing of 2 Nephi 25:1-8.",
    31: "The voice of the Son (31:12, 14) and the voice of the Father "
        "(31:11, 15).",
})

add("Jacob", 7, "Small Plates", [
    S(1, 7, (JACOB, PA)),
    S(2, 2, (LORD, DV)),
    S(5, 5, (ZENOS, QU)),
    S(6, 6, (ZENOS, CI)),
    S(7, 7, (SHEREM, SP)),
], {
    2: "Jacob's temple sermon on pride and chastity; the word of the Lord comes "
       "to Jacob (2:11), and the Lord speaks in 2:23-33.",
    5: "Jacob introduces the allegory (5:1); the allegory of the olive tree, "
       "attributed to Zenos, follows (5:2-77).",
    6: "Jacob's commentary on the allegory; Zenos is named again (6:1).",
    7: "Jacob's encounter and dialogue with Sherem.",
})

add("Enos", 1, "Small Plates", [
    S(1, 1, (ENOS, PA), (LORD, DV)),
], {1: "The voice of the Lord answers Enos (1:5, 1:8, 1:10, 1:12, 1:18)."})

add("Jarom", 1, "Small Plates", [S(1, 1, (JAROM, PA))])

add("Omni", 1, "Small Plates", [
    S(1, 1, (OMNI, PA), (AMARON, PA), (CHEMISH, PA), (ABINADOM, PA),
      (AMALEKI, PA)),
], {1: "Omni 1:1-3 Omni; 1:4-8 Amaron; 1:9 Chemish; 1:10-11 Abinadom; "
       "1:12-30 Amaleki."})

# ------------------------------ PLATES OF MORMON ---------------------------
add("Words of Mormon", 1, "Plates of Mormon", [
    S(1, 1, (MORMON, PA)),
], {1: "Mormon's editorial bridge joining the small plates to his abridgment."})

# ---------------------- LARGE PLATES (ABRIDGED BY MORMON) ------------------
add("Mosiah", 29, "Large Plates (Mormon)", [
    S(1, 29, (MORMON, AB)),
    S(1, 5, (BENJAMIN, SP)),
    S(3, 3, (ANGEL, SP)),
    S(5, 5, (PEOPLE_BENJAMIN, SP)),
    S(7, 8, (AMMON_Z, SP), (LIMHI, SP)),
    S(9, 22, (ZENIFF, SR)),
    S(11, 17, (ABINADI, SP)),
    S(11, 12, (LORD, DV)),
    S(12, 18, (ALMA_E, SR)),
    S(12, 13, (MOSES, QU)),
    S(12, 12, (ISAIAH, QU)),
    S(14, 15, (ISAIAH, QU)),
    S(18, 18, (ALMA_E, SP)),
    S(20, 20, (LIMHI, SP), (LAMANITE_KING_MOSIAH, SP), (GIDEON, SP)),
    S(22, 22, (GIDEON, SP)),
    S(23, 24, (ALMA_E, SR)),
    S(23, 23, (ALMA_E, SP)),
    S(24, 24, (LORD, DV)),
    S(26, 26, (ALMA_E, SR), (LORD, DV)),
    S(27, 27, (ALMA_Y, SP), (ANGEL, SP), (LORD, DV)),
    S(28, 28, (LORD, DV)),
    S(29, 29, (MOSIAH_II, LW)),
], {
    2: "King Benjamin's address begins (chs. 2-5).",
    3: "Benjamin relays the words of an angel (3:2-27).",
    5: "The people answer Benjamin with one voice, declaring their faith and "
       "covenant (5:2-5).",
    7: "This Ammon is a descendant of Zarahemla (7:3), not the son of Mosiah. "
       "Limhi questions Ammon (7:9-11) and addresses his people (7:18-33).",
    8: "Ammon and Limhi discuss the Jaredite plates and seers (8:6-20).",
    9: "The Record of Zeniff (heading before ch. 9) spans chs. 9-22. Zeniff "
       "narrates in first person in chs. 9-10; the record continues in third "
       "person under Noah and Limhi and is described as the record kept by "
       "Limhi's people (8:5; 22:14).",
    11: "Abinadi's first prophecy, delivered as the Lord's words ('Thus saith "
        "the Lord,' 11:20-25).",
    12: "Abinadi's second prophecy, again in the Lord's words (12:1-8). Alma "
        "the Elder as source for Abinadi's words (chs. 12-17) is inferred from "
        "17:4. A priest quotes Isaiah 52:7-10 (12:20-24); Abinadi begins "
        "reciting the commandments given through Moses (12:34-36).",
    13: "Abinadi continues the Ten Commandments (13:12-24).",
    14: "Isaiah 53.",
    15: "Abinadi quotes Isaiah 52:8-10 (15:29-31).",
    17: "Alma the Elder writes all the words of Abinadi (17:4).",
    18: "Alma the Elder's teaching at the waters of Mormon. Alma as source is "
        "inferred from 17:4 and the heading before ch. 23.",
    19: "Narrative of Noah's death and Limhi's accession; no speech is "
        "recorded at length.",
    20: "Limhi questions the captured Lamanite king (20:14, 20:16); the king "
        "answers (20:15) and swears an oath of peace (20:24); Gideon counsels "
        "Limhi (20:17-22).",
    21: "Narrated account. Limhi's and Ammon's exchange (21:28) is reported "
        "indirectly, so neither is listed as a speaker.",
    22: "Gideon proposes the plan of escape (22:3-9); Mosiah receives the "
        "records kept by the people of Limhi (22:14).",
    23: "The heading before ch. 23 introduces an account of Alma and his "
        "people (chs. 23-24) but does not name a record; Alma the Elder as "
        "source is inferred. Alma declines kingship (23:7-14).",
    24: "The voice of the Lord comes to Alma's people (24:13-14, 24:16) and to "
        "Alma (24:23).",
    25: "Mosiah reads the records to the people, and Alma preaches and baptizes "
        "(25:5-18); both are reported in narration, so neither is listed as a "
        "speaker.",
    26: "The voice of the Lord answers Alma (26:15-32); Alma writes the words "
        "down (26:33). Alma the Elder as source for the chapter is inferred "
        "from 26:33.",
    27: "The angel rebukes Alma the Younger (27:13-16); Alma relays the Lord's "
        "words (27:25).",
    28: "The Lord answers Mosiah regarding his sons' mission (28:7).",
    29: "Mosiah's written proclamation to the people (29:5-32; see 29:4).",
})

add("Alma", 63, "Large Plates (Mormon)", [
    S(1, 63, (MORMON, AB)),
    S(1, 44, (ALMA_Y, SR)),
    S(45, 62, (HELAMAN_A, SR)),
    S(63, 63, (SHIBLON, SR)),
    S(1, 1, (ALMA_Y, SP)),
    S(3, 3, (LORD, DV)),
    S(5, 5, (ALMA_Y, SP), (SPIRIT, DV), (LORD, DV)),
    S(7, 9, (ALMA_Y, SP)),
    S(8, 8, (AMULEK, SP), (ANGEL, SP)),
    S(10, 11, (AMULEK, SP)),
    S(10, 10, (ANGEL, SP)),
    S(11, 11, (ZEEZROM, SP)),
    S(12, 13, (ALMA_Y, SP)),
    S(12, 12, (FATHER, DV)),
    S(14, 14, (ALMA_Y, SP), (AMULEK, SP)),
    S(15, 15, (ALMA_Y, SP)),
    S(17, 17, (AMMON_M, SP), (LORD, DV)),
    S(18, 20, (AMMON_M, SP), (LAMONI, SP)),
    S(20, 20, (LAMANITE_KING_ALMA, SP), (LORD, DV)),
    S(21, 22, (AARON_M, SP)),
    S(22, 22, (LAMANITE_KING_ALMA, SP)),
    S(24, 24, (ANTI_NEPHI_LEHI, SP)),
    S(26, 27, (AMMON_M, SP)),
    S(27, 27, (LORD, DV)),
    S(29, 33, (ALMA_Y, SP)),
    S(30, 30, (KORIHOR, SP)),
    S(31, 31, (ZORAMITES, SP)),
    S(33, 33, (ZENOS, QU), (ZENOCK, CI), (MOSES, CI)),
    S(34, 34, (AMULEK, SP), (ZENOS, CI), (ZENOCK, CI), (MOSES, CI)),
    S(36, 42, (ALMA_Y, SP)),
    S(36, 36, (ANGEL, SP), (LEHI, CI)),
    S(37, 37, (LORD, DV)),
    S(44, 44, (CAPT_MORONI, SP), (ZERAHEMNAH, SP)),
    S(45, 45, (ALMA_Y, SP), (HELAMAN_A, SP), (LORD, DV)),
    S(46, 46, (CAPT_MORONI, SP), (JACOB_ISRAEL, CI), (MORMON, CM)),
    S(48, 48, (MORMON, CM)),
    S(50, 50, (MORMON, CM)),
    S(52, 52, (CAPT_MORONI, SP)),
    S(54, 54, (CAPT_MORONI, LW), (AMMORON, LW)),
    S(55, 55, (CAPT_MORONI, SP)),
    S(56, 58, (HELAMAN_A, LW)),
    S(60, 60, (CAPT_MORONI, LW)),
    S(61, 61, (PAHORAN, LW)),
], {
    1: "Alma's record begins (chs. 1-44). Nehor's teaching is summarized in "
       "narration (1:3-6), so he is not listed; Alma pronounces judgment on "
       "him (1:13-14).",
    3: "The words of God to Nephi concerning the curse and mark are quoted in "
       "the Lord's first person (3:14-17).",
    5: "Alma's sermon in Zarahemla. Alma quotes the Lord's invitation in the "
       "Lord's first person (5:33-35) and the words of the Spirit (5:50-51).",
    7: "Alma's sermon in Gideon.",
    8: "An angel tells Alma to return to Ammonihah (8:15-17); Amulek receives "
       "him (8:19-20).",
    10: "Amulek's address; he recounts the angel's words to him (10:7-8).",
    12: "Alma quotes God calling on men in the name of his Son (12:33-35); "
        "identification of the speaker as the Father is inferred from "
        "'through mine Only Begotten Son' (12:33).",
    15: "Alma questions Zeezrom before healing him (15:6-10); Amulek's actions "
        "are narrated only.",
    17: "The heading before ch. 17 introduces an account of the sons of Mosiah "
        "(chs. 17-27) 'according to the record of Alma,' so Alma the Younger "
        "remains the source record keeper. The Lord comforts the sons of "
        "Mosiah (17:10-11); Ammon answers the Lamanite king (17:23-25).",
    20: "The voice of the Lord comes to Ammon (20:2); Lamoni's father speaks "
        "(20:9-27).",
    21: "Aaron contends with an Amalekite (21:5-9).",
    22: "Aaron teaches Lamoni's father.",
    23: "The king's proclamation is summarized in narration (23:1-3), so he is "
        "not listed.",
    24: "Speech of the Anti-Nephi-Lehi king (24:7-16).",
    26: "Ammon's rejoicing in the Lord.",
    27: "The Lord answers Ammon (27:11-12); Ammon reports it to the king (27:13).",
    29: "Alma's expression of desire.",
    30: "Korihor's speeches and his written confession (30:52-53).",
    31: "The Zoramites' Rameumptom prayer (31:15-18); Alma's prayer (31:26-35).",
    33: "Alma quotes Zenos at length (33:4-11) and Zenock briefly (33:16); "
        "Moses and the brazen serpent (33:19).",
    34: "Amulek's sermon continues Alma's; Zenos, Zenock, and Moses named as "
        "witnesses (34:7).",
    36: "Alma's counsel to Helaman; retells the angel's rebuke (36:8-9, 36:11) "
        "and echoes Lehi's words on seeing God upon his throne (36:22; cf. "
        "1 Nephi 1:8).",
    37: "Alma's counsel to Helaman; Alma charges Helaman to take the records "
        "(37:1). Alma quotes the Lord's words concerning Gazelem (37:23).",
    38: "Alma's counsel to Shiblon.",
    39: "Alma's counsel to Corianton (chs. 39-42).",
    44: "Moroni and Zerahemnah negotiate surrender.",
    45: "The heading before ch. 45 attributes chs. 45-62 to the record of "
        "Helaman. Alma blesses and prophesies to Helaman.",
    46: "Mormon's commentary on Amalickiah's dissension (46:8-10). Moroni "
        "raises the title of liberty and quotes the dying words of Jacob "
        "concerning Joseph's coat (46:24-26).",
    48: "Mormon's commentary on the character of Captain Moroni (48:11-18).",
    50: "Mormon's commentary on the fulfillment of the Lord's words to Lehi "
        "(50:19-23).",
    52: "Moroni offers terms to the besieged Lamanites (52:37).",
    54: "Epistles between Captain Moroni and Ammoron (Moroni 54:5-14; "
        "Ammoron 54:16-24).",
    55: "Moroni's response to Ammoron's epistle (55:1-2); the chapter then "
        "narrates the stratagem against the Lamanite guards.",
    56: "Helaman's epistle to Captain Moroni (chs. 56-58).",
    59: "Moroni's epistle to Pahoran (59:3) is summarized, not reproduced, so "
        "he is not listed as a letter writer.",
    60: "Captain Moroni's epistle to Pahoran.",
    61: "Pahoran's reply to Captain Moroni.",
    62: "62:52 reports Helaman's death, so the closing notice is not Helaman's "
        "own writing.",
    63: "Shiblon takes possession of the records (63:1) and confers them on "
        "Helaman, son of Helaman (63:11); 63:17 closes the account of Alma, "
        "Helaman, and Shiblon. Shiblon as source record keeper is inferred "
        "from his custody of the records; no heading names his record.",
})

add("Helaman", 16, "Large Plates (Mormon)", [
    S(1, 16, (MORMON, AB)),
    S(1, 3, (HELAMAN_H, SR)),
    S(4, 11, (NEPHI_H, SR)),
    S(13, 16, (NEPHI_H, SR)),
    S(3, 3, (MORMON, CM)),
    S(5, 5, (HELAMAN_H, SP), (NEPHI_H, SP), (LEHI_H, SP), (LORD, DV),
      (AMINADAB, SP), (FATHER, DV)),
    S(6, 6, (MORMON, CM)),
    S(7, 9, (NEPHI_H, SP)),
    S(7, 7, (LORD, DV)),
    S(10, 11, (NEPHI_H, SP)),
    S(8, 8, (MOSES, CI), (ABRAHAM, CI), (ZENOS, CI), (ZENOCK, CI), (EZIAS, CI),
      (ISAIAH, CI), (JEREMIAH, CI), (LEHI, CI), (NEPHI, CI)),
    S(10, 10, (LORD, DV)),
    S(12, 12, (MORMON, CM)),
    S(13, 15, (SAMUEL, SP)),
    S(13, 14, (ANGEL, SP)),
    S(13, 13, (LORD, DV)),
    S(15, 15, (LORD, DV), (ZENOS, CI)),
], {
    1: "The book's heading cites the record of Helaman, son of Helaman, and "
       "the records of his sons.",
    3: "Mormon's commentary on the mercy of the Lord (3:27-30). Helaman's "
       "death is reported at 3:37; his son Nephi succeeds him. Attribution of "
       "chs. 4-16 to Nephi's record is inferred from this succession and the "
       "book's heading.",
    5: "Helaman's counsel to his sons Nephi and Lehi (5:6-12), which they "
       "remember (5:5, 5:14); Nephi and Lehi speak from the fire (5:26); the "
       "voice above the cloud of darkness (5:29, 5:32); Aminadab speaks "
       "(5:41). The second voice (5:46-47) speaks of 'my Well Beloved'; "
       "its identification as the Father is inferred.",
    6: "Mormon's commentary on the Nephites' decline and the Lamanites' "
       "conversion (6:34-36).",
    7: "Nephi's prayer and address from his garden tower (7:13-29); he "
       "delivers the Lord's words in the first person (7:23).",
    8: "Nephi names prophets as witnesses of Christ (8:11-22), including Lehi "
       "and Nephi, son of Lehi (8:22).",
    10: "The Lord's promise and commission to Nephi (10:4-11); Nephi declares "
        "the Lord's warning to the people (10:14). His wider preaching is "
        "narrated (10:12, 10:17).",
    12: "Mormon's commentary on human frailty. The whole chapter is commentary, "
        "so no source record keeper is assigned.",
    13: "Samuel's prophecy on the wall of Zarahemla (chs. 13-15); Samuel "
        "delivers the Lord's words (13:8 ff.) and invokes the angel's message "
        "(13:7).",
    14: "Samuel quotes the angel's charge (14:9) and relays its words "
        "(14:26, 14:28).",
    15: "Samuel cites Zenos on the restoration of the Lamanites (15:11) and "
        "quotes the Lord (15:16-17).",
    16: "The people's response to Samuel is narrated; neither Samuel nor Nephi "
        "speaks directly in this chapter.",
})

add("3 Nephi", 30, "Large Plates (Mormon)", [
    S(1, 30, (MORMON, AB)),
    S(1, 28, (NEPHI_N, SR)),
    S(1, 1, (LORD, DV)),
    S(3, 3, (GIDDIANHI, LW), (LACHONEUS, SP), (GIDGIDDONI, SP)),
    S(5, 5, (MORMON, CM)),
    S(9, 10, (JESUS, DV)),
    S(10, 10, (ZENOS, CI), (ZENOCK, CI), (MORMON, CM)),
    S(11, 28, (JESUS, DV)),
    S(11, 11, (FATHER, DV)),
    S(16, 16, (ISAIAH, QU)),
    S(20, 20, (ISAIAH, QU), (MOSES, CI)),
    S(21, 21, (ISAIAH, CI)),
    S(22, 22, (ISAIAH, QU)),
    S(23, 23, (ISAIAH, CI), (SAMUEL, CI)),
    S(24, 25, (MALACHI, QU)),
    S(26, 26, (MORMON, CM)),
    S(28, 30, (MORMON, CM)),
    S(30, 30, (JESUS, DV)),
], {
    1: "Nephi, son of Helaman, has departed and left the records to his son "
       "Nephi (1:2-3). Nephi, son of Nephi, prays and hears the voice of the "
       "Lord announcing the sign of Christ's birth (1:13-14).",
    3: "Giddianhi's epistle (3:2-10); Lachoneus's proclamation is summarized "
       "(3:13-14), and his direct address follows (3:15); Gidgiddoni's counsel "
       "(3:21).",
    5: "Mormon states that he makes his record according to the record of "
       "Nephi (5:9-10) and speaks in first person (5:12 ff.). Attribution of "
       "chs. 1-28 to Nephi, son of Nephi, follows this statement.",
    7: "Nephi's ministry is reported (7:15-18) without recorded direct speech.",
    10: "Mormon's commentary on the preservation of the righteous and the "
        "fulfillment of Zenos and Zenock (10:11-19).",
    12: "Sermon at the temple begins (chs. 12-14, paralleling Matthew 5-7).",
    16: "Jesus quotes Isaiah 52:8-10 (16:18-20).",
    20: "Jesus cites Moses (20:23) and quotes Isaiah 52 (20:32-45). Prophecies "
        "paralleling Micah 5:8-9 (20:16-17) and Micah 4:12-13 (20:18-19) are "
        "noted only; Micah is not named.",
    21: "Jesus draws brief phrases from Isaiah 52 (21:8, 21:10, 21:29), "
        "identified by his reference to the words of Isaiah (20:11); these are "
        "short, so Isaiah is listed as cited rather than quoted. A prophecy "
        "paralleling Micah 5:8-15 (21:12-18, 21:21) is noted only; Micah is "
        "not named.",
    22: "Isaiah 54.",
    23: "Jesus commends the words of Isaiah (23:1-3) and asks why Samuel the "
        "Lamanite's fulfilled prophecy was not recorded (23:9-13).",
    24: "Malachi 3; Jesus reports the words the Father gave to Malachi (24:1).",
    25: "Malachi 4.",
    26: "Mormon explains that he writes only a part of what Jesus taught "
        "(26:6-12).",
    28: "Mormon's remarks on the Three Nephites (28:17-24, 28:36-40).",
    29: "Mormon's commentary; no source record keeper is assigned.",
    30: "Mormon relays the words of Jesus to the Gentiles.",
})

add("4 Nephi", 1, "Large Plates (Mormon)", [
    S(1, 1, (MORMON, AB), (NEPHI_4, SR), (AMOS_1, SR), (AMOS_2, SR),
      (AMMARON, SR)),
], {1: "1:1-19 Nephi; 1:19-21 Amos, son of Nephi; 1:21-47 Amos, son of Amos; "
       "1:47-49 Ammaron. The book's heading names its first record keeper as "
       "Nephi, son of Nephi, one of the disciples of Jesus Christ; he is "
       "therefore the son of the Nephi who kept the record of 3 Nephi."})

add("Mormon", 9, "Plates of Mormon", [
    S(1, 7, (MORMON, PA)),
    S(1, 1, (AMMARON, SP)),
    S(3, 3, (LORD, DV)),
    S(8, 9, (MORONI, PA)),
    S(9, 9, (JESUS, DV)),
], {
    1: "Ammaron instructs Mormon (1:2-4).",
    3: "The voice of the Lord commands Mormon to cry repentance (3:2) and "
       "declares 'Vengeance is mine' (3:15).",
    7: "Mormon's farewell address to the remnant.",
    8: "Moroni takes over the record at 8:1.",
    9: "Moroni's address to those who deny Christ; he quotes Christ's "
       "commission to his disciples (9:22-25).",
})

# ------------------------------ JAREDITE ---------------------------------
add("Ether", 15, "Jaredite Record (Moroni)", [
    S(1, 15, (MORONI, AB)),
    S(1, 4, (ETHER, SR)),
    S(6, 15, (ETHER, SR)),
    S(1, 1, (JARED, SP)),
    S(1, 2, (LORD, DV)),
    S(2, 3, (BROTHER_OF_JARED, SP)),
    S(2, 5, (MORONI, CM)),
    S(3, 4, (JESUS, DV)),
    S(8, 8, (DAUGHTER_OF_JARED, SP), (MORONI, CM)),
    S(12, 13, (MORONI, CM)),
    S(12, 12, (LORD, DV)),
], {
    1: "Ether's genealogy (1:6-32); Jared urges his brother to pray (1:34-39); "
       "the Lord answers the brother of Jared (1:40-43).",
    2: "The brother of Jared's plea concerning the barges (2:18-22); the Lord "
       "answers (2:20-25). Moroni comments on the decrees of God concerning "
       "the land (2:9-12).",
    3: "The brother of Jared's prayer (3:2-5) and vision of the Lord, who "
       "identifies himself as Jesus Christ (3:14). Moroni comments on the "
       "vision (3:17-20).",
    4: "Moroni comments (4:1-5) and relays the Lord's words on sealing the "
       "record (4:6-19); the speaker names himself as Jesus Christ (4:7-8).",
    5: "Moroni's address concerning the witnesses of the plates. The whole "
       "chapter is commentary, so no source record keeper is assigned.",
    8: "The daughter of Jared (son of Omer, not the Jared of Ether 1) proposes "
       "the plot with Akish (8:9-10). Moroni's warning against secret "
       "combinations (8:20-26).",
    12: "Moroni's discourse on faith; the Lord answers Moroni (12:26-28, 12:37).",
    13: "Moroni summarizes Ether's prophecy of the New Jerusalem (13:1-13) and "
        "states that he is forbidden to write more of it.",
    15: "The epistles exchanged by Coriantumr and Shiz (15:4-5, 15:18) are "
        "summarized, not reproduced, so neither is listed as a letter writer. "
        "Ether finishes his record (15:33-34).",
})

# ------------------------------- MORONI ----------------------------------
add("Moroni", 10, "Final Record (Moroni)", [
    S(1, 10, (MORONI, PA)),
    S(2, 2, (JESUS, DV)),
    S(7, 7, (MORMON, SP)),
    S(8, 9, (MORMON, LW)),
    S(8, 8, (JESUS, DV)),
], {
    2: "Moroni relays the words of Christ to the twelve disciples.",
    7: "Mormon's sermon on faith, hope, and charity (introduced at 7:1).",
    8: "Mormon's first epistle to Moroni, on infant baptism; Mormon records "
       "the words of Christ revealed to him (8:7-8).",
    9: "Mormon's second epistle to Moroni.",
    10: "Moroni's farewell and promise concerning the Holy Ghost. His closing "
        "exhortation (10:31) parallels Isaiah 52:1-2 and 54:2 but does not "
        "name Isaiah, so Isaiah is noted only.",
})


EXPECTED_BOOKS = [
    ("1 Nephi", 22),
    ("2 Nephi", 33),
    ("Jacob", 7),
    ("Enos", 1),
    ("Jarom", 1),
    ("Omni", 1),
    ("Words of Mormon", 1),
    ("Mosiah", 29),
    ("Alma", 63),
    ("Helaman", 16),
    ("3 Nephi", 30),
    ("4 Nephi", 1),
    ("Mormon", 9),
    ("Ether", 15),
    ("Moroni", 10),
]
EXPECTED_TOTAL_CHAPTERS = 239


def build_books():
    """Build and validate all book records."""
    books = []
    for book, count, section, spec, notes in BOOKS:
        books.append({
            "book": book,
            "section": section,
            "chapter_count": count,
            "chapters": build_chapters(book, count, spec, notes),
        })

    actual = [(b["book"], b["chapter_count"]) for b in books]
    if actual != EXPECTED_BOOKS:
        raise ValueError(
            f"Book/chapter structure mismatch: expected {EXPECTED_BOOKS}, "
            f"built {actual}"
        )

    total = sum(b["chapter_count"] for b in books)
    if total != EXPECTED_TOTAL_CHAPTERS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_CHAPTERS} chapters, built {total}"
        )

    # Validate the final records, not merely the input specification.
    for b in books:
        numbers = [c["chapter"] for c in b["chapters"]]
        if numbers != list(range(1, b["chapter_count"] + 1)):
            raise ValueError(f'{b["book"]}: chapter sequence mismatch: {numbers}')
        for c in b["chapters"]:
            names = [a["name"] for a in c["authors"]]
            if not names:
                raise ValueError(f'{b["book"]} {c["chapter"]} has no authors')
            if len(names) != len(set(names)):
                raise ValueError(
                    f'{b["book"]} {c["chapter"]} has duplicate author records'
                )
            for a in c["authors"]:
                if not a["name"] or not a["roles"]:
                    raise ValueError(
                        f'{b["book"]} {c["chapter"]} has malformed author '
                        f"record: {a!r}"
                    )
    return books, total


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "book_of_mormon_authors_by_chapter.json")

    books, total = build_books()

    data = {
        "metadata": {
            "title": "Claimed authors by chapter, Book of Mormon",
            "total_books": len(books),
            "total_chapters": total,
            "method": (
                "Chapter-level attribution based on the text's own claims. "
                "Named speakers, record keepers, abridgers, letter writers, "
                "divine speakers, and substantial quoted or cited sources are "
                "included. Speech and epistles that are only summarized in "
                "narration or indirect speech are omitted. Abridger commentary "
                "is tagged when it runs to three or more consecutive verses. "
                "Inferred attributions are flagged in chapter notes."
            ),
            "roles": dict(ROLE_DEFINITIONS),
        },
        "books": books,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(books)} books, {total} chapters)")


if __name__ == "__main__":
    main()