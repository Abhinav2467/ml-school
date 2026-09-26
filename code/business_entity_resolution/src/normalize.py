"""Country-agnostic name/address normalization.

Do not freeze country to {US, India}. Suffix and postal regexes are
extensible tables, not one-hot vocabularies.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from unidecode import unidecode

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "plc",
    "pvt", "private", "pvt ltd", "private limited",
    "llc", "l.l.c", "gmbh", "ag", "kg",
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc",
    "bv", "nv", "oy", "ab", "pty", "pte",
    "llc", "pc", "pllc",
}

NAME_STOP = {
    "the", "and", "of", "for", "a", "an", "&",
    "dba", "trading", "as",
}

ADDR_ABBR = {
    "rd": "road", "st": "street", "ste": "suite", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "cir": "circle", "apt": "apartment", "fl": "floor",
    "bldg": "building", "no": "number", "nr": "near", "opp": "opposite",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "rte": "route", "rue": "rue", "bd": "boulevard", "av.": "avenue",
}

# Open-set postal extractors. Add patterns; do not branch on country labels.
POSTAL_PATTERNS = [
    re.compile(r"\b\d{5}(?:-\d{4})?\b"),          # US ZIP / ZIP+4 / FR 5-digit
    re.compile(r"\b\d{6}\b"),                      # IN PIN
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I),  # UK-like
]

LANDMARK_RE = re.compile(
    r"\b(near|opp|opposite|behind|next to|beside|landmark)\b.*$",
    re.I,
)
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
WS = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")


def _fold(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = unidecode(text)
    text = text.lower().replace("&", " and ")
    text = NON_ALNUM.sub(" ", text)
    return WS.sub(" ", text).strip()


def strip_suffixes(name: str) -> str:
    tokens = name.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    # two-token suffixes already covered via last-token loop after fold
    # also drop trailing "pvt ltd" after fold ("pvt", "ltd")
    while len(tokens) >= 2 and " ".join(tokens[-2:]) in LEGAL_SUFFIXES:
        tokens = tokens[:-2]
    return " ".join(tokens)


def core_name(name: str) -> str:
    tokens = [t for t in strip_suffixes(name).split() if t not in NAME_STOP]
    return " ".join(tokens)


def expand_address(addr: str) -> str:
    addr = LANDMARK_RE.sub(" ", addr)
    tokens = []
    for t in addr.split():
        tokens.append(ADDR_ABBR.get(t, t))
    return " ".join(tokens)


def extract_postal(addr: str) -> Optional[str]:
    raw = str(addr or "")
    for pat in POSTAL_PATTERNS:
        m = pat.search(raw)
        if m:
            return re.sub(r"\s+", "", m.group(0)).upper()
    return None


def extract_numbers(text: str) -> list[str]:
    return DIGITS.findall(text or "")


def soundex(s: str) -> str:
    """Tiny Soundex so blocking does not need extra deps."""
    if not s:
        return ""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    first = s[0]
    mapping = {
        **dict.fromkeys(list("bfpv"), "1"),
        **dict.fromkeys(list("cgjkqsxz"), "2"),
        **dict.fromkeys(list("dt"), "3"),
        **dict.fromkeys(list("l"), "4"),
        **dict.fromkeys(list("mn"), "5"),
        **dict.fromkeys(list("r"), "6"),
    }
    digits = [mapping.get(ch, "") for ch in s[1:]]
    out = [first]
    prev = mapping.get(first, "")
    for d in digits:
        if d and d != prev:
            out.append(d)
        if d:
            prev = d
        else:
            prev = ""
        if len(out) == 4:
            break
    return ("".join(out) + "000")[:4]


@dataclass
class NormRecord:
    entity_id: str
    country: str
    country_key: str
    name_raw: str
    addr_raw: str
    name: str
    core: str
    addr: str
    postal: Optional[str]
    numbers: list[str]
    name_prefix: str
    core_sorted: str
    soundex: str
    tokens_name: frozenset
    tokens_addr: frozenset


def normalize_record(row: dict) -> NormRecord:
    name_raw = row.get("business_name") or ""
    addr_raw = row.get("business_address") or ""
    country = str(row.get("country") or "").strip()
    name = _fold(name_raw)
    addr = expand_address(_fold(addr_raw))
    core = core_name(name)
    postal = extract_postal(addr_raw) or extract_postal(addr)
    nums = extract_numbers(addr)
    prefix = re.sub(r"[^a-z0-9]", "", core)[:8]
    core_sorted = " ".join(sorted(core.split()))
    sx = soundex(core.split()[0] if core else name)
    return NormRecord(
        entity_id=str(row["entity_id"]),
        country=country,
        country_key=_fold(country),
        name_raw=str(name_raw),
        addr_raw=str(addr_raw),
        name=name,
        core=core,
        addr=addr,
        postal=postal,
        numbers=nums,
        name_prefix=prefix,
        core_sorted=core_sorted,
        soundex=sx,
        tokens_name=frozenset(core.split()),
        tokens_addr=frozenset(t for t in addr.split() if len(t) > 1),
    )
