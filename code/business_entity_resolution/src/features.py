"""Pairwise similarity features. Keep them country-agnostic."""
from __future__ import annotations

from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import NormRecord


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)


def _jw(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


FEATURE_NAMES = [
    "name_lev",
    "name_jw",
    "core_lev",
    "core_jw",
    "sorted_eq",
    "sorted_jw",
    "name_jaccard",
    "addr_jaccard",
    "addr_lev",
    "postal_eq",
    "postal_both_missing",
    "num_overlap",
    "first_num_eq",
    "country_eq",
    "soundex_eq",
    "prefix_eq",
    "len_name_ratio",
    "token_count_diff",
    "shared_token_count",
    "token_containment",
    "shared_long_token",
]


def pair_features(a: NormRecord, b: NormRecord) -> list[float]:
    name_j = _jaccard(a.tokens_name, b.tokens_name)
    addr_j = _jaccard(a.tokens_addr, b.tokens_addr)
    nums_a, nums_b = set(a.numbers), set(b.numbers)
    nlen = max(len(a.name), len(b.name), 1)
    inter = a.tokens_name & b.tokens_name
    denom = min(len(a.tokens_name), len(b.tokens_name)) or 1
    return [
        _ratio(a.name, b.name),
        _jw(a.name, b.name),
        _ratio(a.core, b.core),
        _jw(a.core, b.core),
        float(a.core_sorted == b.core_sorted and bool(a.core_sorted)),
        _jw(a.core_sorted, b.core_sorted),
        name_j,
        addr_j,
        _ratio(a.addr, b.addr),
        float(bool(a.postal) and a.postal == b.postal),
        float(a.postal is None and b.postal is None),
        float(len(nums_a & nums_b)),
        float(bool(a.numbers) and a.numbers[:1] == b.numbers[:1]),
        float(a.country_key == b.country_key and bool(a.country_key)),
        float(a.soundex == b.soundex and bool(a.soundex)),
        float(a.name_prefix == b.name_prefix and len(a.name_prefix) >= 4),
        min(len(a.name), len(b.name)) / nlen,
        float(abs(len(a.tokens_name) - len(b.tokens_name))),
        float(len(inter)),
        float(len(inter) / denom),
        float(any(len(t) >= 5 for t in inter)),
    ]
