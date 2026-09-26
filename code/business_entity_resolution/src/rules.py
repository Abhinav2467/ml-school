"""High-precision deterministic rules.

These fire *before* the learned threshold. On F0.5, a few extra true
positives at zero false-merge cost are free; a sloppy rule is not.
Keep the predicates tight.
"""
from __future__ import annotations

from rapidfuzz.distance import JaroWinkler

from .normalize import NormRecord


def high_precision_match(a: NormRecord, b: NormRecord) -> bool:
    if a.country_key and b.country_key and a.country_key != b.country_key:
        return False
    core_a, core_b = a.core or a.name, b.core or b.name
    if not core_a or not core_b:
        return False
    jw = JaroWinkler.similarity(core_a, core_b)
    same_postal = bool(a.postal and a.postal == b.postal)
    same_sorted = bool(a.core_sorted and a.core_sorted == b.core_sorted)
    same_num = bool(a.numbers and a.numbers[:1] == b.numbers[:1])
    sort_jw = JaroWinkler.similarity(a.core_sorted or core_a, b.core_sorted or core_b)
    long_shared = any(len(t) >= 5 for t in (a.tokens_name & b.tokens_name))
    if same_sorted and len(core_a) >= 4:
        return True
    if same_postal and jw >= 0.88:
        return True
    if same_postal and same_num and (jw >= 0.80 or long_shared):
        return True
    if same_postal and long_shared:
        return True
    if sort_jw >= 0.93 and (same_postal or same_num):
        return True
    if jw >= 0.97 and (same_num or same_postal):
        return True
    return False
