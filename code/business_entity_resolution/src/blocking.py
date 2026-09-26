"""Multi-key blocking. Union of cheap keys + TF-IDF top-k.

candidate_pairs.tsv must be THIS set — the last list the model scores.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MAX_CANDIDATES_PER_S1, SAME_COUNTRY_ONLY_IN_BLOCK, TFIDF_TOPK
from .normalize import NormRecord


def _keys(rec: NormRecord) -> list[str]:
    keys = []
    ck = rec.country_key or "unk"
    if rec.postal:
        keys.append(f"post|{ck}|{rec.postal}")
    if rec.core_sorted:
        keys.append(f"sorted|{ck}|{rec.core_sorted}")
    if rec.name_prefix and len(rec.name_prefix) >= 4:
        keys.append(f"pref|{ck}|{rec.name_prefix[:6]}")
    if rec.soundex:
        keys.append(f"sx|{ck}|{rec.soundex}")
    # first two significant name tokens
    toks = list(rec.tokens_name)
    if len(toks) >= 2:
        keys.append(f"bigram|{ck}|{' '.join(sorted(toks[:2]))}")
    elif toks:
        keys.append(f"tok|{ck}|{toks[0]}")
    if rec.numbers:
        keys.append(f"num|{ck}|{rec.numbers[0]}")
    return keys


def inverted_index(records: Iterable[NormRecord]) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        for k in _keys(rec):
            inv[k].append(rec.entity_id)
    return inv


def _tfidf_candidates(
    s1: list[NormRecord],
    others: list[NormRecord],
    topk: int,
) -> dict[str, set[str]]:
    if not s1 or not others or topk <= 0:
        return {}
    corpus_s1 = [f"{r.core} {r.addr}" for r in s1]
    corpus_o = [f"{r.core} {r.addr}" for r in others]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    # Fit on both so unseen test tokens still work
    vec.fit(corpus_s1 + corpus_o)
    A = vec.transform(corpus_s1)
    B = vec.transform(corpus_o)
    out: dict[str, set[str]] = defaultdict(set)
    # chunk to avoid dense N×M if large
    chunk = 512
    oids = [r.entity_id for r in others]
    o_country = [r.country_key for r in others]
    for i0 in range(0, len(s1), chunk):
        sim = cosine_similarity(A[i0:i0 + chunk], B)
        for i, row in enumerate(sim):
            rec = s1[i0 + i]
            if topk >= len(row):
                idxs = np.argsort(-row)
            else:
                idxs = np.argpartition(-row, topk)[:topk]
                idxs = idxs[np.argsort(-row[idxs])]
            for j in idxs[:topk]:
                if SAME_COUNTRY_ONLY_IN_BLOCK and rec.country_key and o_country[j]:
                    if rec.country_key != o_country[j]:
                        continue
                if row[j] <= 0:
                    continue
                out[rec.entity_id].add(oids[j])
    return out


def generate_candidates(
    s1_recs: list[NormRecord],
    s2_recs: list[NormRecord],
    s3_recs: list[NormRecord],
    max_per: int = MAX_CANDIDATES_PER_S1,
) -> dict[str, list[str]]:
    others = s2_recs + s3_recs
    by_id = {r.entity_id: r for r in others}
    inv = inverted_index(others)

    cands: dict[str, set[str]] = defaultdict(set)
    for rec in s1_recs:
        seen = set()
        for k in _keys(rec):
            for oid in inv.get(k, ()):
                o = by_id[oid]
                if SAME_COUNTRY_ONLY_IN_BLOCK and rec.country_key and o.country_key:
                    if rec.country_key != o.country_key:
                        continue
                seen.add(oid)
        cands[rec.entity_id] |= seen

    tfidf = _tfidf_candidates(s1_recs, others, TFIDF_TOPK)
    for sid, ids in tfidf.items():
        cands[sid] |= ids

    # cap: keep all key-block hits first, then fill with tfidf order already mixed
    out: dict[str, list[str]] = {}
    for rec in s1_recs:
        ids = list(cands.get(rec.entity_id, set()))
        if len(ids) > max_per:
            # prefer postal / exact-sorted hits
            def score(oid: str) -> tuple:
                o = by_id[oid]
                return (
                    int(bool(rec.postal and rec.postal == o.postal)),
                    int(rec.core_sorted == o.core_sorted and rec.core_sorted != ""),
                    int(rec.soundex == o.soundex),
                    len(rec.tokens_name & o.tokens_name),
                )
            ids = sorted(ids, key=score, reverse=True)[:max_per]
        out[rec.entity_id] = ids
    return out
