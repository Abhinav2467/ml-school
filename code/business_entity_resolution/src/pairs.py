"""Build labeled training pairs from blocking + ground truth.

Split by Source 1 entity so the same business never leaks across folds.
"""
from __future__ import annotations

import random
from typing import Iterator

import numpy as np

from .config import POS_NEG_RATIO, RANDOM_SEED
from .features import pair_features
from .normalize import NormRecord


def iter_labeled_pairs(
    s1: dict[str, NormRecord],
    others: dict[str, NormRecord],
    candidates: dict[str, list[str]],
    gold: dict[str, list[str]],
    entity_ids: list[str],
    rng: random.Random | None = None,
) -> Iterator[tuple[list[float], int]]:
    rng = rng or random.Random(RANDOM_SEED)
    for sid in entity_ids:
        a = s1[sid]
        gold_ids = set(gold.get(sid, []))
        cand_ids = list(candidates.get(sid, []))
        # positives: all gold that we can look up
        pos = [oid for oid in gold_ids if oid in others]
        # negatives: blocked non-matches
        neg_block = [oid for oid in cand_ids if oid not in gold_ids and oid in others]
        rng.shuffle(neg_block)
        n_neg = min(len(neg_block), max(POS_NEG_RATIO * max(len(pos), 1), 4))
        chosen_neg = neg_block[:n_neg]
        for oid in pos:
            yield pair_features(a, others[oid]), 1
        for oid in chosen_neg:
            yield pair_features(a, others[oid]), 0


def matrix_from_pairs(pairs) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in pairs:
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.zeros((0, 1)), np.zeros((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)
