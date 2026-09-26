"""Macro F0.5 including singletons — the contest metric."""
from __future__ import annotations


def f05(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_f05(pred: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(pred), set(gold)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set and pred_set:
        return 0.0
    if gold_set and not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return f05(precision, recall)


def macro_f05(preds: dict[str, list[str]], gold: dict[str, list[str]]) -> float:
    scores = [entity_f05(preds.get(k, []), gold.get(k, [])) for k in gold]
    return sum(scores) / max(len(scores), 1)


def blocking_recall(cands: dict[str, list[str]], gold: dict[str, list[str]]) -> float:
    hit = total = 0
    for k, g in gold.items():
        for gid in g:
            total += 1
            if gid in set(cands.get(k, [])):
                hit += 1
    return hit / total if total else 1.0
