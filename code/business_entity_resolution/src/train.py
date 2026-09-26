#!/usr/bin/env python3
"""Train matcher on a Source-1-entity holdout and pick an F0.5 threshold."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocking import generate_candidates
from src.config import RANDOM_SEED, TRAIN_DIR, VAL_FRACTION
from src.io_utils import load_ground_truth, load_split
from src.metrics import blocking_recall, macro_f05
from src.model import save_bundle, sweep_threshold, train_lgbm, predict_proba
from src.normalize import normalize_record
from src.pairs import iter_labeled_pairs, matrix_from_pairs


def _index(df):
    return {r["entity_id"]: normalize_record(r) for r in df.to_dict(orient="records")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    args = p.parse_args()

    sources = load_split(args.train_dir, "train")
    gt = load_ground_truth(args.train_dir / "train_ground_truth.tsv")
    gold = dict(zip(gt["source1_entity_id"], gt["matched_list"]))

    s1 = _index(sources["s1"])
    others = {**_index(sources["s2"]), **_index(sources["s3"])}

    s1_ids = list(s1.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(s1_ids)
    cut = max(1, int(len(s1_ids) * (1 - VAL_FRACTION)))
    train_ids, val_ids = s1_ids[:cut], s1_ids[cut:]

    s1_train = [s1[i] for i in train_ids]
    s1_val = [s1[i] for i in val_ids]
    other_list = list(others.values())

    print(f"S1 train/val entities: {len(train_ids)}/{len(val_ids)}")
    print("Blocking train fold...")
    c_train = generate_candidates(s1_train, list(_index(sources["s2"]).values()),
                                  list(_index(sources["s3"]).values()))
    print("Blocking val fold...")
    c_val = generate_candidates(s1_val, list(_index(sources["s2"]).values()),
                                list(_index(sources["s3"]).values()))

    gold_train = {k: gold.get(k, []) for k in train_ids}
    gold_val = {k: gold.get(k, []) for k in val_ids}
    print(f"Blocking recall train={blocking_recall(c_train, gold_train):.3f} "
          f"val={blocking_recall(c_val, gold_val):.3f}")

    Xtr, ytr = matrix_from_pairs(iter_labeled_pairs(s1, others, c_train, gold, train_ids, rng))
    Xva, yva = matrix_from_pairs(iter_labeled_pairs(s1, others, c_val, gold, val_ids, rng))
    print(f"Pairs train={len(ytr)} pos={int(ytr.sum())}  val={len(yva)} pos={int(yva.sum())}")

    model = train_lgbm(Xtr, ytr, Xva, yva)

    scores = {}
    for sid in val_ids:
        feats, oids = [], []
        for oid in c_val.get(sid, []):
            if oid not in others:
                continue
            feats.append(__import__("src.features", fromlist=["pair_features"]).pair_features(s1[sid], others[oid]))
            oids.append(oid)
        if not oids:
            scores[sid] = []
            continue
        import numpy as np
        proba = predict_proba(model, np.asarray(feats, dtype=np.float32))
        scores[sid] = list(zip(oids, proba.tolist()))

    if len(val_ids) < 8:
        from src.config import DEFAULT_THRESHOLD
        t, s = DEFAULT_THRESHOLD, float("nan")
        print(f"Val fold too small ({len(val_ids)} S1) — keeping default threshold {t:.3f}")
    else:
        t, s = sweep_threshold(scores, gold_val)
    preds = {sid: [oid for oid, p in pairs if p >= t] for sid, pairs in scores.items()}
    for k in gold_val:
        preds.setdefault(k, [])
    print(f"Chosen threshold={t:.3f}  val macro-F0.5={s}  "
          f"recomputed={macro_f05(preds, gold_val):.4f}")
    path = save_bundle(model, t)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
