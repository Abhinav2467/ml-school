#!/usr/bin/env python3
"""Train matcher on a Source-1-entity holdout and pick an F0.5 threshold.
Optimized for Apple Silicon / low memory footprint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocking import _keys
from src.config import (
    DEFAULT_THRESHOLD,
    MAX_CANDIDATES_PER_S1,
    RANDOM_SEED,
    SAME_COUNTRY_ONLY_IN_BLOCK,
    TRAIN_DIR,
    VAL_FRACTION,
)
from src.features import pair_features
from src.io_utils import load_ground_truth
from src.metrics import blocking_recall, macro_f05
from src.model import predict_proba, save_bundle, sweep_threshold, train_lgbm
from src.normalize import NormRecord, normalize_record
from src.pairs import iter_labeled_pairs, matrix_from_pairs


def stream_block(
    s1_list: list[NormRecord],
    others: dict[str, NormRecord],
    inv: dict[str, list[str]],
    max_per: int = MAX_CANDIDATES_PER_S1,
) -> dict[str, list[str]]:
    """Fast hash-based candidate generation."""
    cands = {}
    for rec in s1_list:
        seen = set()
        for k in _keys(rec):
            for oid in inv.get(k, ()):
                o = others[oid]
                if SAME_COUNTRY_ONLY_IN_BLOCK and rec.country_key and o.country_key:
                    if rec.country_key != o.country_key:
                        continue
                seen.add(oid)

        cand_ids = list(seen)
        if len(cand_ids) > max_per:
            def score(oid: str) -> tuple:
                o = others[oid]
                return (
                    int(bool(rec.postal and rec.postal == o.postal)),
                    int(rec.core_sorted == o.core_sorted and rec.core_sorted != ""),
                    int(rec.soundex == o.soundex),
                    len(rec.tokens_name & o.tokens_name),
                )
            cand_ids = sorted(cand_ids, key=score, reverse=True)[:max_per]
        cands[rec.entity_id] = cand_ids
    return cands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--sample-s1", type=int, default=60000,
                   help="Number of S1 entities to train on (default 60,000 for fast, high-accuracy training)")
    args = p.parse_args()

    print(f"=== Starting Optimized Training Pipeline ===", flush=True)
    t0 = time.time()

    # 1. Load Ground Truth
    gt_path = args.train_dir / "train_ground_truth.tsv"
    print(f"Loading ground truth: {gt_path.name}...", flush=True)
    gt = load_ground_truth(gt_path)
    gold = dict(zip(gt["source1_entity_id"], gt["matched_list"]))
    all_s1_keys = list(gold.keys())
    print(f"Total ground truth records: {len(gold):,}", flush=True)

    # 2. Select S1 entities (with sampling if large)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_s1_keys)
    if args.sample_s1 and args.sample_s1 < len(all_s1_keys):
        s1_keys_to_use = set(all_s1_keys[:args.sample_s1])
        print(f"Using sampled subset of {len(s1_keys_to_use):,} S1 entities for rapid M4 training...", flush=True)
    else:
        s1_keys_to_use = set(all_s1_keys)

    # 3. Stream S1 training records
    s1 = {}
    s1_path = args.train_dir / "train_source1.tsv"
    print(f"Reading S1 records from {s1_path.name}...", flush=True)
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in s1_keys_to_use:
                s1[eid] = normalize_record(row)
    print(f"Loaded {len(s1):,} S1 records into memory in {time.time()-t0:.1f}s", flush=True)

    # 4. Stream S2 & S3 into others + inverted index
    others: dict[str, NormRecord] = {}
    inv: dict[str, list[str]] = defaultdict(list)

    t1 = time.time()
    s2_path = args.train_dir / "train_source2.tsv"
    print(f"Streaming and indexing {s2_path.name}...", flush=True)
    with open(s2_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rec = normalize_record(row)
            others[rec.entity_id] = rec
            for k in _keys(rec):
                inv[k].append(rec.entity_id)
    print(f"  Indexed Source 2 in {time.time()-t1:.1f}s", flush=True)

    t2 = time.time()
    s3_path = args.train_dir / "train_source3.tsv"
    print(f"Streaming and indexing {s3_path.name}...", flush=True)
    with open(s3_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rec = normalize_record(row)
            others[rec.entity_id] = rec
            for k in _keys(rec):
                inv[k].append(rec.entity_id)
    print(f"  Indexed Source 3 in {time.time()-t2:.1f}s", flush=True)
    print(f"Total target records in memory: {len(others):,}", flush=True)

    gc.collect()

    # 5. Train/Val Split by S1 ID
    s1_ids = list(s1.keys())
    rng.shuffle(s1_ids)
    cut = max(1, int(len(s1_ids) * (1 - VAL_FRACTION)))
    train_ids, val_ids = s1_ids[:cut], s1_ids[cut:]

    s1_train = [s1[i] for i in train_ids]
    s1_val = [s1[i] for i in val_ids]

    print(f"Train/Val entities: {len(train_ids):,} train / {len(val_ids):,} val", flush=True)
    print("Generating candidates for train fold...", flush=True)
    c_train = stream_block(s1_train, others, inv)
    print("Generating candidates for val fold...", flush=True)
    c_val = stream_block(s1_val, others, inv)

    gold_train = {k: gold.get(k, []) for k in train_ids}
    gold_val = {k: gold.get(k, []) for k in val_ids}
    rec_tr = blocking_recall(c_train, gold_train)
    rec_va = blocking_recall(c_val, gold_val)
    print(f"Blocking recall: Train={rec_tr:.3f} | Val={rec_va:.3f}", flush=True)

    # 6. Build pairwise feature matrices
    print("Constructing pairwise training feature matrix...", flush=True)
    Xtr, ytr = matrix_from_pairs(iter_labeled_pairs(s1, others, c_train, gold, train_ids, rng))
    Xva, yva = matrix_from_pairs(iter_labeled_pairs(s1, others, c_val, gold, val_ids, rng))
    print(f"Pair dataset: Train={len(ytr):,} (positives={int(ytr.sum()):,}) | Val={len(yva):,} (positives={int(yva.sum()):,})", flush=True)

    # 7. Train LightGBM
    print("Fitting LightGBM binary classifier...", flush=True)
    model = train_lgbm(Xtr, ytr, Xva, yva)

    # 8. Score Val entities and sweep F0.5 threshold
    print("Sweeping probability threshold to optimize Macro F0.5...", flush=True)
    scores = {}
    for sid in val_ids:
        feats, oids = [], []
        for oid in c_val.get(sid, []):
            if oid not in others:
                continue
            feats.append(pair_features(s1[sid], others[oid]))
            oids.append(oid)
        if not oids:
            scores[sid] = []
            continue
        proba = predict_proba(model, np.asarray(feats, dtype=np.float32))
        scores[sid] = list(zip(oids, proba.tolist()))

    t_best, s_best = sweep_threshold(scores, gold_val)
    preds = {sid: [oid for oid, p in pairs if p >= t_best] for sid, pairs in scores.items()}
    for k in gold_val:
        preds.setdefault(k, [])
    final_f05 = macro_f05(preds, gold_val)
    print(f"Selected Threshold: {t_best:.3f} (Val Macro F0.5 = {final_f05:.4f})", flush=True)

    # 9. Save model bundle
    bundle_path = save_bundle(model, t_best)
    print(f"Saved trained model bundle to: {bundle_path}", flush=True)
    print(f"Total training pipeline completed in {(time.time()-t0)/60:.1f} minutes!", flush=True)


if __name__ == "__main__":
    main()
