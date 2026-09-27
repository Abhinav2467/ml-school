#!/usr/bin/env python3
"""Streaming blocking + matcher inference pipeline. Memory-safe for large splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocking import _keys
from src.config import (
    DEFAULT_THRESHOLD,
    MAX_CANDIDATES_PER_S1,
    OUTPUT_DIR,
    SAME_COUNTRY_ONLY_IN_BLOCK,
    TEST_DIR,
)
from src.features import pair_features
from src.io_utils import format_id_list
from src.model import load_bundle, predict_proba
from src.normalize import normalize_record
from src.rules import high_precision_match


def load_source_indexed(tsv_path: Path, others: dict, inv: dict[str, list[str]]) -> int:
    """Stream TSV directly into others dict and inverted index (zero pandas memory overhead)."""
    count = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rec = normalize_record(row)
            others[rec.entity_id] = rec
            for k in _keys(rec):
                inv[k].append(rec.entity_id)
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=TEST_DIR)
    p.add_argument("--prefix", default="test", help="file prefix: test or train")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    bundle = None
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    try:
        bundle = load_bundle(args.model)
        model = bundle["model"]
        if args.threshold is None:
            threshold = float(bundle["threshold"])
        print(f"Loaded trained model (threshold={threshold:.3f})", flush=True)
    except FileNotFoundError:
        print("No trained model found — falling back to Jaro-Winkler core-name score.", flush=True)
        model = None

    others = {}
    inv: dict[str, list[str]] = defaultdict(list)

    t0 = time.time()
    s2_path = args.data_dir / f"{args.prefix}_source2.tsv"
    print(f"Streaming and indexing {s2_path.name}...", flush=True)
    n2 = load_source_indexed(s2_path, others, inv)
    print(f"  Indexed {n2:,} Source 2 records in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    s3_path = args.data_dir / f"{args.prefix}_source3.tsv"
    print(f"Streaming and indexing {s3_path.name}...", flush=True)
    n3 = load_source_indexed(s3_path, others, inv)
    print(f"  Indexed {n3:,} Source 3 records in {time.time()-t1:.1f}s", flush=True)
    print(f"Total target records indexed: {len(others):,} (in {time.time()-t0:.1f}s)", flush=True)

    gc.collect()

    s1_path = args.data_dir / f"{args.prefix}_source1.tsv"
    print(f"Processing {s1_path.name} and generating matches...", flush=True)

    match_out_path = out / "matching_results.tsv"
    cand_out_path = out / "candidate_pairs.tsv"

    t_start = time.time()
    processed = 0
    n_match = 0

    with open(match_out_path, "w", encoding="utf-8", newline="") as f_match, \
         open(cand_out_path, "w", encoding="utf-8", newline="") as f_cand:

        match_writer = csv.writer(f_match, delimiter="\t", lineterminator="\n")
        cand_writer = csv.writer(f_cand, delimiter="\t", lineterminator="\n")

        # Header rows
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        with open(s1_path, "r", encoding="utf-8") as f_s1:
            reader = csv.DictReader(f_s1, delimiter="\t")
            for row in reader:
                rec = normalize_record(row)
                sid = rec.entity_id
                processed += 1

                if processed % 50000 == 0:
                    elapsed = time.time() - t_start
                    rate = processed / max(elapsed, 0.001)
                    print(f"  [{processed:,} records processed] ({rate:.0f} rec/s, elapsed: {elapsed/60:.1f}m)", flush=True)

                # 1. Blocking candidates from inverted index
                seen = set()
                for k in _keys(rec):
                    for oid in inv.get(k, ()):
                        o = others[oid]
                        if SAME_COUNTRY_ONLY_IN_BLOCK and rec.country_key and o.country_key:
                            if rec.country_key != o.country_key:
                                continue
                        seen.add(oid)

                cand_ids = list(seen)
                if len(cand_ids) > MAX_CANDIDATES_PER_S1:
                    def score(oid: str) -> tuple:
                        o = others[oid]
                        return (
                            int(bool(rec.postal and rec.postal == o.postal)),
                            int(rec.core_sorted == o.core_sorted and rec.core_sorted != ""),
                            int(rec.soundex == o.soundex),
                            len(rec.tokens_name & o.tokens_name),
                        )
                    cand_ids = sorted(cand_ids, key=score, reverse=True)[:MAX_CANDIDATES_PER_S1]

                cand_writer.writerow([sid, format_id_list(cand_ids)])

                if not cand_ids:
                    match_writer.writerow([sid, ""])
                    continue

                # 2. Matching rules / model scoring
                rule_hit = {oid for oid in cand_ids if high_precision_match(rec, others[oid])}
                kept = []
                if model is None:
                    from rapidfuzz.distance import JaroWinkler
                    for oid in cand_ids:
                        o = others[oid]
                        s = JaroWinkler.similarity(rec.core or rec.name, o.core or o.name)
                        postal_boost = 0.15 if rec.postal and rec.postal == o.postal else 0.0
                        if oid in rule_hit or s + postal_boost >= threshold:
                            kept.append(oid)
                else:
                    X = np.asarray([pair_features(rec, others[oid]) for oid in cand_ids], dtype=np.float32)
                    proba = predict_proba(model, X)
                    kept = [
                        oid for oid, pr in zip(cand_ids, proba)
                        if pr >= threshold or oid in rule_hit
                    ]

                if kept:
                    n_match += 1
                match_writer.writerow([sid, format_id_list(kept)])

    total_time = time.time() - t_start
    print(f"Finished in {total_time/60:.1f} minutes! ({processed:,} records processed, {n_match:,} with matches)", flush=True)
    print(f"Wrote {match_out_path}", flush=True)
    print(f"Wrote {cand_out_path}", flush=True)


if __name__ == "__main__":
    main()
