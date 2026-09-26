#!/usr/bin/env python3
"""Run blocking + matcher on a split and write the two required TSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocking import generate_candidates
from src.config import DEFAULT_THRESHOLD, OUTPUT_DIR, TEST_DIR
from src.features import pair_features
from src.io_utils import format_id_list, load_split, write_tsv
from src.model import load_bundle, predict_proba
from src.normalize import normalize_record
from src.rules import high_precision_match


def _index(df):
    return {r["entity_id"]: normalize_record(r) for r in df.to_dict(orient="records")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=TEST_DIR)
    p.add_argument("--prefix", default="test", help="file prefix: test or train")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    sources = load_split(args.data_dir, args.prefix)
    s1 = _index(sources["s1"])
    s2 = list(_index(sources["s2"]).values())
    s3 = list(_index(sources["s3"]).values())
    others = {r.entity_id: r for r in s2 + s3}

    print(f"Loaded S1={len(s1)} S2={len(s2)} S3={len(s3)}")
    cands = generate_candidates(list(s1.values()), s2, s3)

    bundle = None
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    try:
        bundle = load_bundle(args.model)
        model = bundle["model"]
        if args.threshold is None:
            threshold = float(bundle["threshold"])
        print(f"Loaded model threshold={threshold:.3f}")
    except FileNotFoundError:
        print("No trained model found — falling back to Jaro-Winkler core-name score.")
        model = None

    match_rows, cand_rows = [], []
    s1_ids = list(sources["s1"]["entity_id"])
    for sid in s1_ids:
        cand_ids = cands.get(sid, [])
        cand_rows.append({"source1_entity_id": sid, "candidate_entity_ids": format_id_list(cand_ids)})
        if not cand_ids:
            match_rows.append({"source1_entity_id": sid, "matched_entity_ids": ""})
            continue
        rec = s1[sid]
        kept = []
        rule_hit = {oid for oid in cand_ids if high_precision_match(rec, others[oid])}
        if model is None:
            from rapidfuzz.distance import JaroWinkler
            for oid in cand_ids:
                o = others[oid]
                score = JaroWinkler.similarity(rec.core or rec.name, o.core or o.name)
                postal_boost = 0.15 if rec.postal and rec.postal == o.postal else 0.0
                if oid in rule_hit or score + postal_boost >= threshold:
                    kept.append(oid)
        else:
            X = np.asarray([pair_features(rec, others[oid]) for oid in cand_ids], dtype=np.float32)
            proba = predict_proba(model, X)
            kept = [
                oid for oid, pr in zip(cand_ids, proba)
                if pr >= threshold or oid in rule_hit
            ]
        match_rows.append({"source1_entity_id": sid, "matched_entity_ids": format_id_list(kept)})

    out = args.out_dir
    write_tsv(pd.DataFrame(match_rows), out / "matching_results.tsv")
    write_tsv(pd.DataFrame(cand_rows), out / "candidate_pairs.tsv")
    n_match = sum(1 for r in match_rows if r["matched_entity_ids"])
    print(f"Wrote {out/'matching_results.tsv'}  ({n_match}/{len(match_rows)} S1 with ≥1 match)")
    print(f"Wrote {out/'candidate_pairs.tsv'}")


if __name__ == "__main__":
    main()
