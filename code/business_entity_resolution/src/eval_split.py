#!/usr/bin/env python3
"""Score matching_results.tsv against train_ground_truth.tsv with macro F0.5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_ground_truth, read_tsv
from src.metrics import blocking_recall, macro_f05


def _col(df, name_opts):
    for n in name_opts:
        if n in df.columns:
            return n
    raise SystemExit(f"need one of {name_opts}, got {list(df.columns)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matching", type=Path, required=True)
    p.add_argument("--candidate", type=Path, default=None)
    p.add_argument("--gold", type=Path, required=True)
    args = p.parse_args()

    gold_df = load_ground_truth(args.gold)
    gold = dict(zip(gold_df["source1_entity_id"], gold_df["matched_list"]))

    pred_df = read_tsv(args.matching)
    k = _col(pred_df, ["source1_entity_id"])
    v = _col(pred_df, ["matched_entity_ids"])
    preds = {
        r[k]: [x for x in str(r[v]).split(",") if x]
        for r in pred_df.to_dict(orient="records")
    }
    print(f"macro F0.5 = {macro_f05(preds, gold):.4f}  over {len(gold)} S1 entities")

    if args.candidate:
        c_df = read_tsv(args.candidate)
        ck = _col(c_df, ["source1_entity_id"])
        cv = _col(c_df, ["candidate_entity_ids"])
        cands = {
            r[ck]: [x for x in str(r[cv]).split(",") if x]
            for r in c_df.to_dict(orient="records")
        }
        print(f"blocking recall = {blocking_recall(cands, gold):.4f}")


if __name__ == "__main__":
    main()
