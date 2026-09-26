#!/usr/bin/env python3
"""Tiny synthetic dataset so the pipeline can be smoke-tested without contest files."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import write_tsv
import pandas as pd

TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"


def _row(eid, name, addr, country):
    return {
        "entity_id": eid,
        "business_name": name,
        "business_address": addr,
        "country": country,
    }


def main():
    TRAIN.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)

    s1 = [
        _row("S1-00001", "Acme Corporation", "12 Market Street, Springfield, IL 62701", "US"),
        _row("S1-00002", "Sharma Pvt Ltd", "Near SBI ATM, Andheri West, Mumbai 400053", "India"),
        _row("S1-00003", "Orchid Bakery", "14 Rue de Rivoli, 75001 Paris", "France"),
        _row("S1-00004", "Lone Star Grill", "500 Congress Ave, Austin TX 78701", "US"),
        _row("S1-00005", "Nanda Trading Co", "15 MG Road Bengaluru 560001", "India"),
    ]
    s2 = [
        _row("S2-00001", "ACME Corp", "12 Market St Springfield IL 62701", "US"),
        _row("S2-00002", "Acme Corp.", "Market Street Springfield", "US"),
        _row("S2-00003", "Sharma Private Limited", "Andheri W, Mumbai 400053", "India"),
        _row("S2-00004", "Boulangerie Orchid", "14 rue de Rivoli 75001 Paris", "France"),
        _row("S2-00005", "Unrelated Hardware", "88 Pine Rd, Dallas TX 75201", "US"),
    ]
    s3 = [
        _row("S3-00001", "Acme Corporation Inc", "12 Mkt St, Springfield 62701", "US"),
        _row("S3-00002", "Sharma Trading", "Andheri West Mumbai", "India"),
        _row("S3-00003", "Nanda Traders", "MG Road, Bangalore 560001", "India"),
        _row("S3-00004", "Random Pharma Ltd", "Banjara Hills Hyderabad 500034", "India"),
    ]
    gt = [
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001,S2-00002,S3-00001"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00003,S3-00002"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": "S2-00004"},
        {"source1_entity_id": "S1-00004", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-00005", "matched_entity_ids": "S3-00003"},
    ]

    # Train == test on dummy so both scripts have files. Replace with real data.
    for split, folder in (("train", TRAIN), ("test", TEST)):
        write_tsv(pd.DataFrame(s1), folder / f"{split}_source1.tsv")
        write_tsv(pd.DataFrame(s2), folder / f"{split}_source2.tsv")
        write_tsv(pd.DataFrame(s3), folder / f"{split}_source3.tsv")
    write_tsv(pd.DataFrame(gt), TRAIN / "train_ground_truth.tsv")
    print(f"Wrote dummy train/test under {ROOT / 'dataset'}")


if __name__ == "__main__":
    main()
