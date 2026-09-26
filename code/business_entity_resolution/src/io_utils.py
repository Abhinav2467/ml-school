from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def load_source(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Did you forget sep='\\t'?")
    return df


def load_split(split_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    return {
        "s1": load_source(split_dir / f"{prefix}_source1.tsv"),
        "s2": load_source(split_dir / f"{prefix}_source2.tsv"),
        "s3": load_source(split_dir / f"{prefix}_source3.tsv"),
    }


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    if "matched_entity_ids" not in df.columns:
        df["matched_entity_ids"] = ""
    df["matched_list"] = df["matched_entity_ids"].apply(
        lambda x: [i for i in str(x).split(",") if i]
    )
    return df


def format_id_list(ids: list[str]) -> str:
    seen = []
    for i in ids:
        if i and i not in seen:
            seen.append(i)
    return ",".join(seen)
