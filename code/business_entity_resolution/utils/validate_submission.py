#!/usr/bin/env python3
"""Local format validator (stdlib only), matching the contest helper's spirit."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _read_ids(path: Path, col: str) -> set[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f, delimiter="\t")
        if col not in (rows.fieldnames or []):
            raise SystemExit(f"{path} missing column {col}")
        return {r[col] for r in rows if r.get(col)}


def _read_map(path: Path, key: str, val: str) -> dict[str, list[str]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path} is empty")
    if rows[0].keys() == {None} or key not in rows[0]:
        raise SystemExit(f"{path} is not tab-separated or missing {key}")
    for r in rows:
        sid = r[key]
        ids = [x for x in (r.get(val) or "").split(",") if x]
        out.setdefault(sid, [])
        out[sid].extend(ids)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matching", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--test-dir", required=True, type=Path)
    args = p.parse_args()
    issues = []

    s1 = _read_ids(args.test_dir / "test_source1.tsv", "entity_id")
    s2 = _read_ids(args.test_dir / "test_source2.tsv", "entity_id")
    s3 = _read_ids(args.test_dir / "test_source3.tsv", "entity_id")
    allowed = s2 | s3

    matches = _read_map(args.matching, "source1_entity_id", "matched_entity_ids")
    cands = _read_map(args.candidate, "source1_entity_id", "candidate_entity_ids")

    if set(matches) != s1:
        missing = s1 - set(matches)
        extra = set(matches) - s1
        if missing:
            issues.append(f"matching missing {len(missing)} S1 ids e.g. {next(iter(missing))}")
        if extra:
            issues.append(f"matching has {len(extra)} unknown S1 ids e.g. {next(iter(extra))}")
    if set(cands) != s1:
        missing = s1 - set(cands)
        extra = set(cands) - s1
        if missing:
            issues.append(f"candidates missing {len(missing)} S1 ids")
        if extra:
            issues.append(f"candidates have {len(extra)} unknown S1 ids")

    for sid, ids in matches.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{sid}: duplicate matched ids")
        for i in ids:
            if i not in allowed:
                issues.append(f"{sid}: matched id {i} not in test S2/S3")
            if i.startswith("S1-"):
                issues.append(f"{sid}: self-match {i}")
            if i not in set(cands.get(sid, [])):
                issues.append(f"{sid}: matched {i} never appeared as a candidate")

    for sid, ids in cands.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{sid}: duplicate candidate ids")
        for i in ids:
            if i not in allowed:
                issues.append(f"{sid}: candidate {i} not in test S2/S3")

    if issues:
        print("FAIL")
        for i, msg in enumerate(issues[:50], 1):
            print(f"{i}. {msg}")
        if len(issues) > 50:
            print(f"... {len(issues) - 50} more")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
