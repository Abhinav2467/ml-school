#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/code/business_entity_resolution"
cd "$CODE"
python3 src/make_dummy_data.py
python3 src/train.py --train-dir "$ROOT/dataset/train"
python3 src/infer.py --data-dir "$ROOT/dataset/test" --prefix test --out-dir "$ROOT/output"
python3 utils/validate_submission.py \
  --matching "$ROOT/output/matching_results.tsv" \
  --candidate "$ROOT/output/candidate_pairs.tsv" \
  --test-dir "$ROOT/dataset/test"
python3 src/eval_split.py \
  --matching "$ROOT/output/matching_results.tsv" \
  --candidate "$ROOT/output/candidate_pairs.tsv" \
  --gold "$ROOT/dataset/train/train_ground_truth.tsv"
echo "--- matching_results.tsv ---"
cat "$ROOT/output/matching_results.tsv"
echo "--- candidate_pairs.tsv ---"
cat "$ROOT/output/candidate_pairs.tsv"
