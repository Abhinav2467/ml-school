# Business Entity Resolution

End-to-end pipeline: normalize → multi-key + TF-IDF blocking → LightGBM pairwise matcher → F0.5 threshold.

Country is treated as an **open string**. Do not one-hot or filter to `{US, India}`. France (and any later label) must still get a row.

## Layout

```
src/           pipeline
utils/         local submission validator
artifacts/     trained matcher.joblib (created after train)
```

Contest data is expected at `../../dataset/{train,test}/` relative to this folder
(i.e. `ber_starter/dataset/...` in the starter package).

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

From `code/business_entity_resolution/`:

```bash
# 1. If you do not have contest files yet, generate a tiny synthetic set
python src/make_dummy_data.py

# 2. Train + pick F0.5 threshold on a Source-1 holdout
python src/train.py --train-dir ../../dataset/train

# 3. Infer on the test split (writes both required TSVs)
python src/infer.py --data-dir ../../dataset/test --prefix test --out-dir ../../output

# 4. Format check (does not score)
python utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

`matching_results.tsv` is the leaderboard file. `candidate_pairs.tsv` is the exact candidate set the model scored.

## Design notes

- Blocking is a **union** of country+postal, sorted core-name, prefix, soundex, token bigram, street number, and char-wb TF-IDF top-k.
- Matcher features are string similarities only (no external lookup, no geocoding).
- Threshold is swept to maximise **macro F0.5 including singletons**, not F1.
- Fallback: if no `artifacts/matcher.joblib` exists, infer uses Jaro-Winkler + postal boost so you can still emit valid TSVs.
