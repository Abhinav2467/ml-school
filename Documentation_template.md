# Methodology

## Problem formulation

Cross-source entity resolution with Source 1 as a deduplicated reference. For each S1 record predict the set of matching S2/S3 records (possibly empty). Evaluation is macro-averaged F0.5 over S1 entities, singletons included. False merges are more expensive than misses, so the decision threshold is precision-biased.

Country is an open string label. Training has US and India; test also has France. The pipeline never one-hots or filters to a frozen country set. Every test S1 row is emitted.

No external databases, geocoders, or commercial ER APIs are used.

## Pipeline overview

```
raw TSV
  → unicode/NFKC fold, suffix strip, address abbr expand, postal regex
  → multi-key blocking ∪ char TF-IDF top-k     → candidate_pairs.tsv
  → pairwise features
  → LightGBM probability
  → threshold chosen on S1-holdout F0.5        → matching_results.tsv
```

## Candidate generation / blocking

Keys (always namespaced by folded country string, not by a closed enum):

1. postal / PIN / ZIP / CP when present
2. sorted core-name tokens (legal suffixes removed)
3. 6-char alphanumeric prefix of core name
4. Soundex of first core token
5. first-two-token bigram
6. first street/building number
7. character-wb TF-IDF (3–5 grams) top-30 on `core + address`

Candidates are the union of those lists, country-agreed when both sides have a country, then capped (default 80) with priority: same postal → exact sorted name → soundex → token overlap.

This set is exactly what the matcher scores and exactly what is written to `candidate_pairs.tsv`.

## Model and features

Pairwise LightGBM (`binary_logloss`). Features:

- Levenshtein and Jaro-Winkler on full name and core name
- token Jaccard on name and address
- address Levenshtein
- postal equality / both-missing
- street-number overlap
- country-string equality (feature, not architecture)
- soundex / prefix equality
- length ratio, token-count difference, shared-token count

Training pairs: all gold links that survive lookup, plus blocked non-matches downsampled to ~8:1. Split **by S1 entity id** so a business cannot leak across train/val.

## Threshold

On the S1 holdout, sweep probability in `[0.35, 0.95]` and keep the value that maximises macro F0.5, including singletons (empty/empty = 1.0, empty/non-empty = 0.0). Default fallback 0.72 if no model is present.

Any candidate with score ≥ threshold is kept (multi-match allowed). Below threshold → singleton.

## France / unseen countries

Normalization tables include common FR legal suffixes (`sarl`, `sas`, `sa`, …) and a 5-digit postal regex that also covers French codes. Blocking keys are string-valued. No train-time country dummy variables.

## What not to do

- Do not look up businesses on the internet or in registries.
- Do not geocode.
- Do not drop S1 rows whose country was unseen in train.
- Do not optimise F1 or pair-AUC and then ship that threshold.

## Reproducibility

See `code/business_entity_resolution/README.md`. Pinned dependencies in `requirements.txt`. Model license target: LightGBM (MIT) + RapidFuzz (MIT) + scikit-learn (BSD). No model above 8B parameters.
