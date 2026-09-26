# Concrete architecture

This is the system the starter implements. Swap LightGBM for a small encoder later if you want; do not change the stage boundaries.

```
raw TSV  →  normalize  →  block  →  candidate_pairs.tsv
                              ↓
                    pairwise features
                              ↓
              rules OR LightGBM ≥ threshold
                              ↓
                    matching_results.tsv
```

Metric to optimise at every decision: **macro F0.5 over Source 1 entities, singletons included**.

- true empty + pred empty → 1.0
- true empty + any pred id → 0.0
- precision weighted 2× recall

Never tune on F1.

---

## 1. Normalization

File: `src/normalize.py`

Country is an open string. Tables are extensible; there is no `if country in {"US","India"}`.

| Step | What |
|---|---|
| Fold | NFKC, unidecode, lower, `&`→`and`, strip punctuation |
| Name core | drop legal suffixes (`ltd`, `pvt`, `inc`, `sarl`, `sas`, `sa`, …) and stopwords |
| Address | expand `rd/st/ave`, drop `near/opp/landmark …` tails |
| Postal | stacked regex: 5-digit / ZIP+4, 6-digit PIN, UK-like |
| Keys | `core`, `core_sorted`, 8-char prefix, soundex, street numbers, token sets |

Derived fields are the only thing blocking and features see.

---

## 2. Blocking keys

File: `src/blocking.py`

Union of cheap exact keys plus TF-IDF retrieval. This **is** `candidate_pairs.tsv` — the last list the model scores.

| Key | Example | Catches |
|---|---|---|
| `post\|{country}\|62701` | same ZIP / PIN / CP | strongest cheap block |
| `sorted\|{country}\|acme` | suffix-stripped sorted tokens | Corp vs Corporation |
| `pref\|{country}\|acme` | 6-char core prefix | typos after char 6 |
| `sx\|{country}\|A250` | soundex of first core token | Sharma / Sharmma |
| `bigram\|{country}\|acme corp` | first two name tokens | short names |
| `num\|{country}\|12` | first street number | same building, different name order |
| char-wb TF-IDF 3–5g top-30 on `core + address` | messy residuals | word-order / partial address |

Then:

- drop cross-country pairs when **both** sides have a country string
- cap at 80, ranking: same postal → exact sorted name → soundex → token overlap

Measure **pair-recall of gold links inside this set** on a Source-1 holdout. If it is not ~0.95+, add a key. Do not tighten blocking to “help precision”; the matcher does that.

---

## 3. Pairwise features

File: `src/features.py`

| Feature | Role |
|---|---|
| Levenshtein + Jaro-Winkler on full name and core name | typos, suffixes |
| Jaro-Winkler on **sorted** core tokens | “Orchid Bakery” vs “Bakery Orchid” |
| Token Jaccard name / address | overlap |
| Address Levenshtein | format variants |
| Postal equal / both missing | PIN/ZIP/CP |
| Street-number overlap / first-number equal | building |
| Country-string equal | feature, not architecture |
| Soundex / prefix equal | cheap phonetic / prefix |
| Length ratio, token-count diff | scale |
| Shared token count, containment, shared token ≥5 chars | distinctive words |

No geocoding. No registry lookup. No embeddings in the starter (add a small MIT/Apache encoder later if val F0.5 stalls).

---

## 4. Training-pair construction

File: `src/pairs.py`, `src/train.py`

1. Split **by Source 1 entity id** (not by pair). Same business must not appear in both folds.
2. Block train S1 against all S2/S3.
3. Positives = gold S2/S3 ids that exist in the source files.
4. Negatives = blocked ids that are **not** gold, downsampled to about 8:1.
5. Fit LightGBM (`binary_logloss`).
6. On the val S1 entities, score every candidate and sweep threshold.

Do not sample easy random negatives from outside the block — they make the model look good and fail on the pairs you actually score at inference.

---

## 5. Decision rule / threshold

Files: `src/model.py` (`sweep_threshold`), `src/rules.py`, `src/infer.py`

For each candidate:

```
keep if high_precision_rule(a, b)
    or P_model(a, b) ≥ t*
```

High-precision rules (must-link only):

- same sorted core (length ≥ 4) and same country
- same postal and core JW ≥ 0.88
- same postal and a shared token of length ≥ 5
- same postal + same street number + (JW ≥ 0.80 or long shared token)
- sorted-token JW ≥ 0.93 and (postal or street number)

`t*` is chosen on the S1 holdout to **maximise macro F0.5**, grid `0.35 … 0.95`. Default if the fold is tiny: `0.72`.

All candidates above `t*` are kept (1-to-many is allowed). Below `t*` and no rule → singleton.

If two S1 entities claim the same S2/S3 id and train shows that is rare, add a conflict resolver later: keep the higher score. Not in the starter.

---

## 6. Scoring you should log every run

File: `src/metrics.py`, `src/eval_split.py`

- blocking recall (gold links present in `candidate_pairs.tsv`)
- reduction ratio (pairs scored / all S1×(S2∪S3))
- macro F0.5 including singletons
- singleton accuracy (empty/empty vs false merge)
- F0.5 vs threshold curve
- per-country F0.5 (US vs India on val; France only exists on test)

Baseline to beat first: “always emit empty”. If most S1 are singletons that number is already high.

---

## 7. Submission contract

```
output/matching_results.tsv     scored
output/candidate_pairs.tsv      audited; matches ⊆ candidates
```

Rules the validator enforces:

- one row per test S1 id, including France
- tab-separated, exact column names
- S2/S3 ids only, no duplicates, no S1 self-matches
- empty string when there is no match

```bash
python utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

Final zip also needs `code/business_entity_resolution/{src,README.md,requirements.txt}` and `Documentation_template.md`. Model: MIT/Apache, ≤ 8B params. LightGBM + RapidFuzz + sklearn satisfies that.

---

## 8. What to do on the real data, in order

1. Profile: sizes, country counts, singleton rate, multi-match rate, suffix histogram, postal coverage.
2. Run this starter unchanged. Record blocking recall and F0.5 vs always-empty.
3. Add blocking keys until val pair-recall is ≥ ~0.95.
4. Sweep threshold on F0.5. Raise it until singleton false-merges stop.
5. Only then add embeddings or a small cross-encoder on leftover hard pairs.
6. Train on US, score on India (and the reverse). If that gap is large, France will be worse — fix normalization, not the model class.
7. Ship the same `matching_results.tsv` you validated locally.

Do not look up businesses, registries, or geocoders. That is an automatic DQ.
