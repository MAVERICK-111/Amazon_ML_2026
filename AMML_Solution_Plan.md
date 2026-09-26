# Amazon ML Challenge 2026 — Business Entity Resolution
## Solution Plan & Architecture

---

## 1. Problem Recap

Match business records across 3 independent, noisy sources:

- **Source 1** — deduplicated reference set. Every S1 entity must appear in the submission.
- **Source 2 / Source 3** — noisy record sets. For each S1 entity, find every matching record in S2 and/or S3 (zero, one, or many matches).

Fields: `entity_id`, `business_name`, `business_address`, `country`. Country is an **open set** — training only has US/India, test adds France. Never hardcode or filter on specific country values.

**Noise to expect:** legal-suffix variants (Corp/Corporation, Pvt/Private), DBA names, punctuation (& vs "and"), word-order swaps, typos, transliteration, abbreviations (Rd/Road, St/Street), missing address components, landmark references, municipal numbering variants, component reordering.

**Scoring:** macro-averaged F_0.5 per S1 entity (precision weighted 2x over recall). Singletons (no true match) score 1.0 for a correctly empty prediction, 0.0 for any false positive — so correctly predicting "no match" is worth full credit, and over-matching is punished hard.

**Hard constraints:**
- No external lookups (no APIs, no geocoding, no registries, no internet augmentation) — training data only.
- Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.
- Output: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (blocking set, unscored but validated — matches must be a subset of candidates).
- Must pass `utils/validate_submission.py` before every leaderboard upload.

---

## 2. Chosen Architecture

| Stage | Approach |
|---|---|
| **Preprocessing** | Normalize names & addresses; treat `country` as an arbitrary string label |
| **Blocking / candidate generation** | Embed `business_name` + `business_address` per record with a large open-source embedding model (MIT/Apache, near 8B params) → build an **HNSW** approximate nearest-neighbor index → for each S1 entity, retrieve top-*k* nearest S2/S3 records |
| **Feature engineering** | Per (S1, candidate) pair: string-similarity features (Levenshtein, Jaccard/token overlap, TF-IDF cosine) on name and address separately, plus embedding cosine similarity, plus lightweight structural features (country match, length deltas) |
| **Matching model** | Gradient-boosted tree classifier (XGBoost or LightGBM) trained on labeled pairs from `train_ground_truth.tsv`, outputting a match probability per pair |
| **Thresholding** | Tune classifier decision threshold on a held-out validation split to maximize macro F_0.5 (biased toward precision) |
| **Output generation** | All HNSW-retrieved pairs → `candidate_pairs.tsv`; pairs above threshold, grouped per S1 entity → `matching_results.tsv` |

### Why this combination
- HNSW blocking scales sub-linearly and is robust to the typo/transliteration noise that breaks pure rule-based blocking keys.
- A large embedding model gives strong semantic recall at the blocking stage, where recall ceiling matters most.
- Combining string-similarity features *with* embedding similarity in a tree classifier keeps the final decision interpretable and precision-tunable — important since F_0.5 punishes false merges twice as hard as misses, and tree models handle heterogeneous, noisy tabular features well.

### Key risk to manage
Encoding every record in all 3 sources with a large (~8B-param) embedding model is the most expensive step in the pipeline. Plan to:
- Batch encode and cache embeddings to disk (never re-embed the same record twice across experiments).
- Fix the retrieval depth *k* early — it sets the recall ceiling for everything downstream and directly controls the size of `candidate_pairs.tsv` and classifier training data.
- Keep a small dev subset for fast iteration on the classifier/threshold before running full-scale.

---

## 3. Repository / File Structure

```
code/business_entity_resolution/
├── src/
│   ├── preprocessing/
│   │   ├── normalize_names.py       # legal suffix stripping, casing, punctuation
│   │   ├── normalize_addresses.py   # abbreviation expansion, component parsing
│   │   └── load_data.py             # TSV loaders with sep="\t", schema checks
│   ├── blocking/
│   │   ├── embed_records.py         # batch embedding + disk cache
│   │   ├── build_hnsw_index.py      # index construction per source
│   │   └── generate_candidates.py   # top-k retrieval → candidate_pairs.tsv
│   ├── features/
│   │   └── pairwise_features.py     # string sim + embedding sim + structural features
│   ├── model/
│   │   ├── train_classifier.py      # XGBoost/LightGBM training on labeled pairs
│   │   ├── tune_threshold.py        # F_0.5-optimal threshold search on val split
│   │   └── predict.py               # inference → matching_results.tsv
│   └── evaluate/
│       └── score_f05.py             # local macro F_0.5 scorer on held-out split
├── README.md                        # end-to-end reproduction instructions
└── requirements.txt                 # pinned dependencies
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

---

## 4. Build Order (Phased)

**Phase 1 — Data & preprocessing**
- Load all TSVs with explicit `sep="\t"`.
- Build name/address normalization functions; unit-test against the noise patterns listed in the problem statement.
- Hold out a validation split from training data (stratified so it contains a realistic mix of singletons and multi-matches).

**Phase 2 — Blocking**
- Embed all normalized records (S1, S2, S3) with the chosen large embedding model; cache to disk.
- Build HNSW index over S2+S3 embeddings.
- Retrieve top-*k* candidates per S1 entity → write `candidate_pairs.tsv`.
- Sanity-check recall ceiling: on the validation split, what fraction of true matches are actually present in the candidate set? This number caps your final F_0.5 no matter how good the classifier is.

**Phase 3 — Feature engineering**
- For every (S1, candidate) pair in `candidate_pairs.tsv`, compute the full feature vector (string similarities on name/address, embedding cosine, structural features).
- Label pairs using `train_ground_truth.tsv` for the training split.

**Phase 4 — Matching model**
- Train XGBoost/LightGBM on labeled pairs (train split).
- Tune the decision threshold on the validation split, optimizing macro F_0.5 directly — not accuracy or plain F1.
- Inspect precision/recall trade-off explicitly; err toward precision given the metric's weighting.

**Phase 5 — Inference & output**
- Run the full pipeline on the test set (S1/S2/S3 test files) → generate `matching_results.tsv` and `candidate_pairs.tsv`.
- Run `utils/validate_submission.py` before every leaderboard upload.

**Phase 6 — Iteration**
- Use validation-split F_0.5 (per-entity macro average, singleton-inclusive) to guide changes to *k*, features, or classifier hyperparameters.
- Watch specifically for over-matching on singletons — it's the fastest way to lose points under this metric.

**Phase 7 — Packaging**
- Fill in `Documentation_template.md` (methodology, blocking strategy, model architecture, feature engineering).
- Assemble final submission zip: `output/`, `code/business_entity_resolution/` (with README + requirements.txt), and the methodology doc.

---

## 5. Validation Checklist (before every submission)

- [ ] Every test-set S1 entity has exactly one row in `matching_results.tsv`.
- [ ] Singletons have an empty `matched_entity_ids` field, not omitted rows.
- [ ] No duplicate entity IDs within any ID list; no duplicate `source1_entity_id` rows.
- [ ] Every ID in `matching_results.tsv` also appears in `candidate_pairs.tsv` for that S1 entity.
- [ ] All matched/candidate IDs are S2-/S3- prefixed and exist in the test set (no self-matches to S1).
- [ ] Ran `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` and got `PASS`.

---

## 6. Open Parameters to Decide During Implementation

- HNSW retrieval depth *k* (recall/compute trade-off).
- Specific embedding model checkpoint (near 8B params, MIT/Apache license).
- Classifier choice: XGBoost vs LightGBM (largely an implementation preference; both fit the plan).
- Final decision threshold (set empirically from the validation split, not fixed in advance).
