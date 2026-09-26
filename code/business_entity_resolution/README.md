# Amazon ML Challenge 2026 — Business Entity Resolution

End-to-end pipeline for matching business records across three noisy sources.

---

## Repository Layout

```
code/business_entity_resolution/
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py          # TSV loaders with schema validation
│   │   ├── normalize_names.py    # Legal-suffix stripping, DBA, &→and, unicode→ASCII
│   │   └── normalize_addresses.py# Abbreviation expansion + token-sort form
│   ├── blocking/
│   │   ├── embed_records.py      # Batch embedding + disk cache (BAAI/bge-large-en-v1.5)
│   │   ├── build_hnsw_index.py   # HNSW index over S2+S3 embeddings
│   │   └── generate_candidates.py# Top-k retrieval → candidate_pairs.tsv
│   ├── features/
│   │   └── pairwise_features.py  # 16 pairwise features per (S1, candidate) pair
│   ├── model/
│   │   ├── train_classifier.py   # XGBoost/LightGBM training
│   │   ├── tune_threshold.py     # F_0.5-optimal threshold sweep on val split
│   │   └── predict.py            # Full inference → matching_results.tsv
│   └── evaluate/
│       └── score_f05.py          # Local macro F_0.5 scorer
├── README.md
└── requirements.txt

utils/
└── validate_submission.py        # Pre-submission validator (must PASS before upload)

output/
├── matching_results.tsv          # Final submission (scored)
└── candidate_pairs.tsv           # Blocking set (unscored, validated)

dataset/
├── train/
│   ├── source1.tsv
│   ├── source2.tsv
│   ├── source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── source1.tsv
    ├── source2.tsv
    └── source3.tsv
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 required.

---

## End-to-End Reproduction

All commands are run from the **project root** (`Amazon_ML_2026/`).

### Phase 1 — Preprocessing (no CLI needed; happens inside later steps)

The normalization modules (`normalize_names`, `normalize_addresses`) are called
automatically by the embedding and training steps. To unit-test them:

```bash
python -c "
from code.business_entity_resolution.src.preprocessing.normalize_names import normalize_name
from code.business_entity_resolution.src.preprocessing.normalize_addresses import normalize_address
print(normalize_name('Acme Corp. & Sons Pvt. Ltd.'))
print(normalize_address('123 Main St., Apt 4B'))
"
```

---

### Phase 2 — Blocking

**Step 2a — Embed all sources (train)**

```bash
python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/train/source1.tsv --source s1_train --cache-dir .cache/embeddings

python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/train/source2.tsv --source s2_train --cache-dir .cache/embeddings

python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/train/source3.tsv --source s3_train --cache-dir .cache/embeddings
```

**Step 2b — Build HNSW index (train)**

```bash
python -m code.business_entity_resolution.src.blocking.build_hnsw_index \
    --s2-emb  .cache/embeddings/s2_train_embeddings.npy \
    --s3-emb  .cache/embeddings/s3_train_embeddings.npy \
    --s2-ids  dataset/train/source2.tsv \
    --s3-ids  dataset/train/source3.tsv \
    --index-dir .cache/index --tag train
```

**Step 2c — Generate candidates (train)**

```bash
python -m code.business_entity_resolution.src.blocking.generate_candidates \
    --s1-emb    .cache/embeddings/s1_train_embeddings.npy \
    --s1-tsv    dataset/train/source1.tsv \
    --index-dir .cache/index --index-tag train \
    --k 50 \
    --output    output/candidate_pairs_train.tsv \
    --gt        dataset/train/train_ground_truth.tsv
```
> **Check the printed recall ceiling.** It must be high (>0.90 ideally) before continuing.

---

### Phase 3+4 — Training

```bash
python -m code.business_entity_resolution.src.model.train_classifier \
    --candidates  output/candidate_pairs_train.tsv \
    --s1-tsv      dataset/train/source1.tsv \
    --s2-tsv      dataset/train/source2.tsv \
    --s3-tsv      dataset/train/source3.tsv \
    --s1-emb      .cache/embeddings/s1_train_embeddings.npy \
    --s2-emb      .cache/embeddings/s2_train_embeddings.npy \
    --s3-emb      .cache/embeddings/s3_train_embeddings.npy \
    --gt          dataset/train/train_ground_truth.tsv \
    --model-dir   .cache/model \
    --backend     xgboost
```

---

### Phase 4 — Threshold tuning (on held-out validation split)

> Generate a validation candidate set and features first (same commands as above
> but pointing to your val split TSVs), then:

```bash
python -m code.business_entity_resolution.src.model.tune_threshold \
    --candidates  output/candidate_pairs_val.tsv \
    --features    output/features_val.npy \
    --gt          dataset/train/train_ground_truth.tsv \
    --s1-ids      dataset/train/source1.tsv \
    --model-dir   .cache/model \
    --backend     xgboost
```

---

### Phase 5 — Inference on test set

**Embed test sources:**

```bash
python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/test/source1.tsv --source s1_test --cache-dir .cache/embeddings
python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/test/source2.tsv --source s2_test --cache-dir .cache/embeddings
python -m code.business_entity_resolution.src.blocking.embed_records \
    --input dataset/test/source3.tsv --source s3_test --cache-dir .cache/embeddings
```

**Build test index + generate test candidates:**

```bash
python -m code.business_entity_resolution.src.blocking.build_hnsw_index \
    --s2-emb .cache/embeddings/s2_test_embeddings.npy \
    --s3-emb .cache/embeddings/s3_test_embeddings.npy \
    --s2-ids dataset/test/source2.tsv \
    --s3-ids dataset/test/source3.tsv \
    --index-dir .cache/index --tag test

python -m code.business_entity_resolution.src.blocking.generate_candidates \
    --s1-emb    .cache/embeddings/s1_test_embeddings.npy \
    --s1-tsv    dataset/test/source1.tsv \
    --index-dir .cache/index --index-tag test \
    --k 50 \
    --output    output/candidate_pairs.tsv
```

**Run predict:**

```bash
python -m code.business_entity_resolution.src.model.predict \
    --candidates  output/candidate_pairs.tsv \
    --s1-tsv      dataset/test/source1.tsv \
    --s2-tsv      dataset/test/source2.tsv \
    --s3-tsv      dataset/test/source3.tsv \
    --s1-emb      .cache/embeddings/s1_test_embeddings.npy \
    --s2-emb      .cache/embeddings/s2_test_embeddings.npy \
    --s3-emb      .cache/embeddings/s3_test_embeddings.npy \
    --model-dir   .cache/model \
    --backend     xgboost \
    --output-dir  output
```

---

### Phase 5 — Validate before submission

```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

Must print `PASS` before uploading.

---

## Open Parameters

| Parameter | Current default | Notes |
|---|---|---|
| Retrieval depth `k` | 50 | Raise to 75–100 if recall ceiling is low |
| Embedding model | `BAAI/bge-large-en-v1.5` | Swap in `embed_records.py:EMBED_MODEL_NAME` |
| Classifier | XGBoost | `--backend lightgbm` to switch |
| Decision threshold | Tuned empirically on val split | `tune_threshold.py` writes `best_threshold.txt` |

---

## Scoring Reference

`F_0.5 = 1.25 × P × R / (0.25 × P + R)` — precision weighted 2× over recall.
Singletons (no true match): 1.0 if prediction empty, 0.0 for any false positive.
