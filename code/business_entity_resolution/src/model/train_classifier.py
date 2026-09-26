"""
train_classifier.py — Train XGBoost or LightGBM on labeled pairs.

Workflow:
1. Load candidate_pairs_train.tsv (train split)
2. Load source TSVs + embeddings → compute pairwise features
3. Label pairs using train_ground_truth.tsv (1 = match, 0 = non-match)
4. Train classifier; save model to disk.

Usage:
    python -m src.model.train_classifier \
        --candidates  output/candidate_pairs_train.tsv \
        --s1-tsv      dataset/train/source1.tsv \
        --s2-tsv      dataset/train/source2.tsv \
        --s3-tsv      dataset/train/source3.tsv \
        --s1-emb      .cache/embeddings/s1_train_embeddings.npy \
        --s2-emb      .cache/embeddings/s2_train_embeddings.npy \
        --s3-emb      .cache/embeddings/s3_train_embeddings.npy \
        --gt           dataset/train/train_ground_truth.tsv \
        --model-dir    .cache/model \
        --backend      xgboost   # or lightgbm
"""

from __future__ import annotations
import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURE_COLS: list[str] = []   # populated at import time from pairwise_features


def _label_pairs(pairs_df: pd.DataFrame, gt_df: pd.DataFrame) -> np.ndarray:
    """Return binary label array (1=match) aligned with pairs_df."""
    # Build set of true (s1_id, cand_id) pairs
    true_pairs: set[tuple[str, str]] = set()
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        for col in ("source2_entity_id", "source3_entity_id"):
            cid = str(row.get(col, "")).strip()
            if cid:
                true_pairs.add((s1, cid))

    labels = np.array(
        [int((r["source1_entity_id"], r["candidate_entity_id"]) in true_pairs)
         for _, r in pairs_df.iterrows()],
        dtype=np.int32,
    )
    return labels


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    backend: str = "xgboost",
    val_frac: float = 0.15,
    random_state: int = 42,
) -> object:
    """Train and return a fitted classifier."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_frac, stratify=y, random_state=random_state
    )
    pos   = y_tr.sum()
    neg   = len(y_tr) - pos
    scale = neg / max(pos, 1)
    print(f"[train] pos={pos}, neg={neg}, scale_pos_weight={scale:.2f}")

    if backend == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    elif backend == "lightgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    y_pred = clf.predict(X_val)
    print("[train] Validation classification report:")
    print(classification_report(y_val, y_pred, target_names=["no-match", "match"]))
    return clf


def save_model(clf, feature_names: list[str], model_dir: str | pathlib.Path,
               backend: str) -> None:
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"classifier_{backend}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "features": feature_names, "backend": backend}, f)
    print(f"[train] Model saved to {model_path}")


def load_model(model_dir: str | pathlib.Path, backend: str) -> dict:
    model_path = pathlib.Path(model_dir) / f"classifier_{backend}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source, load_ground_truth
    from src.preprocessing.normalize_names import normalize_name_series
    from src.preprocessing.normalize_addresses import normalize_address_series
    from src.features.pairwise_features import compute_features, TfidfPairScorer, FEATURE_COLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--s1-tsv",      required=True)
    parser.add_argument("--s2-tsv",      required=True)
    parser.add_argument("--s3-tsv",      required=True)
    parser.add_argument("--s1-emb",      required=True)
    parser.add_argument("--s2-emb",      required=True)
    parser.add_argument("--s3-emb",      required=True)
    parser.add_argument("--gt",          required=True)
    parser.add_argument("--model-dir",   default=".cache/model")
    parser.add_argument("--backend",     default="xgboost", choices=["xgboost", "lightgbm"])
    args = parser.parse_args()

    pairs_df = pd.read_csv(args.candidates, sep="\t", dtype=str, keep_default_na=False)
    gt_df    = load_ground_truth(args.gt)

    for df, path in [(None, args.s1_tsv), (None, args.s2_tsv), (None, args.s3_tsv)]:
        pass  # just to be explicit below

    s1_df = load_source(args.s1_tsv)
    s2_df = load_source(args.s2_tsv)
    s3_df = load_source(args.s3_tsv)
    for df in (s1_df, s2_df, s3_df):
        df["norm_name"]    = normalize_name_series(df["business_name"])
        df["norm_address"] = normalize_address_series(df["business_address"])

    emb_s1 = np.load(args.s1_emb)
    emb_s2 = np.load(args.s2_emb)
    emb_s3 = np.load(args.s3_emb)

    # Fit TF-IDF scorers on the train corpus
    all_names = list(s1_df["norm_name"]) + list(s2_df["norm_name"]) + list(s3_df["norm_name"])
    all_addrs = list(s1_df["norm_address"]) + list(s2_df["norm_address"]) + list(s3_df["norm_address"])
    name_tfidf = TfidfPairScorer().fit(all_names)
    addr_tfidf = TfidfPairScorer().fit(all_addrs)

    print("[train] Computing pairwise features…")
    feat_df = compute_features(pairs_df, s1_df, s2_df, s3_df,
                               emb_s1, emb_s2, emb_s3,
                               name_tfidf=name_tfidf, addr_tfidf=addr_tfidf)
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y = _label_pairs(pairs_df, gt_df)
    print(f"[train] Dataset: {len(X)} pairs, {y.sum()} positives ({100*y.mean():.2f}%)")

    clf = train(X, y, FEATURE_COLS, backend=args.backend)
    save_model(clf, FEATURE_COLS, args.model_dir, backend=args.backend)

    # Save TF-IDF vectorizers alongside the model
    import pickle
    mdir = pathlib.Path(args.model_dir)
    with open(mdir / "tfidf_name.pkl", "wb") as f:
        pickle.dump(name_tfidf, f)
    with open(mdir / "tfidf_addr.pkl", "wb") as f:
        pickle.dump(addr_tfidf, f)
    print("[train] Done.")


if __name__ == "__main__":
    _main()
