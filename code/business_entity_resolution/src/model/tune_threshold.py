"""
tune_threshold.py — Sweep decision thresholds on a validation split;
find the one that maximises macro F_0.5 (precision-biased).

Usage:
    python -m src.model.tune_threshold \
        --candidates  output/candidate_pairs_val.tsv \
        --features    output/features_val.npy \
        --gt          dataset/train/train_ground_truth.tsv \
        --s1-ids      dataset/train/source1.tsv \
        --model-dir   .cache/model \
        --backend     xgboost
"""

from __future__ import annotations
import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd


def sweep_thresholds(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    gt_df: pd.DataFrame,
    all_s1_ids: list[str],
    n_steps: int = 200,
) -> tuple[float, float]:
    """Sweep threshold from 0.01 to 0.99; return (best_threshold, best_f05)."""
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.evaluate.score_f05 import score, build_gt_sets

    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f = 0.5, 0.0

    for t in thresholds:
        preds = _build_pred_df(pairs_df, proba, t, all_s1_ids)
        result = score(preds, gt_df, all_s1_ids)
        f = result["macro_f05"]
        if f > best_f:
            best_f = f
            best_t = t

    return best_t, best_f


def _build_pred_df(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    all_s1_ids: list[str],
) -> pd.DataFrame:
    """Convert probability array + threshold into a matching_results-style DataFrame."""
    pairs_df = pairs_df.copy()
    pairs_df["prob"] = proba
    matched = pairs_df[pairs_df["prob"] >= threshold]

    # Group per S1 entity
    grouped = (
        matched.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: "|".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    # Every S1 entity must appear (singletons get empty string)
    all_s1 = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = all_s1.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source, load_ground_truth
    from src.model.train_classifier import load_model
    from src.features.pairwise_features import FEATURE_COLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--features",    required=True, help=".npy feature matrix (val split)")
    parser.add_argument("--gt",          required=True)
    parser.add_argument("--s1-ids",      required=True)
    parser.add_argument("--model-dir",   default=".cache/model")
    parser.add_argument("--backend",     default="xgboost")
    parser.add_argument("--n-steps",     type=int, default=200)
    args = parser.parse_args()

    pairs_df   = pd.read_csv(args.candidates, sep="\t", dtype=str, keep_default_na=False)
    X_val      = np.load(args.features)
    gt_df      = load_ground_truth(args.gt)
    all_s1_ids = list(load_source(args.s1_ids)["entity_id"])

    model_bundle = load_model(args.model_dir, args.backend)
    clf          = model_bundle["clf"]
    proba        = clf.predict_proba(X_val)[:, 1]

    best_t, best_f = sweep_thresholds(pairs_df, proba, gt_df, all_s1_ids,
                                      n_steps=args.n_steps)
    print(f"[tune_threshold] Best threshold = {best_t:.4f}  →  macro F_0.5 = {best_f:.6f}")

    # Save threshold alongside model
    threshold_path = pathlib.Path(args.model_dir) / "best_threshold.txt"
    threshold_path.write_text(str(best_t))
    print(f"[tune_threshold] Threshold saved to {threshold_path}")


if __name__ == "__main__":
    _main()
