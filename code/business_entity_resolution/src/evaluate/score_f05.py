"""
score_f05.py — Local macro F_0.5 scorer matching the competition metric.

F_0.5 weights precision 2× over recall:
    F_0.5 = (1 + 0.5²) * P * R / (0.5² * P + R)
           = 1.25 * P * R / (0.25 * P + R)

Rules replicated from the problem statement:
  - Every S1 entity must have exactly one row in predictions.
  - Singletons (no true match) score 1.0 if predicted empty, 0.0 otherwise.
  - Macro average = mean of per-entity F_0.5 scores.

Usage:
    python -m src.evaluate.score_f05 \
        --pred    output/matching_results_val.tsv \
        --gt      dataset/train/train_ground_truth.tsv \
        --s1-ids  dataset/train/source1.tsv
"""

from __future__ import annotations
import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Dict, Set

import pandas as pd


def _f05(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def build_gt_sets(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Map each S1 entity_id → set of matching S2/S3 entity_ids."""
    gt: Dict[str, Set[str]] = defaultdict(set)
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        for col in ("source2_entity_id", "source3_entity_id"):
            cid = str(row.get(col, "")).strip()
            if cid:
                gt[s1].add(cid)
    return gt


def score(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    all_s1_ids: list[str],
) -> dict:
    """Compute macro F_0.5 over all S1 entities.

    pred_df columns: source1_entity_id, matched_entity_ids (pipe-delimited or empty)
    gt_df columns:   source1_entity_id, source2_entity_id, source3_entity_id
    all_s1_ids:      complete list of S1 entity_ids

    Returns dict with 'macro_f05', 'per_entity' (dict).
    """
    gt_sets = build_gt_sets(gt_df)

    # Parse predictions: source1_entity_id → set of matched ids
    pred_map: Dict[str, Set[str]] = {}
    for _, row in pred_df.iterrows():
        s1  = row["source1_entity_id"]
        raw = str(row.get("matched_entity_ids", "")).strip()
        ids = {x.strip() for x in raw.split("|") if x.strip()} if raw else set()
        pred_map[s1] = ids

    per_entity: Dict[str, float] = {}
    for s1_id in all_s1_ids:
        true_set = gt_sets.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())

        if not true_set:
            # Singleton: score 1.0 if prediction is empty
            per_entity[s1_id] = 1.0 if not pred_set else 0.0
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            per_entity[s1_id] = _f05(tp, fp, fn)

    macro = sum(per_entity.values()) / len(per_entity) if per_entity else 0.0
    return {"macro_f05": macro, "per_entity": per_entity}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source, load_ground_truth

    parser = argparse.ArgumentParser(description="Compute macro F_0.5.")
    parser.add_argument("--pred",   required=True, help="matching_results.tsv")
    parser.add_argument("--gt",     required=True, help="train_ground_truth.tsv")
    parser.add_argument("--s1-ids", required=True, help="source1.tsv (for full S1 list)")
    args = parser.parse_args()

    pred_df    = pd.read_csv(args.pred, sep="\t", dtype=str, keep_default_na=False)
    gt_df      = load_ground_truth(args.gt)
    all_s1_ids = list(load_source(args.s1_ids)["entity_id"])

    result = score(pred_df, gt_df, all_s1_ids)
    print(f"Macro F_0.5 = {result['macro_f05']:.6f}")
    n_singleton_correct = sum(
        1 for sid in all_s1_ids
        if not result["per_entity"].get(sid, 0.0) < 1.0
        and not any(sid in str(row) for _, row in gt_df.iterrows())
    )
    worst = sorted(result["per_entity"].items(), key=lambda x: x[1])[:10]
    print("Worst 10 entities:")
    for sid, sc in worst:
        print(f"  {sid}: {sc:.4f}")


if __name__ == "__main__":
    _main()
