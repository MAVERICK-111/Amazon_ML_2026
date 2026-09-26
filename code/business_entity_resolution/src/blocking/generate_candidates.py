"""
generate_candidates.py — Retrieve top-k nearest S2/S3 records for each S1 entity
and write candidate_pairs.tsv.

candidate_pairs.tsv schema:
    source1_entity_id <TAB> candidate_entity_id <TAB> candidate_source

Also computes (if ground-truth supplied) the recall ceiling — fraction of true
matches that appear in the candidate set.

Usage:
    python -m src.blocking.generate_candidates \
        --s1-emb     .cache/embeddings/s1_train_embeddings.npy \
        --s1-tsv     dataset/train/source1.tsv \
        --index-dir  .cache/index \
        --index-tag  train \
        --k          50 \
        --output     output/candidate_pairs_train.tsv \
        [--gt        dataset/train/train_ground_truth.tsv]
"""

from __future__ import annotations
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_K = 50       # top-k per S1 entity; recall ceiling is fixed here
DEFAULT_EF = 200     # HNSW query-time candidate list (≥ k for good recall)


def generate_candidates(
    emb_s1: np.ndarray,
    ids_s1: list[str],
    index,                        # hnswlib.Index (already loaded)
    combined_ids: list[str],
    combined_sources: list[str],
    k: int = DEFAULT_K,
    ef: int = DEFAULT_EF,
) -> pd.DataFrame:
    """Query each S1 embedding against the index; return candidate pairs DataFrame.

    Returns:
        DataFrame with columns:
            source1_entity_id, candidate_entity_id, candidate_source, distance
    """
    index.set_ef(max(ef, k))
    rows = []
    for i, s1_id in enumerate(tqdm(ids_s1, desc="[generate_candidates] Querying")):
        labels, distances = index.knn_query(emb_s1[i : i + 1], k=k)
        for label, dist in zip(labels[0], distances[0]):
            rows.append({
                "source1_entity_id":  s1_id,
                "candidate_entity_id": combined_ids[label],
                "candidate_source":   combined_sources[label],
                "distance":           float(dist),
            })
    return pd.DataFrame(rows)


def compute_recall_ceiling(
    candidates_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> float:
    """Fraction of true (S1, S2/S3) pairs that appear in candidates_df."""
    # Build set of (s1_id, candidate_id) from candidates
    cand_set = set(zip(candidates_df["source1_entity_id"],
                       candidates_df["candidate_entity_id"]))

    total = 0
    found = 0
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        for col in ("source2_entity_id", "source3_entity_id"):
            cid = row.get(col, "")
            if cid and cid.strip():
                total += 1
                if (s1, cid.strip()) in cand_set:
                    found += 1
    return found / total if total > 0 else 1.0


def write_candidates(df: pd.DataFrame, output_path: str | pathlib.Path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write only the columns required by the submission format
    out = df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    out.to_csv(str(output_path), sep="\t", index=False)
    print(f"[generate_candidates] Wrote {len(out)} pairs to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source, load_ground_truth
    from src.blocking.build_hnsw_index import load_index

    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-emb",    required=True)
    parser.add_argument("--s1-tsv",    required=True)
    parser.add_argument("--index-dir", default=".cache/index")
    parser.add_argument("--index-tag", default="train")
    parser.add_argument("--k",         type=int, default=DEFAULT_K)
    parser.add_argument("--ef",        type=int, default=DEFAULT_EF)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--gt",        default=None, help="Ground-truth TSV for recall check")
    args = parser.parse_args()

    emb_s1 = np.load(args.s1_emb)
    df_s1  = load_source(args.s1_tsv)
    ids_s1 = list(df_s1["entity_id"])

    idx, combined_ids, combined_sources = load_index(args.index_dir, tag=args.index_tag)

    cands = generate_candidates(emb_s1, ids_s1, idx, combined_ids, combined_sources,
                                k=args.k, ef=args.ef)
    write_candidates(cands, args.output)

    if args.gt:
        gt = load_ground_truth(args.gt)
        rc = compute_recall_ceiling(cands, gt)
        print(f"[generate_candidates] Recall ceiling @ k={args.k}: {rc:.4f}")

    print("[generate_candidates] Done.")


if __name__ == "__main__":
    _main()
