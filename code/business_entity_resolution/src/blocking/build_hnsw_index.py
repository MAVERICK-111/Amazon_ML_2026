"""
build_hnsw_index.py — Build and persist an HNSW ANN index over S2+S3 embeddings.

We index S2 and S3 together, tagging each vector with its source ('s2' or 's3')
so that retrieve step can map integer HNSW IDs back to entity_ids.

Usage:
    python -m src.blocking.build_hnsw_index \
        --s2-emb   .cache/embeddings/s2_embeddings.npy \
        --s3-emb   .cache/embeddings/s3_embeddings.npy \
        --s2-ids   dataset/train/source2.tsv \
        --s3-ids   dataset/train/source3.tsv \
        --index-dir .cache/index \
        --tag train
"""

from __future__ import annotations
import argparse
import pathlib
import pickle
import sys

import numpy as np
import hnswlib

# HNSW hyper-parameters (tune if needed)
HNSW_SPACE  = "cosine"    # embeddings are already L2-normalised → cosine = 1-dot
HNSW_M      = 32          # number of bi-directional links per node (higher = better recall, more RAM)
HNSW_EF_CONSTRUCTION = 200  # build-time candidate list size


def build_index(
    emb_s2: np.ndarray,
    emb_s3: np.ndarray,
    ids_s2: list[str],
    ids_s3: list[str],
    index_dir: str | pathlib.Path,
    tag: str = "train",
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
) -> tuple[hnswlib.Index, list[str], list[str]]:
    """Build HNSW index over concatenated S2+S3 embeddings.

    Returns:
        index: fitted hnswlib.Index
        combined_ids: list of entity_ids in HNSW integer-label order
        combined_sources: list of 's2' or 's3' in the same order
    """
    index_dir = pathlib.Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    assert emb_s2.shape[0] == len(ids_s2), "S2 embeddings/ids length mismatch"
    assert emb_s3.shape[0] == len(ids_s3), "S3 embeddings/ids length mismatch"

    emb_all = np.vstack([emb_s2, emb_s3]).astype(np.float32)
    combined_ids     = list(ids_s2) + list(ids_s3)
    combined_sources = ["s2"] * len(ids_s2) + ["s3"] * len(ids_s3)
    n, dim = emb_all.shape

    print(f"[build_hnsw] Building index: {n} vectors, dim={dim}, space={HNSW_SPACE}")
    idx = hnswlib.Index(space=HNSW_SPACE, dim=dim)
    idx.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    idx.set_num_threads(4)
    idx.add_items(emb_all, list(range(n)))

    # Persist
    idx_path  = index_dir / f"hnsw_{tag}.bin"
    meta_path = index_dir / f"hnsw_{tag}_meta.pkl"
    idx.save_index(str(idx_path))
    with open(meta_path, "wb") as f:
        pickle.dump(
            {"combined_ids": combined_ids, "combined_sources": combined_sources, "dim": dim},
            f,
        )
    print(f"[build_hnsw] Index saved to {idx_path}")
    print(f"[build_hnsw] Metadata saved to {meta_path}")
    return idx, combined_ids, combined_sources


def load_index(
    index_dir: str | pathlib.Path,
    tag: str = "train",
    ef_query: int = 100,
) -> tuple[hnswlib.Index, list[str], list[str]]:
    """Load a previously built HNSW index from disk."""
    index_dir = pathlib.Path(index_dir)
    idx_path  = index_dir / f"hnsw_{tag}.bin"
    meta_path = index_dir / f"hnsw_{tag}_meta.pkl"

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    dim = meta["dim"]
    idx = hnswlib.Index(space=HNSW_SPACE, dim=dim)
    idx.load_index(str(idx_path), max_elements=len(meta["combined_ids"]))
    idx.set_ef(ef_query)
    return idx, meta["combined_ids"], meta["combined_sources"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import pandas as pd
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source

    parser = argparse.ArgumentParser(description="Build HNSW index over S2+S3.")
    parser.add_argument("--s2-emb",   required=True)
    parser.add_argument("--s3-emb",   required=True)
    parser.add_argument("--s2-ids",   required=True, help="S2 source TSV (to read entity_ids)")
    parser.add_argument("--s3-ids",   required=True, help="S3 source TSV (to read entity_ids)")
    parser.add_argument("--index-dir", default=".cache/index")
    parser.add_argument("--tag",       default="train")
    args = parser.parse_args()

    emb_s2 = np.load(args.s2_emb)
    emb_s3 = np.load(args.s3_emb)
    ids_s2 = list(load_source(args.s2_ids)["entity_id"])
    ids_s3 = list(load_source(args.s3_ids)["entity_id"])

    build_index(emb_s2, emb_s3, ids_s2, ids_s3,
                index_dir=args.index_dir, tag=args.tag)
    print("[build_hnsw] Done.")


if __name__ == "__main__":
    _main()
