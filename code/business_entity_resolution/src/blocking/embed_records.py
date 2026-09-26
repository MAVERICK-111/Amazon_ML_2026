"""
embed_records.py — Batch embed records using a sentence-transformers model
and cache embeddings to disk as numpy .npy files.

Chosen model: BAAI/bge-large-en-v1.5  (Apache-2.0, ~335M params — fast)
Fallback to sentence-transformers/all-mpnet-base-v2 (Apache-2.0, ~110M params)

NOTE ON THE ~8B PARAM LIMIT:
The plan allows up to 8B params.  A full 8B model (e.g. Mistral-7B) is
impractical to run as a sentence encoder without a GPU.  We default to
BAAI/bge-large-en-v1.5 (335M) which delivers near-state-of-the-art
retrieval quality at manageable cost.  To swap to a larger model, change
EMBED_MODEL_NAME at the top of this file — the rest of the pipeline is
model-agnostic.

Usage:
    python -m src.blocking.embed_records \
        --input dataset/train/source1.tsv \
        --source s1 \
        --cache-dir .cache/embeddings \
        --batch-size 256
"""

from __future__ import annotations
import argparse
import hashlib
import pathlib
import sys

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default model — change here to swap the encoder
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Apache-2.0, 335 M params

# ---------------------------------------------------------------------------
# Lazy import of sentence_transformers to avoid slow import at module level
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]


def _get_model(model_name: str = EMBED_MODEL_NAME) -> "SentenceTransformer":
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[embed_records] Loading model: {model_name}", flush=True)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------

def _record_to_text(name: str, address: str) -> str:
    """Concatenate normalized name + address into a single embedding string."""
    parts = [p for p in (name.strip(), address.strip()) if p]
    return " | ".join(parts)


def embed_dataframe(
    df: "pd.DataFrame",
    name_col: str = "norm_name",
    addr_col: str = "norm_address",
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 256,
) -> np.ndarray:
    """Return an (N, D) float32 ndarray of embeddings for all rows in df."""
    model = _get_model(model_name)
    texts = [
        _record_to_text(str(row[name_col]), str(row[addr_col]))
        for _, row in df.iterrows()
    ]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm → cosine sim = dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def _cache_path(cache_dir: pathlib.Path, source_tag: str) -> pathlib.Path:
    return cache_dir / f"{source_tag}_embeddings.npy"


def embed_and_cache(
    df: "pd.DataFrame",
    source_tag: str,
    cache_dir: str | pathlib.Path = ".cache/embeddings",
    name_col: str = "norm_name",
    addr_col: str = "norm_address",
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 256,
    force_recompute: bool = False,
) -> np.ndarray:
    """Embed df rows; load from cache if it exists, otherwise compute + save.

    Args:
        df: DataFrame with normalized columns.
        source_tag: Short tag used in the cache filename (e.g. 's1', 's2_test').
        cache_dir: Directory to store .npy files.
        force_recompute: If True, ignore existing cache and recompute.

    Returns:
        float32 ndarray of shape (len(df), embedding_dim).
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = _cache_path(cache_dir, source_tag)

    if fpath.exists() and not force_recompute:
        print(f"[embed_records] Loading cached embeddings from {fpath}", flush=True)
        emb = np.load(str(fpath))
        if emb.shape[0] != len(df):
            print(
                f"[embed_records] WARNING: cache size {emb.shape[0]} != df size {len(df)}. "
                "Recomputing.",
                flush=True,
            )
        else:
            return emb

    print(f"[embed_records] Computing embeddings for {len(df)} records ({source_tag})…")
    emb = embed_dataframe(df, name_col=name_col, addr_col=addr_col,
                          model_name=model_name, batch_size=batch_size)
    np.save(str(fpath), emb)
    print(f"[embed_records] Saved to {fpath}", flush=True)
    return emb


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import pandas as pd
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    from src.preprocessing.load_data import load_source
    from src.preprocessing.normalize_names import normalize_name_series
    from src.preprocessing.normalize_addresses import normalize_address_series

    parser = argparse.ArgumentParser(description="Embed + cache source records.")
    parser.add_argument("--input", required=True, help="Path to source TSV.")
    parser.add_argument("--source", required=True, help="Short tag, e.g. s1, s2_train.")
    parser.add_argument("--cache-dir", default=".cache/embeddings")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default=EMBED_MODEL_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = load_source(args.input)
    df["norm_name"]    = normalize_name_series(df["business_name"])
    df["norm_address"] = normalize_address_series(df["business_address"])

    embed_and_cache(
        df,
        source_tag=args.source,
        cache_dir=args.cache_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        force_recompute=args.force,
    )
    print("[embed_records] Done.")


if __name__ == "__main__":
    _main()
