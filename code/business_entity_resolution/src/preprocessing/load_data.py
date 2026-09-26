"""
load_data.py — TSV loaders with sep='\t' and schema checks.

Expected columns:
  source1/2/3: entity_id, business_name, business_address, country
  train_ground_truth.tsv: source1_entity_id, source2_entity_id, source3_entity_id
"""

from __future__ import annotations
import pathlib
import pandas as pd

_SRC_SCHEMA = {"entity_id", "business_name", "business_address", "country"}
_GT_SCHEMA   = {"source1_entity_id", "source2_entity_id", "source3_entity_id"}


def _load_tsv(path: str | pathlib.Path, required_cols: set[str]) -> pd.DataFrame:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return df


def load_source(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a source TSV (S1, S2, or S3).  Returns DataFrame with _SRC_SCHEMA columns."""
    df = _load_tsv(path, _SRC_SCHEMA)
    df = df.dropna(subset=["entity_id"]).reset_index(drop=True)
    return df


def load_ground_truth(path: str | pathlib.Path) -> pd.DataFrame:
    """Load train_ground_truth.tsv.

    Returns long-form DataFrame with columns:
        source1_entity_id, source_id (the S2/S3 id), source_num (2 or 3)
    plus the raw source2_entity_id / source3_entity_id columns.
    """
    df = _load_tsv(path, _GT_SCHEMA)
    return df


def load_all_train(train_dir: str | pathlib.Path) -> dict[str, pd.DataFrame]:
    """
    Convenience: load everything under a train directory.
    Expects:
        {train_dir}/source1.tsv
        {train_dir}/source2.tsv
        {train_dir}/source3.tsv
        {train_dir}/train_ground_truth.tsv
    """
    d = pathlib.Path(train_dir)
    return {
        "s1": load_source(d / "source1.tsv"),
        "s2": load_source(d / "source2.tsv"),
        "s3": load_source(d / "source3.tsv"),
        "gt": load_ground_truth(d / "train_ground_truth.tsv"),
    }


def load_all_test(test_dir: str | pathlib.Path) -> dict[str, pd.DataFrame]:
    """
    Convenience: load everything under a test directory.
    Expects:
        {test_dir}/source1.tsv
        {test_dir}/source2.tsv
        {test_dir}/source3.tsv
    """
    d = pathlib.Path(test_dir)
    return {
        "s1": load_source(d / "source1.tsv"),
        "s2": load_source(d / "source2.tsv"),
        "s3": load_source(d / "source3.tsv"),
    }
