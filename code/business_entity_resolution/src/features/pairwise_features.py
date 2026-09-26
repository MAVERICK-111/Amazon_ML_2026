"""
pairwise_features.py — Compute the feature vector for every (S1, candidate) pair.

Features computed per pair:
  Name features:
    name_lev_ratio          Levenshtein ratio (0-1) on normalized names
    name_token_sort_ratio   Token-sort Levenshtein ratio
    name_token_set_ratio    Token-set Levenshtein ratio
    name_jaccard            Jaccard over unigram token sets
    name_tfidf_cosine       TF-IDF cosine similarity (fitted on train corpus)
    name_emb_cosine         Embedding cosine similarity

  Address features:
    addr_lev_ratio
    addr_token_sort_ratio
    addr_token_set_ratio
    addr_jaccard
    addr_tfidf_cosine
    addr_emb_cosine

  Structural features:
    country_match           1 if country strings are identical, else 0
    name_len_delta          |len(norm_name_s1) - len(norm_name_cand)|
    addr_len_delta          |len(norm_addr_s1) - len(norm_addr_cand)|
    emb_distance            Raw HNSW cosine distance (already in candidates df)
"""

from __future__ import annotations
from typing import Sequence
import pathlib
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ---------------------------------------------------------------------------
# String similarity helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _lev_ratio(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0


def _token_sort_ratio(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(a, b) / 100.0


def _token_set_ratio(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a, b) / 100.0


# ---------------------------------------------------------------------------
# TF-IDF cosine (fitted once on training corpus)
# ---------------------------------------------------------------------------

class TfidfPairScorer:
    """Fit TF-IDF on a corpus then score pairs efficiently."""

    def __init__(self, analyzer: str = "word", ngram_range: tuple = (1, 2),
                 max_features: int = 50_000, sublinear_tf: bool = True):
        self.vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfPairScorer":
        self.vec.fit(texts)
        self._fitted = True
        return self

    def score_pairs(self, a_texts: Sequence[str], b_texts: Sequence[str]) -> np.ndarray:
        """Compute element-wise cosine similarity for (a_texts[i], b_texts[i]) pairs."""
        assert self._fitted, "Call fit() first."
        a_mat = self.vec.transform(a_texts)
        b_mat = self.vec.transform(b_texts)
        # Element-wise dot product of L2-normalised rows
        from sklearn.preprocessing import normalize
        a_norm = normalize(a_mat, norm="l2")
        b_norm = normalize(b_mat, norm="l2")
        sims = np.array(a_norm.multiply(b_norm).sum(axis=1)).flatten()
        return sims.astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding cosine (vectors are already unit-normalised → dot = cosine)
# ---------------------------------------------------------------------------

def _emb_cosine_for_pairs(
    emb_s1: np.ndarray,
    emb_cands: np.ndarray,
    s1_indices: np.ndarray,
    cand_indices: np.ndarray,
) -> np.ndarray:
    """Fast element-wise dot product."""
    a = emb_s1[s1_indices]      # (N, D)
    b = emb_cands[cand_indices] # (N, D)
    return (a * b).sum(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    pairs_df: pd.DataFrame,           # must contain source1_entity_id, candidate_entity_id, candidate_source
    s1_df: pd.DataFrame,              # s1 with norm_name, norm_address, country, entity_id
    s2_df: pd.DataFrame,              # s2 with norm_name, norm_address, country, entity_id
    s3_df: pd.DataFrame,              # s3 with norm_name, norm_address, country, entity_id
    emb_s1: np.ndarray,               # (|s1|, D) float32
    emb_s2: np.ndarray,               # (|s2|, D) float32
    emb_s3: np.ndarray,               # (|s3|, D) float32
    name_tfidf: TfidfPairScorer | None = None,
    addr_tfidf: TfidfPairScorer | None = None,
) -> pd.DataFrame:
    """Return features DataFrame aligned with pairs_df index."""

    # Build id → row-index maps for fast lookup
    s1_idx  = {eid: i for i, eid in enumerate(s1_df["entity_id"])}
    s2_idx  = {eid: i for i, eid in enumerate(s2_df["entity_id"])}
    s3_idx  = {eid: i for i, eid in enumerate(s3_df["entity_id"])}

    # Candidate id → source-specific index
    def cand_df_row(cand_id: str, cand_src: str) -> dict:
        if cand_src == "s2":
            row = s2_df.iloc[s2_idx[cand_id]]
        else:
            row = s3_df.iloc[s3_idx[cand_id]]
        return row

    # Pre-build combined candidate embedding lookup
    emb_combined = np.vstack([emb_s2, emb_s3])  # s2 first, then s3
    s2_offset = 0
    s3_offset = len(s2_df)

    def cand_emb_idx(cand_id: str, cand_src: str) -> int:
        if cand_src == "s2":
            return s2_offset + s2_idx[cand_id]
        return s3_offset + s3_idx[cand_id]

    records = []
    for _, pair in pairs_df.iterrows():
        s1_id   = pair["source1_entity_id"]
        c_id    = pair["candidate_entity_id"]
        c_src   = pair["candidate_source"]

        s1_row  = s1_df.iloc[s1_idx[s1_id]]
        c_row   = cand_df_row(c_id, c_src)

        nn_s1   = str(s1_row.get("norm_name", ""))
        nn_c    = str(c_row.get("norm_name", ""))
        na_s1   = str(s1_row.get("norm_address", ""))
        na_c    = str(c_row.get("norm_address", ""))
        country_s1 = str(s1_row.get("country", "")).lower().strip()
        country_c  = str(c_row.get("country", "")).lower().strip()

        # --- Embedding cosine ---
        s1_ei = s1_idx[s1_id]
        c_ei  = cand_emb_idx(c_id, c_src)
        e_cos = float(np.dot(emb_s1[s1_ei], emb_combined[c_ei]))

        feat = {
            # name
            "name_lev_ratio":       _lev_ratio(nn_s1, nn_c),
            "name_token_sort_ratio": _token_sort_ratio(nn_s1, nn_c),
            "name_token_set_ratio":  _token_set_ratio(nn_s1, nn_c),
            "name_jaccard":         _jaccard(nn_s1, nn_c),
            "name_emb_cosine":      e_cos,
            # address
            "addr_lev_ratio":        _lev_ratio(na_s1, na_c),
            "addr_token_sort_ratio": _token_sort_ratio(na_s1, na_c),
            "addr_token_set_ratio":  _token_set_ratio(na_s1, na_c),
            "addr_jaccard":          _jaccard(na_s1, na_c),
            "addr_emb_cosine":       e_cos,   # same embedding includes both name+addr
            # structural
            "country_match":  int(country_s1 == country_c and country_s1 != ""),
            "name_len_delta": abs(len(nn_s1) - len(nn_c)),
            "addr_len_delta": abs(len(na_s1) - len(na_c)),
            "emb_distance":   float(pair.get("distance", 1 - e_cos)),
        }
        records.append(feat)

    feat_df = pd.DataFrame(records, index=pairs_df.index)

    # --- TF-IDF cosine (batch, fast) ---
    if name_tfidf is not None:
        feat_df["name_tfidf_cosine"] = name_tfidf.score_pairs(
            pairs_df["_nn_s1"].tolist() if "_nn_s1" in pairs_df.columns
            else [str(s1_df.iloc[s1_idx[i]].get("norm_name", "")) for i in pairs_df["source1_entity_id"]],
            [str(cand_df_row(r["candidate_entity_id"], r["candidate_source"]).get("norm_name", ""))
             for _, r in pairs_df.iterrows()],
        )
    else:
        feat_df["name_tfidf_cosine"] = 0.0

    if addr_tfidf is not None:
        feat_df["addr_tfidf_cosine"] = addr_tfidf.score_pairs(
            [str(s1_df.iloc[s1_idx[i]].get("norm_address", "")) for i in pairs_df["source1_entity_id"]],
            [str(cand_df_row(r["candidate_entity_id"], r["candidate_source"]).get("norm_address", ""))
             for _, r in pairs_df.iterrows()],
        )
    else:
        feat_df["addr_tfidf_cosine"] = 0.0

    return feat_df


FEATURE_COLS = [
    "name_lev_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaccard", "name_tfidf_cosine", "name_emb_cosine",
    "addr_lev_ratio", "addr_token_sort_ratio", "addr_token_set_ratio",
    "addr_jaccard", "addr_tfidf_cosine", "addr_emb_cosine",
    "country_match", "name_len_delta", "addr_len_delta", "emb_distance",
]
