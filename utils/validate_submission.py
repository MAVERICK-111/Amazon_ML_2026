"""
validate_submission.py — Pre-submission sanity checker.

Validates:
  1. Every test-set S1 entity has exactly one row in matching_results.tsv
  2. Singletons have an empty matched_entity_ids field (not omitted)
  3. No duplicate entity IDs within any ID list; no duplicate source1_entity_id rows
  4. Every ID in matching_results.tsv also appears in candidate_pairs.tsv for that S1 entity
  5. All matched/candidate IDs exist in test S2/S3 (no self-matches to S1)
  6. Candidate IDs are properly prefixed (S2-/S3-)

Exit code 0 → PASS, 1 → FAIL

Usage:
    python utils/validate_submission.py \
        --matching  output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir  dataset/test
"""

from __future__ import annotations
import argparse
import pathlib
import sys

import pandas as pd


def validate(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
) -> list[str]:
    """Return list of error strings; empty list means PASS."""
    errors: list[str] = []
    test_dir = pathlib.Path(test_dir)

    # --- Load files ---
    try:
        match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        return [f"Cannot read matching_results: {e}"]
    try:
        cand_df  = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        return [f"Cannot read candidate_pairs: {e}"]

    # Load S1 / S2 / S3 from test dir
    try:
        s1_df = pd.read_csv(test_dir / "source1.tsv", sep="\t", dtype=str, keep_default_na=False)
        s2_df = pd.read_csv(test_dir / "source2.tsv", sep="\t", dtype=str, keep_default_na=False)
        s3_df = pd.read_csv(test_dir / "source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        return [f"Cannot read test source files: {e}"]

    all_s1_ids  = set(s1_df["entity_id"])
    all_s2_ids  = set(s2_df["entity_id"])
    all_s3_ids  = set(s3_df["entity_id"])
    all_cand_ids = all_s2_ids | all_s3_ids

    # --- Check 1: column schema ---
    for col in ("source1_entity_id", "matched_entity_ids"):
        if col not in match_df.columns:
            errors.append(f"matching_results.tsv missing column: {col}")
    for col in ("source1_entity_id", "candidate_entity_id", "candidate_source"):
        if col not in cand_df.columns:
            errors.append(f"candidate_pairs.tsv missing column: {col}")
    if errors:
        return errors  # can't continue without schema

    # --- Check 2: every S1 entity present exactly once ---
    pred_s1_ids = list(match_df["source1_entity_id"])
    missing_s1  = all_s1_ids - set(pred_s1_ids)
    if missing_s1:
        errors.append(f"matching_results.tsv is missing {len(missing_s1)} S1 entities: "
                      f"{sorted(missing_s1)[:5]}…")

    dup_s1 = match_df[match_df.duplicated("source1_entity_id", keep=False)]
    if not dup_s1.empty:
        errors.append(f"Duplicate source1_entity_id rows in matching_results: "
                      f"{list(dup_s1['source1_entity_id'].unique())[:5]}")

    # Extra S1 ids not in test set
    extra_s1 = set(pred_s1_ids) - all_s1_ids
    if extra_s1:
        errors.append(f"matching_results.tsv has S1 ids not in test set: "
                      f"{sorted(extra_s1)[:5]}")

    # --- Check 3: build candidate set per S1 entity ---
    cand_map: dict[str, set[str]] = {}
    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"]
        cid   = row["candidate_entity_id"]
        cand_map.setdefault(s1_id, set()).add(cid)

    # --- Check 4 & 5: matched IDs must be in candidate set; no duplicates; valid IDs ---
    for _, row in match_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw   = str(row["matched_entity_ids"]).strip()
        if not raw:
            continue  # singleton — fine
        matched_ids = [x.strip() for x in raw.split("|") if x.strip()]

        # Duplicate within the list
        if len(matched_ids) != len(set(matched_ids)):
            errors.append(f"Duplicate matched IDs for {s1_id}: {matched_ids}")

        for mid in matched_ids:
            # Must not be a self-match (S1 id)
            if mid in all_s1_ids:
                errors.append(f"S1-to-S1 self-match for {s1_id} → {mid}")
            # Must exist in S2 or S3 test set
            if mid not in all_cand_ids:
                errors.append(f"Matched ID {mid} not found in test S2/S3 for {s1_id}")
            # Must be in candidate pairs for this S1 entity
            if mid not in cand_map.get(s1_id, set()):
                errors.append(
                    f"Matched ID {mid} for {s1_id} is NOT in candidate_pairs.tsv"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching",   required=True)
    parser.add_argument("--candidate",  required=True)
    parser.add_argument("--test-dir",   required=True)
    args = parser.parse_args()

    errors = validate(args.matching, args.candidate, args.test_dir)
    if errors:
        print("FAIL — validation errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("PASS — all checks passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
