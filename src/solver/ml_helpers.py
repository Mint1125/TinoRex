"""Submission validation and column patching utilities for the tree search solver."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def validate_submission_report(submission_path: Path, workdir: Path) -> dict:
    """Validate submission.csv against sample submission and test.csv.

    Returns dict with keys: valid (bool), errors (list), warnings (list), summary (str).
    """
    errors: list[str] = []
    warnings: list[str] = []
    data_dir = workdir / "home" / "data"

    if not submission_path.is_file():
        return {
            "valid": False,
            "errors": ["submission.csv does not exist"],
            "warnings": [],
            "summary": "FAIL: submission.csv was not produced.",
        }

    try:
        sub = pd.read_csv(submission_path)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"Cannot read submission.csv: {exc}"],
            "warnings": [],
            "summary": f"FAIL: submission.csv is not a valid CSV: {exc}",
        }

    # Find sample submission
    sample_paths = sorted(glob.glob(str(data_dir / "sample_submission*.csv")))
    sample: pd.DataFrame | None = None
    if sample_paths:
        try:
            sample = pd.read_csv(sample_paths[0])
        except Exception as exc:
            warnings.append(f"Could not read sample submission: {exc}")

    if sample is not None:
        exp_cols = list(sample.columns)
        act_cols = list(sub.columns)

        missing = [c for c in exp_cols if c not in sub.columns]
        extra = [c for c in act_cols if c not in exp_cols]

        if missing:
            errors.append(f"Missing columns vs sample: {missing}")
        if extra:
            warnings.append(f"Extra columns vs sample: {extra}")
        if not missing and not extra and exp_cols != act_cols:
            warnings.append(f"Column order differs from sample. Expected {exp_cols}, got {act_cols}")

        if len(sub) != len(sample):
            errors.append(f"Row count {len(sub)} != sample row count {len(sample)}")
    else:
        warnings.append("No sample_submission found; skipped schema check.")

    # Check against test.csv row count
    test_paths = sorted(glob.glob(str(data_dir / "test.csv")))
    if test_paths:
        try:
            test_df = pd.read_csv(test_paths[0])
            if len(sub) != len(test_df):
                errors.append(f"Row count {len(sub)} != test.csv rows {len(test_df)}")
        except Exception:
            pass

    # Check NA values
    na_cols = sub.columns[sub.isna().any()].tolist()
    if na_cols:
        na_counts = {col: int(sub[col].isna().sum()) for col in na_cols}
        errors.append(f"NA values in columns: {na_counts}")

    valid = len(errors) == 0
    parts = []
    for e in errors:
        parts.append(f"FAIL: {e}")
    for w in warnings:
        parts.append(f"WARN: {w}")
    if valid and not warnings:
        parts.append("OK: submission passes all checks.")
    summary = "\n".join(parts)

    return {"valid": valid, "errors": errors, "warnings": warnings, "summary": summary}


def patch_submission_columns(submission_path: Path, workdir: Path) -> bool:
    """Attempt to fix submission.csv column/row issues. Returns True if successful."""
    try:
        data_dir = workdir / "home" / "data"
        sample_paths = sorted(glob.glob(str(data_dir / "sample_submission*.csv")))
        if not sample_paths:
            return True

        sample = pd.read_csv(sample_paths[0])
        sub = pd.read_csv(submission_path)

        changed = False

        # Fix missing columns
        missing_cols = [col for col in sample.columns if col not in sub.columns]
        if missing_cols:
            test_paths = sorted(glob.glob(str(data_dir / "test.csv")))
            test_df = pd.read_csv(test_paths[0]) if test_paths else None

            for col in missing_cols:
                if test_df is not None and col in test_df.columns:
                    if len(sub) == len(test_df):
                        sub[col] = test_df[col].values
                    else:
                        sub[col] = sample[col].values
                else:
                    sub[col] = sample[col].values
            changed = True
            logger.info("Patched missing columns: %s", missing_cols)

        # Drop extra columns
        extra_cols = [col for col in sub.columns if col not in sample.columns]
        if extra_cols:
            sub = sub.drop(columns=extra_cols)
            changed = True
            logger.info("Dropped extra columns: %s", extra_cols)

        # Reorder columns to match sample
        ordered_cols = [col for col in sample.columns if col in sub.columns]
        if list(sub.columns) != ordered_cols:
            sub = sub[ordered_cols]
            changed = True

        # Fill NA values
        na_cols = sub.columns[sub.isna().any()].tolist()
        if na_cols:
            for col in na_cols:
                if col in sample.columns:
                    fill_val = sample[col].mode().iloc[0] if not sample[col].mode().empty else 0
                    sub[col] = sub[col].fillna(fill_val)
                else:
                    sub[col] = sub[col].fillna(0)
            changed = True
            logger.info("Filled NA values in columns: %s", na_cols)

        if changed:
            sub.to_csv(submission_path, index=False)

        return True
    except Exception as exc:
        logger.warning("Column patching failed: %s", exc)
        return False
