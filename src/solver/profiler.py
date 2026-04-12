"""Generic data profiler — discovers and summarizes competition data dynamically."""

from __future__ import annotations

import logging
from pathlib import Path

from interpreter import Interpreter

logger = logging.getLogger(__name__)

# This script discovers all files, profiles CSVs, reads description.
# Nothing is hardcoded — it adapts to whatever data exists.
_PROFILER_SCRIPT = '''\
import os
import pandas as pd

data_dir = "./home/data"
lines = []
def p(s=""):
    lines.append(str(s))
    print(s)

p("=" * 60)
p("DATA PROFILE (auto-generated)")
p("=" * 60)

p("\\nFiles in data directory:")
for f in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, f)
    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        p(f"  {f} ({size_kb:.1f} KB)")

csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

for fname in csv_files:
    path = os.path.join(data_dir, fname)
    try:
        df = pd.read_csv(path, nrows=5000)
        full_len = sum(1 for _ in open(path)) - 1
        p(f"\\n--- {fname} (rows={full_len}, columns={len(df.columns)}) ---")
        p(f"Columns: {list(df.columns)}")
        p(f"Dtypes:")
        for col in df.columns:
            nulls = df[col].isnull().sum()
            null_str = f"  ({nulls} nulls)" if nulls > 0 else ""
            p(f"  {col}: {df[col].dtype}{null_str}")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols and len(num_cols) <= 30:
            p(f"\\nNumeric summary:")
            p(df[num_cols].describe().round(4).to_string())

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            p(f"\\nCategorical unique counts:")
            for col in cat_cols[:15]:
                uniq = df[col].nunique()
                examples = df[col].dropna().unique()[:5].tolist()
                p(f"  {col}: {uniq} unique, examples={examples}")

        p(f"\\nFirst 3 rows:")
        p(df.head(3).to_string())
    except Exception as e:
        p(f"\\n--- {fname} --- ERROR reading: {e}")

# Infer task type from sample submission
sample_files = [f for f in csv_files if "sample" in f.lower() and "submission" in f.lower()]
if sample_files:
    try:
        sample = pd.read_csv(os.path.join(data_dir, sample_files[0]))
        pred_cols = [c for c in sample.columns if c not in ("Id", "id", "ID")]
        if pred_cols:
            col = pred_cols[0]
            if sample[col].dtype in ("float64", "float32"):
                p(f"\\nInferred task: REGRESSION (target column: {col})")
            elif sample[col].nunique() <= 20:
                p(f"\\nInferred task: CLASSIFICATION (target: {col}, classes={sample[col].unique().tolist()})")
            else:
                p(f"\\nInferred task: PREDICTION (target: {col}, {sample[col].nunique()} unique values)")
    except Exception:
        pass

with open("_data_profile.txt", "w", encoding="utf-8") as f:
    f.write("\\n".join(lines))
'''


def run_profiler(workdir: Path, timeout: int = 60) -> str:
    """Run data profiler and return the profile text. Saves to _data_profile.txt."""
    profile_path = workdir / "_data_profile.txt"

    # Skip if already profiled
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8", errors="replace")

    interp = Interpreter(workdir=workdir, timeout=timeout)
    try:
        result = interp.run(_PROFILER_SCRIPT)
    finally:
        interp.cleanup()

    if profile_path.exists():
        text = profile_path.read_text(encoding="utf-8", errors="replace")
        logger.info("Data profile generated: %d chars", len(text))
        return text

    # Fallback: use stdout if file wasn't created
    logger.warning("Profiler did not create _data_profile.txt, using stdout")
    return result.stdout[:6000] if result.stdout else "<profiling failed>"
