"""
data_cleaning.py
-----------------
Reusable data-cleaning utilities for the Breast Cancer Wisconsin
(Diagnostic) dataset project.

These functions are intentionally small and single-purpose so they can be
imported into the notebook (or any other script) and unit-tested easily.
Nothing here invents or alters real values -- it only standardizes column
names, corrects dtypes, checks for duplicates/missing values, and detects
(but does not blindly delete) statistical outliers.
"""

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sklearn's 'mean radius' style names to snake_case
    ('mean_radius') so columns are easier to reference in code."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the diagnosis label to a pandas category dtype instead of
    a free-text object column -- saves memory and makes the intended
    categorical nature of the column explicit."""
    df = df.copy()
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype("category")
    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small table of missing-value counts and percentages per
    column (only columns with at least one missing value are included)."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def find_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return every row involved in an exact duplicate (keep=False shows
    all copies, not just the second occurrence)."""
    return df[df.duplicated(keep=False)]


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Boolean mask flagging values outside 1.5 * IQR from Q1/Q3 for a
    single numeric column (the standard Tukey rule)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def outlier_summary(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    """Count how many IQR-flagged outliers exist in each numeric column."""
    counts = {col: int(detect_outliers_iqr(df[col]).sum()) for col in numeric_cols}
    out = pd.DataFrame.from_dict(counts, orient="index", columns=["outlier_count"])
    out["outlier_pct"] = (out["outlier_count"] / len(df) * 100).round(2)
    return out.sort_values("outlier_count", ascending=False)


def find_redundant_features(df: pd.DataFrame, numeric_cols, threshold: float = 0.95):
    """Return pairs of numeric columns whose absolute Pearson correlation
    exceeds `threshold`. Used to justify dropping duplicated/redundant
    measurements (e.g. radius vs. perimeter vs. area are geometrically
    linked) rather than removing columns arbitrarily."""
    corr = df[numeric_cols].corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > threshold:
                pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))
    return sorted(pairs, key=lambda x: -x[2])
