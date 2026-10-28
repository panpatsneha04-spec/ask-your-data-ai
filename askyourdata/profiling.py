"""Summary statistics that describe the shape and quality of a dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetOverview:
    """Headline numbers shown at the top of the app."""

    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    memory_mb: float


def dataset_overview(df: pd.DataFrame) -> DatasetOverview:
    return DatasetOverview(
        rows=len(df),
        columns=df.shape[1],
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=duplicate_row_count(df),
        memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
    )


def duplicate_row_count(df: pd.DataFrame) -> int:
    """Count rows that are exact copies of an earlier row."""
    return int(df.duplicated().sum())


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: name, data type, and how complete / varied it is."""
    return pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": df.notna().sum().to_numpy(),
            "unique_values": df.nunique(dropna=True).to_numpy(),
        }
    )


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count and percentage per column, worst columns first."""
    missing = df.isna().sum()
    percent = (missing / len(df) * 100).round(2) if len(df) else missing * 0.0
    table = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "missing": missing.to_numpy(),
            "missing_pct": percent.to_numpy(),
        }
    )
    return table.sort_values("missing", ascending=False, kind="stable").reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """count / mean / std / min / quartiles / max for numeric columns.

    Returns an empty DataFrame when there are no numeric columns, because
    ``DataFrame.describe`` raises an error in that case.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """count / unique / most frequent value for text, category and bool columns."""
    categorical = df.select_dtypes(include=["object", "category", "bool", "string"])
    if categorical.empty:
        return pd.DataFrame()
    return categorical.describe().T
