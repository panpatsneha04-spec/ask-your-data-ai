"""Reusable data analysis engine.

Every function here:
* takes a pandas DataFrame plus simple arguments (column names, numbers, text),
* validates those arguments and raises ``AnalysisError`` with a user-friendly
  message when something is wrong,
* returns a NEW DataFrame/result and never changes the DataFrame passed in.

There is no Streamlit code here, so the same functions can later be called by
an AI agent or a database loader, and they are easy to unit test.

Safety: user input is never executed as code. Filters use a fixed list of
allowed operators (see ``OPERATORS``) instead of ``eval`` or ``DataFrame.query``.
"""

from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from askyourdata import profiling


class AnalysisError(ValueError):
    """Raised for an invalid analysis request. The message is shown to users."""


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------

NUMERIC = "numeric"
DATETIME = "datetime"
CATEGORICAL = "categorical"
BOOLEAN = "boolean"

MISSING_LABEL = "(missing)"

# A text column counts as a date column when at least this share of a sample
# of its values can be read as dates.
DATE_DETECTION_THRESHOLD = 0.8
DATE_DETECTION_SAMPLE = 500


def is_numeric(series: pd.Series) -> bool:
    """True for number columns. True/False columns are not treated as numbers."""
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def _parse_dates(values: pd.Series) -> pd.Series:
    """Convert values to datetimes; anything unreadable becomes NaT (missing)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except (ValueError, TypeError, OverflowError):
            # e.g. a mix of time zones
            return pd.Series(pd.NaT, index=values.index)


def _looks_like_dates(series: pd.Series) -> bool:
    sample = series.dropna().head(DATE_DETECTION_SAMPLE).astype(str)
    if sample.empty:
        return False
    # Numbers stored as text ("1001") are IDs or amounts, not dates.
    if pd.to_numeric(sample, errors="coerce").notna().mean() >= DATE_DETECTION_THRESHOLD:
        return False
    return _parse_dates(sample).notna().mean() >= DATE_DETECTION_THRESHOLD


def column_kind(series: pd.Series) -> str:
    """Classify a column as numeric, datetime, boolean or categorical."""
    if ptypes.is_bool_dtype(series):
        return BOOLEAN
    if ptypes.is_datetime64_any_dtype(series):
        return DATETIME
    if is_numeric(series):
        return NUMERIC
    if _looks_like_dates(series):
        return DATETIME
    return CATEGORICAL


def column_kinds(df: pd.DataFrame) -> dict[Any, str]:
    return {column: column_kind(df[column]) for column in df.columns}


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c, kind in column_kinds(df).items() if kind == NUMERIC]


def date_columns(df: pd.DataFrame) -> list:
    return [c for c, kind in column_kinds(df).items() if kind == DATETIME]


def categorical_columns(df: pd.DataFrame) -> list:
    """Columns suitable for grouping: text, category and True/False columns."""
    return [c for c, kind in column_kinds(df).items() if kind in (CATEGORICAL, BOOLEAN)]


def to_datetime_series(series: pd.Series) -> pd.Series:
    """Return the column as real datetimes, or raise if it is not a date column."""
    if ptypes.is_datetime64_any_dtype(series):
        return series
    if column_kind(series) != DATETIME:
        raise AnalysisError(f"Column '{series.name}' does not contain dates.")
    return _parse_dates(series)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_dataframe(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise AnalysisError("No dataset was provided.")


def _require_rows(df: Any) -> None:
    _require_dataframe(df)
    if df.empty:
        raise AnalysisError("The dataset has no rows, so there is nothing to analyse.")


def validate_columns(df: pd.DataFrame, columns: Any) -> list:
    """Check that every requested column exists and return them as a list."""
    _require_dataframe(df)
    if columns is None or isinstance(columns, (str, int)):
        columns = [] if columns is None else [columns]
    columns = list(columns)
    if not columns:
        raise AnalysisError("Please choose at least one column.")
    if len(set(columns)) != len(columns):
        raise AnalysisError("The same column was chosen more than once.")
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise AnalysisError(
            f"Unknown column(s): {', '.join(map(str, unknown))}. "
            f"Available columns: {', '.join(map(str, df.columns))}."
        )
    return columns


def _require_numeric(df: pd.DataFrame, column: Any) -> None:
    if not is_numeric(df[column]):
        raise AnalysisError(
            f"Column '{column}' is {column_kind(df[column])}, but this needs a numeric column."
        )


def _validate_choice(value: Any, allowed: Sequence[str], what: str) -> str:
    if value not in allowed:
        raise AnalysisError(f"Unknown {what} '{value}'. Choose one of: {', '.join(allowed)}.")
    return value


def _validate_positive_int(value: Any, what: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AnalysisError(f"{what} must be a whole number (got {value!r}).")
    if value < 1:
        raise AnalysisError(f"{what} must be at least 1 (got {value}).")
    if maximum is not None and value > maximum:
        raise AnalysisError(f"{what} can be at most {maximum} (got {value}).")
    return int(value)


def _group_key(series: pd.Series) -> pd.Series:
    """Copy of a column for grouping, with missing values labelled '(missing)'."""
    return series.astype("object").where(series.notna(), MISSING_LABEL).rename(series.name)


# ---------------------------------------------------------------------------
# Profiling and descriptive statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetProfile:
    overview: profiling.DatasetOverview
    columns: pd.DataFrame  # one row per column, including its detected kind
    missing: pd.DataFrame
    numeric_stats: pd.DataFrame
    categorical_stats: pd.DataFrame


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Everything the app needs to describe a dataset, in one call."""
    _require_dataframe(df)
    columns = profiling.column_info(df)
    columns.insert(2, "kind", [column_kind(df[c]) for c in df.columns])
    return DatasetProfile(
        overview=profiling.dataset_overview(df),
        columns=columns,
        missing=profiling.missing_values(df),
        numeric_stats=profiling.numeric_summary(df),
        categorical_stats=profiling.categorical_summary(df),
    )


def describe_numeric(df: pd.DataFrame, columns: Sequence | None = None) -> pd.DataFrame:
    """count, missing, mean, median, std, min and max for numeric columns.

    Missing values are ignored in the calculations and reported separately.
    """
    _require_rows(df)
    if columns is None:
        columns = numeric_columns(df)
        if not columns:
            raise AnalysisError("This dataset has no numeric columns.")
    columns = validate_columns(df, columns)
    for column in columns:
        _require_numeric(df, column)

    stats = df[columns].agg(["count", "mean", "median", "std", "min", "max"]).T
    stats.insert(1, "missing", df[columns].isna().sum())
    return stats


COLUMN_AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count", "nunique")


@dataclass(frozen=True)
class ColumnAggregate:
    column: Any
    agg: str
    value: Any
    used_values: int  # non-missing values the calculation used
    missing: int  # missing values that were ignored


def aggregate_column(df: pd.DataFrame, column: Any, agg: str) -> ColumnAggregate:
    """One number for one column: sum, mean, median, min, max, count or nunique.

    ``count`` counts non-missing values and ``nunique`` counts distinct
    non-missing values (both work on any column); the others need numbers.
    Missing values are ignored and reported in the result.
    """
    _require_rows(df)
    [column] = validate_columns(df, [column])
    _validate_choice(agg, COLUMN_AGGREGATIONS, "aggregation")
    if agg not in ("count", "nunique"):
        _require_numeric(df, column)

    values = df[column].dropna()
    if agg == "count":
        value = len(values)
    elif agg == "nunique":
        value = int(values.nunique())
    elif values.empty:
        raise AnalysisError(f"Column '{column}' has no values to calculate with (all missing).")
    else:
        value = values.agg(agg)
        value = value.item() if hasattr(value, "item") else value  # numpy -> plain Python

    return ColumnAggregate(
        column=column,
        agg=agg,
        value=value,
        used_values=len(values),
        missing=int(df[column].isna().sum()),
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

EQUALS = "equals"
NOT_EQUALS = "does not equal"
GREATER = "greater than"
GREATER_EQUAL = "greater than or equal to"
LESS = "less than"
LESS_EQUAL = "less than or equal to"
CONTAINS = "contains"
NOT_CONTAINS = "does not contain"
IS_ONE_OF = "is one of"
IS_MISSING = "is missing"
IS_NOT_MISSING = "is not missing"

OPERATORS = (
    EQUALS, NOT_EQUALS, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL,
    CONTAINS, NOT_CONTAINS, IS_ONE_OF, IS_MISSING, IS_NOT_MISSING,
)  # fmt: skip
VALUELESS_OPERATORS = (IS_MISSING, IS_NOT_MISSING)

_COMPARISONS = {
    EQUALS: operator.eq,
    NOT_EQUALS: operator.ne,
    GREATER: operator.gt,
    GREATER_EQUAL: operator.ge,
    LESS: operator.lt,
    LESS_EQUAL: operator.le,
}


@dataclass(frozen=True)
class Condition:
    """One filter rule, e.g. Condition("units", "greater than", 5)."""

    column: Any
    operator: str
    value: Any = None


def operators_for(series: pd.Series) -> list[str]:
    """The filter operators that make sense for this column's type."""
    kind = column_kind(series)
    ops = [EQUALS, NOT_EQUALS]
    if kind in (NUMERIC, DATETIME):
        ops += [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL]
    if kind == CATEGORICAL:
        ops += [CONTAINS, NOT_CONTAINS, IS_ONE_OF]
    return ops + [IS_MISSING, IS_NOT_MISSING]


def _to_number(value: Any, column: Any) -> float:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return value
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        raise AnalysisError(
            f"Column '{column}' contains numbers, so the value must be a number (got '{value}')."
        ) from None


def _to_timestamp(value: Any, column: Any) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value.strip() if isinstance(value, str) else value)
    except (ValueError, TypeError):
        timestamp = pd.NaT
    if pd.isna(timestamp):
        raise AnalysisError(
            f"Column '{column}' contains dates, so the value must be a date like 2026-01-31 "
            f"(got '{value}')."
        )
    return timestamp


def _to_bool(value: Any, column: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False
    raise AnalysisError(f"Column '{column}' is True/False, so the value must be true or false.")


def _normalise_text(series: pd.Series) -> pd.Series:
    """Text comparisons ignore upper/lower case and surrounding spaces."""
    return series.astype("string").str.strip().str.casefold()


def _condition_mask(df: pd.DataFrame, condition: Condition) -> pd.Series:
    """A True/False value per row saying whether it matches the condition."""
    [column] = validate_columns(df, [condition.column])
    series = df[column]
    op = _validate_choice(condition.operator, OPERATORS, "filter condition")
    if op not in operators_for(series):
        raise AnalysisError(
            f"'{op}' cannot be used on '{column}' ({column_kind(series)} column). "
            f"Allowed: {', '.join(operators_for(series))}."
        )

    if op == IS_MISSING:
        return series.isna()
    if op == IS_NOT_MISSING:
        return series.notna()

    value = condition.value
    if value is None or (isinstance(value, str) and not value.strip()):
        raise AnalysisError(f"Please enter a value for: '{column}' {op} ...")

    if op in (CONTAINS, NOT_CONTAINS):
        # regex=False: the text is matched literally, never interpreted as a pattern.
        found = series.astype("string").str.contains(
            str(value).strip(), case=False, regex=False
        ).fillna(False).astype(bool)
        mask = found if op == CONTAINS else ~found
    elif op == IS_ONE_OF:
        items = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
        wanted = {str(item).strip().casefold() for item in items if str(item).strip()}
        if not wanted:
            raise AnalysisError(f"Please enter at least one value for: '{column}' is one of ...")
        mask = _normalise_text(series).isin(wanted).fillna(False).astype(bool)
    else:
        kind = column_kind(series)
        if kind == NUMERIC:
            left, right = series, _to_number(value, column)
        elif kind == DATETIME:
            left, right = to_datetime_series(series), _to_timestamp(value, column)
        elif kind == BOOLEAN:
            left, right = series, _to_bool(value, column)
        else:
            left, right = _normalise_text(series), str(value).strip().casefold()
        try:
            mask = _COMPARISONS[op](left, right)
        except TypeError as exc:
            raise AnalysisError(f"Could not compare '{column}' with '{value}': {exc}") from None
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)

    # Rows where the column is missing never match a value-based condition.
    return mask & series.notna()


def filter_rows(
    df: pd.DataFrame, conditions: Sequence[Condition], match: str = "all"
) -> pd.DataFrame:
    """Return the rows that match the conditions.

    Args:
        conditions: Filter rules; with no rules every row is returned.
        match: "all" (every rule must match) or "any" (at least one rule).
    """
    _require_dataframe(df)
    _validate_choice(match, ("all", "any"), "match mode")
    if not conditions:
        return df.copy()

    masks = [_condition_mask(df, condition) for condition in conditions]
    combine = operator.and_ if match == "all" else operator.or_
    return df.loc[functools.reduce(combine, masks)].copy()


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def _sort_key(series: pd.Series) -> pd.Series:
    kind = column_kind(series)
    if kind == DATETIME and not ptypes.is_datetime64_any_dtype(series):
        return _parse_dates(series)  # sort text dates by date, not alphabetically
    if kind == CATEGORICAL and ptypes.infer_dtype(series, skipna=True).startswith("mixed"):
        return series.astype("string")  # e.g. numbers and text in one Excel column
    return series


def sort_rows(
    df: pd.DataFrame, by: Any, ascending: bool | Sequence[bool] = True
) -> pd.DataFrame:
    """Sort by one or more columns. Missing values always go last."""
    columns = validate_columns(df, by)
    if isinstance(ascending, (bool, np.bool_)):
        ascending = [bool(ascending)] * len(columns)
    ascending = list(ascending)
    if len(ascending) != len(columns):
        raise AnalysisError("Give one sort direction per sort column.")

    try:
        return df.sort_values(
            columns, ascending=ascending, na_position="last", kind="stable", key=_sort_key
        )
    except TypeError as exc:
        raise AnalysisError(f"These columns could not be sorted: {exc}") from None


# ---------------------------------------------------------------------------
# Grouping and top N
# ---------------------------------------------------------------------------

AGGREGATIONS = ("count", "sum", "mean", "median", "min", "max")
DEFAULT_GROUP_AGGREGATIONS = ("count", "sum", "mean", "min", "max")

# A numeric column with more distinct values than this is a measurement
# (price, age...) rather than a category, so grouping by it is refused.
MAX_NUMERIC_GROUP_VALUES = 20


def _measure_label(agg: str, value_column: Any) -> str:
    return f"{agg} of {value_column}"


def group_summary(
    df: pd.DataFrame,
    group_by: Any,
    value_column: Any = None,
    aggregations: Sequence[str] = DEFAULT_GROUP_AGGREGATIONS,
) -> pd.DataFrame:
    """Summarise a numeric column for each group of rows.

    The result always has a ``rows`` column (rows per group). When a numeric
    ``value_column`` is given it also gets one column per aggregation, e.g.
    "sum of units". Missing group values become their own "(missing)" group;
    missing numbers are ignored by sum/mean/min/max.
    """
    _require_rows(df)
    group_columns = validate_columns(df, group_by)
    for column in group_columns:
        if is_numeric(df[column]) and df[column].nunique() > MAX_NUMERIC_GROUP_VALUES:
            raise AnalysisError(
                f"Column '{column}' has {df[column].nunique()} different numbers, which is too "
                "many to group by. Choose a text or category column instead."
            )
    aggregations = list(aggregations)
    for agg in aggregations:
        _validate_choice(agg, AGGREGATIONS, "aggregation")

    keys = [_group_key(df[column]) for column in group_columns]
    grouped = df.groupby(keys, sort=False)
    result = grouped.size().rename("rows").to_frame()

    if value_column is not None:
        [value_column] = validate_columns(df, [value_column])
        if value_column in group_columns:
            raise AnalysisError("The value column cannot also be a grouping column.")
        _require_numeric(df, value_column)
        if aggregations:
            measures = grouped[value_column].agg(aggregations)
            measures.columns = [_measure_label(agg, value_column) for agg in aggregations]
            result = result.join(measures)

    return result.reset_index().sort_values("rows", ascending=False, kind="stable").reset_index(
        drop=True
    )


def top_n(
    df: pd.DataFrame,
    column: Any,
    n: int = 10,
    value_column: Any = None,
    agg: str = "sum",
    largest: bool = True,
) -> pd.DataFrame:
    """Top N categories of ``column``.

    Without ``value_column``: the most frequent values, with count and percent.
    With ``value_column``: categories ranked by e.g. the sum of that column
    ("top 5 products by total units").
    """
    _require_rows(df)
    [column] = validate_columns(df, [column])
    n = _validate_positive_int(n, "N")
    key = _group_key(df[column])

    if value_column is None:
        counts = key.value_counts(ascending=not largest)
        table = counts.head(n).rename_axis(column).reset_index(name="count")
        table["percent"] = (table["count"] / len(df) * 100).round(2)
        return table

    [value_column] = validate_columns(df, [value_column])
    if value_column == column:
        raise AnalysisError("Choose a different column to rank by.")
    _require_numeric(df, value_column)
    _validate_choice(agg, AGGREGATIONS, "aggregation")

    values = df[value_column].groupby(key, sort=False).agg(agg)
    values = values.sort_values(ascending=not largest, na_position="last", kind="stable")
    return values.head(n).rename_axis(column).reset_index(name=_measure_label(agg, value_column))


def top_rows(df: pd.DataFrame, column: Any, n: int = 10, largest: bool = True) -> pd.DataFrame:
    """The N rows with the largest (or smallest) values in a numeric column.

    Rows where that column is missing are skipped.
    """
    _require_rows(df)
    [column] = validate_columns(df, [column])
    _require_numeric(df, column)
    n = _validate_positive_int(n, "N")
    return df.nlargest(n, column) if largest else df.nsmallest(n, column)


# ---------------------------------------------------------------------------
# Date trends
# ---------------------------------------------------------------------------

TREND_FREQUENCIES = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}


@dataclass(frozen=True)
class TrendResult:
    table: pd.DataFrame  # columns: "period" (start date) and ``value_label``
    value_label: str
    skipped_rows: int  # rows left out because their date was missing or unreadable


def date_trend(
    df: pd.DataFrame,
    date_column: Any,
    value_column: Any = None,
    freq: str = "Month",
    agg: str = "sum",
) -> TrendResult:
    """Summarise data per day / week / month / quarter / year.

    Without ``value_column`` it counts rows per period. Periods with no data
    are included (with 0 for count/sum) so gaps are visible on a chart.
    """
    _require_rows(df)
    [date_column] = validate_columns(df, [date_column])
    code = TREND_FREQUENCIES[_validate_choice(freq, list(TREND_FREQUENCIES), "time period")]
    dates = to_datetime_series(df[date_column])

    valid = dates.notna()
    if not valid.any():
        raise AnalysisError(f"Column '{date_column}' has no readable dates.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # time zone information is dropped for periods
        periods = dates[valid].dt.to_period(code)

    if value_column is None:
        label = "rows"
        series = periods.value_counts()
        fill_empty = True
    else:
        [value_column] = validate_columns(df, [value_column])
        _require_numeric(df, value_column)
        _validate_choice(agg, AGGREGATIONS, "aggregation")
        label = _measure_label(agg, value_column)
        series = df.loc[valid, value_column].groupby(periods).agg(agg)
        fill_empty = agg in ("count", "sum")

    all_periods = pd.period_range(periods.min(), periods.max(), freq=code)
    series = series.reindex(all_periods)
    if fill_empty:
        series = series.fillna(0)

    table = pd.DataFrame({"period": all_periods.to_timestamp(), label: series.to_numpy()})
    return TrendResult(table=table, value_label=label, skipped_rows=int((~valid).sum()))


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

CORRELATION_METHODS = ("pearson", "spearman")


def correlation_matrix(
    df: pd.DataFrame, columns: Sequence | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Correlation (-1 to 1) between every pair of numeric columns.

    Each pair uses only the rows where both values are present; pairs with
    fewer than 3 such rows get no value (NaN).
    """
    _require_rows(df)
    _validate_choice(method, CORRELATION_METHODS, "correlation method")
    columns = numeric_columns(df) if columns is None else validate_columns(df, columns)
    for column in columns:
        _require_numeric(df, column)
    if len(columns) < 2:
        raise AnalysisError("Correlations need at least two numeric columns.")
    return df[columns].astype(float).corr(method=method, min_periods=3)


def correlation_strength(r: float) -> str:
    size = abs(r)
    if size >= 0.7:
        label = "strong"
    elif size >= 0.4:
        label = "moderate"
    elif size >= 0.2:
        label = "weak"
    else:
        return "very weak / none"
    return f"{label} {'positive' if r > 0 else 'negative'}"


def strongest_correlations(corr: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Column pairs from a correlation matrix, strongest relationship first."""
    top = _validate_positive_int(top, "Number of pairs")
    columns = list(corr.columns)
    pairs = [
        (a, b, corr.loc[a, b], correlation_strength(corr.loc[a, b]))
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if pd.notna(corr.loc[a, b])
    ]
    table = pd.DataFrame(pairs, columns=["column_a", "column_b", "correlation", "strength"])
    order = table["correlation"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart-ready data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartData:
    """A small table plus the column names a chart should use for its axes."""

    data: pd.DataFrame
    x: str
    y: str
    title: str
    note: str | None = None  # e.g. how many rows were left out


def bar_chart_data(
    df: pd.DataFrame,
    category_column: Any,
    value_column: Any = None,
    agg: str = "sum",
    top: int = 20,
) -> ChartData:
    """Top categories for a bar chart (row counts, or ``agg`` of a numeric column)."""
    table = top_n(df, category_column, n=top, value_column=value_column, agg=agg)
    y = "count" if value_column is None else _measure_label(agg, value_column)
    title = f"{y.capitalize()} by {category_column}"
    return ChartData(data=table, x=category_column, y=y, title=title)


def line_chart_data(
    df: pd.DataFrame,
    date_column: Any,
    value_column: Any = None,
    freq: str = "Month",
    agg: str = "sum",
) -> ChartData:
    """A date trend ready for a line chart."""
    trend = date_trend(df, date_column, value_column=value_column, freq=freq, agg=agg)
    note = None
    if trend.skipped_rows:
        note = f"{trend.skipped_rows} row(s) with a missing or unreadable date were left out."
    title = f"{trend.value_label.capitalize()} per {freq.lower()}"
    return ChartData(data=trend.table, x="period", y=trend.value_label, title=title, note=note)


def histogram_data(df: pd.DataFrame, column: Any, bins: int = 20) -> ChartData:
    """Counts of values per range ("bin") of a numeric column."""
    _require_rows(df)
    [column] = validate_columns(df, [column])
    _require_numeric(df, column)
    bins = _validate_positive_int(bins, "Number of bins", maximum=200)

    values = df[column].dropna().astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        raise AnalysisError(f"Column '{column}' has no numeric values to plot.")

    counts, edges = np.histogram(values, bins=bins)
    table = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": counts})
    table["bin_center"] = (table["bin_start"] + table["bin_end"]) / 2
    table["bin_range"] = [f"{s:,.2f} to {e:,.2f}" for s, e in zip(edges[:-1], edges[1:])]

    left_out = len(df) - len(values)
    note = f"{left_out} missing value(s) were left out." if left_out else None
    return ChartData(
        data=table, x="bin_center", y="count", title=f"Distribution of {column}", note=note
    )
