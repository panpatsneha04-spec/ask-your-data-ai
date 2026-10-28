"""Executive summary of a dataset: KPIs, a trend, a comparison, data quality.

Everything shown is calculated from the data with pandas and the existing
analysis engine. A section is only included when the data supports it, and
``skipped`` says why a section is missing:
* KPIs: row/column counts, completeness, duplicates and averages of numeric
  columns. ID-like columns (order_id, ...) are never averaged.
* A calculated metric (net revenue, else gross amount) only when it can be
  calculated with no clarification and no column of that name exists. Its
  formula and assumptions are part of the summary.
* A trend only with exactly ONE usable date column covering 2+ periods.
* A comparison only for a text column with 2-20 different values.
* Data-quality notes are plain facts (missing values, duplicates, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from askyourdata import data_engine as de
from askyourdata import metrics, profiling
from askyourdata.data_engine import AnalysisError, ChartData
from askyourdata.text_matching import ClarificationNeeded, fmt, looks_like_id, words

MAX_AVERAGE_KPIS = 3
COMPARISON_MIN_GROUPS = 2
COMPARISON_MAX_GROUPS = 20
MAX_QUALITY_COLUMNS = 5
NUMBERS_AS_TEXT_SHARE = 0.8


@dataclass(frozen=True)
class KPI:
    label: str
    value: str
    help: str = ""


@dataclass(frozen=True)
class MetricInfo:
    label: str
    column: str  # name of the calculated column in ``data``
    formula: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Section:
    title: str
    text: str
    chart: ChartData
    kind: str  # "line" or "bar"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutiveSummary:
    headline: str
    kpis: tuple[KPI, ...]
    metric: MetricInfo | None = None
    trend: Section | None = None
    comparison: Section | None = None
    quality: tuple[str, ...] = ()  # data-quality issues (or a single 'all clear' note)
    info: tuple[str, ...] = ()  # neutral facts, e.g. which columns look like IDs
    skipped: tuple[str, ...] = ()  # sections not shown, and why
    data: pd.DataFrame | None = field(default=None, repr=False)  # copy incl. calculated metric


def _col(column: Any) -> str:
    return f"`{column}`"


def _count(n: int, noun: str) -> str:
    """"1 column", "9 columns"."""
    return f"{fmt(n)} {noun}{'' if n == 1 else 's'}"


def _pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "0%"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _metric(df: pd.DataFrame, skipped: list[str]) -> tuple[MetricInfo, pd.Series] | None:
    """Net revenue (else gross amount), only when it is unambiguous."""
    if any(metrics.match_metric(words(c)) for c in df.columns):
        return None  # the dataset has its own revenue/sales column; don't derive another
    for key in (metrics.NET_REVENUE, metrics.GROSS_AMOUNT):
        try:
            if metrics.missing_inputs(df, key):
                continue
            result = metrics.compute_metric(df, key)
        except (AnalysisError, ClarificationNeeded) as exc:
            label = metrics.METRICS[key].label.replace(" (calculated)", "").lower()
            skipped.append(f"{label.capitalize()} was not calculated: {exc}")
            continue
        info = MetricInfo(result.label, result.label, result.formula, result.notes)
        return info, result.values
    return None


def _average_kpis(df: pd.DataFrame, exclude: set[Any]) -> list[KPI]:
    kpis = []
    for column in de.numeric_columns(df):
        if column in exclude or looks_like_id(df, column):
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        missing = int(df[column].isna().sum())
        help_text = f"Median {fmt(values.median())}; {fmt(len(values))} values"
        help_text += f", {fmt(missing)} missing ignored." if missing else "."
        kpis.append(KPI(f"Average {column}", fmt(values.mean()), help_text))
        if len(kpis) == MAX_AVERAGE_KPIS:
            break
    return kpis


def _frequency(days: float) -> str:
    if days > 3 * 366:
        return "Year"
    if days > 90:
        return "Month"
    if days > 21:
        return "Week"
    return "Day"


def _period_label(ts: pd.Timestamp, freq: str) -> str:
    if freq == "Quarter":
        return f"Q{ts.quarter} {ts.year}"
    return ts.strftime({"Day": "%Y-%m-%d", "Week": "week of %Y-%m-%d", "Month": "%b %Y", "Year": "%Y"}[freq])


def _trend(data: pd.DataFrame, metric: MetricInfo | None, skipped: list[str]) -> Section | None:
    dates = de.date_columns(data)
    if not dates:
        skipped.append("Trend: no date column was found.")
        return None
    if len(dates) > 1:
        skipped.append(
            f"Trend: there are several date columns ({', '.join(map(_col, dates))}), so none is "
            "picked automatically. Ask e.g. 'trend by month using <column>' in Ask Your Data."
        )
        return None
    column = dates[0]
    parsed = de.to_datetime_series(data[column]).dropna()
    if parsed.nunique() < 2:
        skipped.append(f"Trend: {_col(column)} has fewer than two different dates.")
        return None
    freq = _frequency((parsed.max() - parsed.min()).days)
    value = metric.column if metric else None
    chart = de.line_chart_data(data, column, value, freq=freq, agg="sum")
    table = chart.data.dropna(subset=[chart.y])
    if len(chart.data) < 2:
        skipped.append(f"Trend: all dates in {_col(column)} fall within one {freq.lower()}.")
        return None

    what = metric.label if metric else "rows"
    peak = table.loc[table[chart.y].idxmax()]
    low = table.loc[table[chart.y].idxmin()]
    first, last = chart.data.iloc[0], chart.data.iloc[-1]
    text = (
        f"{what.capitalize() if not metric else what} per {freq.lower()} from "
        f"{_period_label(first['period'], freq)} to {_period_label(last['period'], freq)} "
        f"({len(chart.data)} {freq.lower()}s, based on {_col(column)})."
    )
    if peak[chart.y] == low[chart.y]:
        text += f" Every {freq.lower()} has the same value: {fmt(peak[chart.y])}."
    else:
        text += (
            f" Highest: **{_period_label(peak['period'], freq)}** ({fmt(peak[chart.y])}); lowest: "
            f"{_period_label(low['period'], freq)} ({fmt(low[chart.y])})."
        )
    notes = ["The first and last periods may only be partly covered by the data."]
    if chart.note:
        notes.append(chart.note)
    title = f"{what.capitalize() if not metric else what} over time"
    return Section(title, text, chart, "line", tuple(notes))


def _comparison_column(df: pd.DataFrame) -> Any | None:
    for column in de.categorical_columns(df):
        if de.column_kind(df[column]) != de.CATEGORICAL or looks_like_id(df, column):
            continue
        if COMPARISON_MIN_GROUPS <= df[column].nunique(dropna=True) <= COMPARISON_MAX_GROUPS:
            return column
    return None


def _comparison(data: pd.DataFrame, metric: MetricInfo | None, skipped: list[str]) -> Section | None:
    column = _comparison_column(data)
    if column is None:
        skipped.append(
            f"Comparison: no text column with {COMPARISON_MIN_GROUPS}-{COMPARISON_MAX_GROUPS} "
            "different values was found."
        )
        return None
    if metric:
        table = de.top_n(data, column, n=COMPARISON_MAX_GROUPS, value_column=metric.column, agg="sum")
        y = table.columns[-1]
        total = data[metric.column].sum()
        chart = ChartData(table, x=column, y=y, title=f"{metric.label} by {column}")
        what = f"the highest total {metric.label}"
    else:
        chart = de.bar_chart_data(data, column, top=COMPARISON_MAX_GROUPS)
        table, y, total = chart.data, "count", len(data)
        chart = ChartData(table, x=column, y=y, title=f"Rows by {column}")
        what = "the most rows"

    top_value = table.iloc[0][y]
    leaders = table[table[y] == top_value][column].tolist()
    if len(leaders) > 1:
        text = (
            f"{' and '.join(f'**{fmt(v)}**' for v in leaders)} share {what} "
            f"({fmt(top_value)} each, {_pct(top_value, total)} of the total) among "
            f"{len(table)} values of {_col(column)}."
        )
    else:
        text = (
            f"**{fmt(leaders[0])}** has {what}: {fmt(top_value)} ({_pct(top_value, total)} of the "
            f"total), among {len(table)} values of {_col(column)}."
        )
    notes = ("Missing values are shown as '(missing)'.",) if de.MISSING_LABEL in table[column].tolist() else ()
    return Section(chart.title, text, chart, "bar", notes)


def _id_columns(df: pd.DataFrame) -> list[str]:
    return [
        f"{_col(c)} looks like an ID, so it is not averaged."
        for c in de.numeric_columns(df)
        if looks_like_id(df, c)
    ]


def _quality(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if df.empty:
        return ["The dataset has no rows, so there is nothing to check."]

    missing = profiling.missing_values(df)
    with_missing = missing[missing["missing"] > 0]
    for _, row in with_missing.head(MAX_QUALITY_COLUMNS).iterrows():
        if row["missing"] == len(df):
            notes.append(f"{_col(row['column'])} is completely empty.")
        else:
            notes.append(
                f"{_col(row['column'])} has {fmt(row['missing'])} missing values ({row['missing_pct']:.1f}%)."
            )
    if len(with_missing) > MAX_QUALITY_COLUMNS:
        notes.append(f"{len(with_missing) - MAX_QUALITY_COLUMNS} more column(s) have missing values.")

    duplicates = profiling.duplicate_row_count(df)
    if duplicates:
        notes.append(f"{fmt(duplicates)} row(s) are exact duplicates of an earlier row.")

    kinds = de.column_kinds(df)
    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            continue
        if len(df) > 1 and values.nunique() == 1:
            notes.append(f"{_col(column)} has the same value in every row ({fmt(values.iloc[0])}).")
        if kinds[column] == de.CATEGORICAL:
            as_numbers = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
            share = as_numbers.notna().mean()
            if share == 1:
                notes.append(
                    f"{_col(column)} contains only numbers but is stored as text (e.g. IDs or "
                    "codes), so it is not used in calculations."
                )
            elif share >= NUMBERS_AS_TEXT_SHARE:
                notes.append(
                    f"{_col(column)} is stored as text: most values are numbers, but "
                    f"{fmt(int(as_numbers.isna().sum()))} are not, so it is not used in calculations."
                )
        if kinds[column] == de.DATETIME and not pd.api.types.is_datetime64_any_dtype(df[column]):
            unreadable = int(de.to_datetime_series(df[column]).isna().sum() - df[column].isna().sum())
            if unreadable:
                notes.append(f"{fmt(unreadable)} value(s) in {_col(column)} could not be read as dates.")

    if not notes:
        notes.append("No missing values, duplicate rows or other data-quality issues were found.")
    return notes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_summary(df: pd.DataFrame) -> ExecutiveSummary:
    """Summarise ``df``. The DataFrame passed in is never modified."""
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return ExecutiveSummary(
            headline="The dataset has no columns, so there is nothing to summarise.",
            kpis=(),
            skipped=("All sections: the dataset has no columns.",),
        )

    overview = profiling.dataset_overview(df)
    cells = overview.rows * overview.columns
    kpis = [
        KPI("Rows", fmt(overview.rows)),
        KPI("Columns", fmt(overview.columns)),
        KPI(
            "Complete cells",
            _pct(cells - overview.missing_cells, cells) if cells else "n/a",
            f"{fmt(overview.missing_cells)} of {fmt(cells)} cells are empty.",
        ),
        KPI("Duplicate rows", fmt(overview.duplicate_rows)),
    ]
    skipped: list[str] = []
    if df.empty:
        return ExecutiveSummary(
            headline=f"The dataset has {_count(overview.columns, 'column')} but no rows.",
            kpis=tuple(kpis),
            quality=tuple(_quality(df)),
            skipped=("Trend, comparison and averages: the dataset has no rows.",),
        )

    data = df
    metric = None
    found = _metric(df, skipped)
    exclude: set[Any] = set()
    if found:
        metric, values = found
        data = df.copy()
        data[metric.column] = values
        known = values.dropna()
        help_text = metric.formula
        if len(known) < len(values):
            help_text += f" {fmt(len(values) - len(known))} row(s) with missing inputs are left out."
        kpis.append(KPI(f"Total {metric.label}", fmt(known.sum()), help_text))
    else:
        for column in df.columns:  # an existing revenue/sales column: its total is meaningful
            if metrics.match_metric(words(column)) and de.is_numeric(df[column]):
                kpis.append(KPI(f"Total {column}", fmt(df[column].sum()), "Sum of the existing column."))
                exclude.add(column)
                break
    kpis += _average_kpis(df, exclude)

    headline = f"{_count(overview.rows, 'row')} and {_count(overview.columns, 'column')}"
    dates = de.date_columns(df)
    if len(dates) == 1:
        parsed = de.to_datetime_series(df[dates[0]]).dropna()
        if not parsed.empty:
            headline += f", from {fmt(parsed.min())} to {fmt(parsed.max())}"
    headline += f"; {kpis[2].value} of cells are filled in."

    return ExecutiveSummary(
        headline=headline,
        kpis=tuple(kpis),
        metric=metric,
        trend=_trend(data, metric, skipped),
        comparison=_comparison(data, metric, skipped),
        quality=tuple(_quality(df)),
        info=tuple(_id_columns(df)),
        skipped=tuple(skipped),
        data=data,
    )
