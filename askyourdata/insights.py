"""Automatic, evidence-based insights about a dataset (deterministic, no LLM).

``find_insights(df)`` looks for a fixed set of patterns and returns structured
``Insight`` objects. Every insight carries the actual values it is based on
and a description of how it was calculated. Nothing is guessed:

Performance
* ranking     highest and lowest group of a text column with 2-20 values
* top items   the top 5 of a text column with 21-500 values (products, cities...)
* metric      net revenue / gross amount, only when ``summary`` can calculate it
              without any clarification
Trends
* change      first vs last FULLY covered period of the single date column
Data quality
* missing values, empty columns, duplicate rows, constant columns,
  potential outliers (3 x IQR rule), negative values where a column name
  suggests they are unusual, percentages outside 0-100, unreadable dates,
  numbers stored as text

Guardrails: ID-like columns are never ranked or summed; prices, rates, scores
and percentages are averaged, never summed; differences are described, not
tested for statistical significance; no causes are claimed; heuristic
anomalies are labelled as potential issues. The DataFrame is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from askyourdata import data_engine as de
from askyourdata import profiling
from askyourdata.data_engine import AnalysisError
from askyourdata.summary import _frequency, _metric, _period_label
from askyourdata.text_matching import fmt, looks_like_id, words

PERFORMANCE = "performance"
TREND = "trend"
QUALITY = "data quality"
CATEGORIES = (PERFORMANCE, TREND, QUALITY)
LIMITS = {PERFORMANCE: 6, TREND: 2, QUALITY: 8}

MAX_RANKING_GROUPS = 20  # 2-20 values: highest / lowest ranking
MAX_LIST_GROUPS = 500  # 21-500 values: a top-5 list
MAX_RANKING_DIMENSIONS = 3
TOP_ITEMS = 5
MIN_GROUP_ROWS = 3  # groups with fewer rows are flagged as "based on few rows"
NOTABLE_RATIO = 1.5  # "x times the lowest" is mentioned from this ratio
FLAT_CHANGE_PCT = 5.0  # changes smaller than this are "roughly flat"
FULL_PERIOD_SHARE = 0.85  # a period counts as complete if the data covers 85% of it
MIN_VALUES_FOR_OUTLIERS = 10
OUTLIER_IQR = 3.0
NUMBERS_AS_TEXT_SHARE = 0.8
MAX_EMPTY_PERIOD_SHARE = 0.3  # coarser periods are used when more periods than this have no rows
MIN_ROWS_PER_PERIOD = 3  # ... or when periods would hold fewer rows than this on average
FREQUENCY_LADDER = ("Day", "Week", "Month", "Quarter", "Year")
MAX_LISTED = 5

# The LAST word of a column name decides whether its values may be added up.
ADDITIVE_WORDS = frozenset(words(
    "quantity qty unit units volume count revenue sale sales amount total profit cost spend income turnover"
))  # fmt: skip
# Summable columns named like money come before quantities ("revenue" before "qty").
MONEY_WORDS = frozenset(words("revenue sale sales amount profit cost spend income turnover"))
# Column names that suggest negative values are unusual (checked on the last word).
NON_NEGATIVE_WORDS = frozenset(words("quantity qty unit units count price amount age rating score weight"))
PERCENT_WORDS = frozenset(words("pct percent percentage"))

NOT_A_TEST = "This is a description of the data, not a statistical test, and it does not explain why."


@dataclass(frozen=True, eq=False)
class Insight:
    key: str  # stable identifier (kind + columns), used for ordering and de-duplication
    category: str  # performance / trend / data quality
    kind: str
    title: str
    explanation: str
    values: tuple[tuple[str, str], ...]  # the supporting numbers, as (label, value)
    evidence: str  # how it was calculated
    score: float = 0.0  # used only to order insights within a category
    table: pd.DataFrame | None = field(default=None, repr=False)


@dataclass(frozen=True, eq=False)
class InsightReport:
    insights: tuple[Insight, ...]
    notes: dict  # category -> why there is nothing (more) to show
    rows: int

    def by_category(self, category: str) -> list[Insight]:
        return [i for i in self.insights if i.category == category]


@dataclass(frozen=True)
class _Measure:
    column: Any  # None means "number of rows"
    label: str  # e.g. "total Quantity", "average Unit_Price", "rows"
    agg: str  # sum / mean / count
    reason: str  # why this measure and aggregation were chosen


def _col(column: Any) -> str:
    return f"`{column}`"


def _cap(text: str) -> str:
    """First letter upper-case only (str.capitalize would lower-case column names)."""
    return text[:1].upper() + text[1:]


def _last_word(column: Any) -> str:
    w = words(column)
    return w[-1] if w else ""


# ---------------------------------------------------------------------------
# Choosing what to measure and what to compare
# ---------------------------------------------------------------------------


def _usable_numbers(df: pd.DataFrame) -> list:
    """Numeric columns that can be measured: not IDs, not empty, not constant."""
    usable = []
    for column in de.numeric_columns(df):
        values = df[column].dropna()
        if values.nunique() < 2 or looks_like_id(df, column):
            continue
        usable.append(column)
    return usable


def _choose_measure(df: pd.DataFrame, notes: list) -> tuple[_Measure, pd.DataFrame, Insight | None]:
    """The main measure: a calculated metric, else a summable column, else an average."""
    skipped: list[str] = []
    found = _metric(df, skipped)
    notes.extend(skipped)
    if found:
        info, values = found
        data = df.copy()
        data[info.column] = values
        known = values.dropna()
        insight = Insight(
            key=f"metric:{info.column}",
            category=PERFORMANCE,
            kind="metric",
            title=f"Total {info.label}: {fmt(known.sum())}",
            explanation=" ".join(
                [f"{info.label} is calculated by the app from existing columns.", *info.notes]
                + ([f"{fmt(len(values) - len(known))} row(s) with a missing input are left out (not counted as 0)."]
                   if len(known) < len(values) else [])
            ),  # fmt: skip
            values=(("Total", fmt(known.sum())), ("Rows used", fmt(len(known))), ("Formula", info.formula)),
            evidence=f"metrics.compute_metric; {info.formula}.",
            score=1e9,  # shown first: it explains the measure used below
        )
        measure = _Measure(info.column, f"total {info.label}", "sum", f"{info.label} is a calculated amount, so it is summed.")
        return measure, data, insight

    numbers = _usable_numbers(df)
    additive = [c for c in numbers if _last_word(c) in ADDITIVE_WORDS]
    additive.sort(key=lambda c: _last_word(c) not in MONEY_WORDS)  # stable: money-like first
    if additive:
        column = additive[0]
        reason = f"{_col(column)} is an amount-like column, so it is summed."
        return _Measure(column, f"total {column}", "sum", reason), df, None
    if numbers:
        column = numbers[0]
        reason = f"{_col(column)} looks like a price, rate, score or measurement, so it is averaged, not summed."
        return _Measure(column, f"average {column}", "mean", reason), df, None
    return _Measure(None, "rows", "count", "No numeric column can be measured, so rows are counted."), df, None


def _text_id_like(values: pd.Series) -> bool:
    """Free text or codes: (almost) every row has its own value."""
    return len(values) > MAX_RANKING_GROUPS and values.nunique() >= 0.9 * len(values)


def _dimensions(df: pd.DataFrame) -> tuple[list, list]:
    """(columns to rank: 2-20 values, columns for a top-5 list: 21-500 values)."""
    ranking, listing = [], []
    for column in de.categorical_columns(df):
        if de.column_kind(df[column]) != de.CATEGORICAL or looks_like_id(df, column):
            continue
        values = df[column].dropna()
        distinct = values.nunique()
        if distinct < 2 or _text_id_like(values):
            continue
        if distinct <= MAX_RANKING_GROUPS:
            ranking.append(column)
        elif distinct <= MAX_LIST_GROUPS:
            listing.append(column)
    return ranking[:MAX_RANKING_DIMENSIONS], listing[:1]


def _group_table(data: pd.DataFrame, dimension: Any, measure: _Measure) -> tuple[pd.DataFrame, Any, int]:
    """Measure per group, without the "(missing)" group: (table, value column, rows left out)."""
    if measure.column is None:
        table = de.group_summary(data, [dimension])
        y = "rows"
    else:
        table = de.group_summary(data, [dimension], measure.column, [measure.agg])
        y = table.columns[-1]
    missing = table[table[dimension] == de.MISSING_LABEL]
    left_out = int(missing["rows"].sum()) if not missing.empty else 0
    table = table[table[dimension] != de.MISSING_LABEL].dropna(subset=[y])
    # Deterministic order: by value (highest first), ties broken alphabetically.
    table = table.assign(_name=table[dimension].astype(str))
    table = table.sort_values([y, "_name"], ascending=[False, True], kind="stable").drop(columns="_name")
    return table.reset_index(drop=True), y, left_out


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def _ranking(data: pd.DataFrame, dimension: Any, measure: _Measure) -> Insight | None:
    table, y, left_out = _group_table(data, dimension, measure)
    if len(table) < 2:
        return None
    top, bottom = table.iloc[0], table.iloc[-1]
    label = measure.label
    evidence = (
        f"data_engine.group_summary(group_by={dimension}, value={measure.column or 'rows'}, "
        f"agg={measure.agg}). {measure.reason}"
    )
    if left_out:
        evidence += f" {fmt(left_out)} row(s) with a missing {dimension} were left out."

    if top[y] == bottom[y]:
        return Insight(
            key=f"ranking:{dimension}", category=PERFORMANCE, kind="ranking",
            title=f"Every {dimension} has the same {label}",
            explanation=f"All {len(table)} values of {_col(dimension)} have {label} = {fmt(top[y])}.",
            values=(("Value", fmt(top[y])), ("Groups", fmt(len(table)))),
            evidence=evidence, score=0.0, table=table,
        )  # fmt: skip

    leaders = table[table[y] == top[y]][dimension].tolist()
    if len(leaders) > 1:
        title = f"{' and '.join(map(str, leaders))} share the highest {label} by {dimension}"
    else:
        title = f"{top[dimension]} has the highest {label} by {dimension}"
    sentences = [
        f"{top[dimension]} has the highest {label} ({fmt(top[y])}, from {fmt(top['rows'])} rows); "
        f"{bottom[dimension]} has the lowest ({fmt(bottom[y])}, from {fmt(bottom['rows'])} rows), "
        f"among {len(table)} values of {_col(dimension)}."
    ]
    values = [("Highest", f"{top[dimension]}: {fmt(top[y])}"), ("Lowest", f"{bottom[dimension]}: {fmt(bottom[y])}"),
              ("Groups compared", fmt(len(table)))]  # fmt: skip
    ratio = top[y] / bottom[y] if bottom[y] > 0 else None
    if ratio is not None and ratio >= NOTABLE_RATIO:
        sentences.append(f"The highest is {ratio:.1f}× the lowest.")
        values.append(("Highest ÷ lowest", f"{ratio:.1f}×"))
    total = table[y].sum()
    if measure.agg in ("sum", "count") and total > 0 and (table[y] >= 0).all():
        share = top[y] / total * 100
        sentences.append(f"{top[dimension]} makes up {share:.1f}% of the total.")
        values.append(("Share of total", f"{share:.1f}%"))
    if (table[y] < 0).any():
        sentences.append("Some groups have negative values.")
    small = [str(r[dimension]) for r in (top, bottom) if r["rows"] < MIN_GROUP_ROWS]
    if small:
        sentences.append(f"Based on fewer than {MIN_GROUP_ROWS} rows for {', '.join(small)}, so it can change easily.")
    sentences.append(NOT_A_TEST)
    spread = (top[y] - bottom[y]) / abs(table[y].mean()) if table[y].mean() else 0.0
    return Insight(
        key=f"ranking:{dimension}", category=PERFORMANCE, kind="ranking", title=title,
        explanation=" ".join(sentences), values=tuple(values), evidence=evidence,
        score=float(spread), table=table,
    )  # fmt: skip


def _top_items(data: pd.DataFrame, dimension: Any, measure: _Measure) -> Insight | None:
    table, y, left_out = _group_table(data, dimension, measure)
    if len(table) < TOP_ITEMS + 1:
        return None
    top = table.head(TOP_ITEMS)
    listing = ", ".join(f"{r[dimension]} ({fmt(r[y])})" for _, r in top.iterrows())
    return Insight(
        key=f"top:{dimension}", category=PERFORMANCE, kind="top items",
        title=f"Top {TOP_ITEMS} {dimension} by {measure.label}",
        explanation=f"Out of {len(table)} values of {_col(dimension)}: {listing}. {NOT_A_TEST}",
        values=tuple((f"#{i}", f"{r[dimension]}: {fmt(r[y])}") for i, (_, r) in enumerate(top.iterrows(), 1)),
        evidence=f"data_engine.group_summary(group_by={dimension}, agg={measure.agg}), sorted highest first. {measure.reason}",
        score=0.5, table=top,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------


def _complete_periods(table: pd.DataFrame, dates: pd.Series, freq: str) -> pd.DataFrame:
    """Drop the first / last period when the data covers less than 85% of it."""
    code = de.TREND_FREQUENCIES[freq]
    first, last = pd.Period(table["period"].iloc[0], freq=code), pd.Period(table["period"].iloc[-1], freq=code)
    keep = pd.Series(True, index=table.index)
    length = lambda p: (p.end_time - p.start_time).total_seconds()  # noqa: E731
    if (first.end_time - dates.min()).total_seconds() < FULL_PERIOD_SHARE * length(first):
        keep.iloc[0] = False
    if (dates.max() - last.start_time).total_seconds() < FULL_PERIOD_SHARE * length(last):
        keep.iloc[-1] = False
    return table[keep].reset_index(drop=True)


def _trend_frequency(dates: pd.Series) -> str:
    """The span-based period (as in the executive summary), made coarser while too many
    periods would have no rows at all (empty periods would count as 0 and distort a trend)."""
    freq = _frequency((dates.max() - dates.min()).days)
    for candidate in FREQUENCY_LADDER[FREQUENCY_LADDER.index(freq):]:
        periods = dates.dt.to_period(de.TREND_FREQUENCIES[candidate])
        full_range = pd.period_range(periods.min(), periods.max(), freq=de.TREND_FREQUENCIES[candidate])
        empty_share = 1 - periods.nunique() / len(full_range)
        if empty_share <= MAX_EMPTY_PERIOD_SHARE and len(dates) / len(full_range) >= MIN_ROWS_PER_PERIOD:
            return candidate
    return FREQUENCY_LADDER[-1]


def _trend(data: pd.DataFrame, measure: _Measure, notes: list) -> list[Insight]:
    dates = de.date_columns(data)
    if not dates:
        notes.append("No date column was found, so no trend can be calculated.")
        return []
    if len(dates) > 1:
        notes.append(
            f"There are several date columns ({', '.join(map(_col, dates))}), so no trend is chosen "
            "automatically. Ask e.g. “trend by month using <column>” in Ask Your Data."
        )
        return []
    column = dates[0]
    parsed = de.to_datetime_series(data[column]).dropna()
    if parsed.nunique() < 2:
        notes.append(f"{_col(column)} has fewer than two different valid dates.")
        return []
    freq = _trend_frequency(parsed)

    insights = []
    measures = [measure] if measure.column is None else [measure, _Measure(None, "rows", "count", "Rows are counted.")]
    for m in measures:
        result = de.date_trend(data, column, m.column, freq=freq, agg=m.agg)
        table = result.table.dropna(subset=[result.value_label])
        complete = _complete_periods(table, parsed, freq) if len(table) >= 2 else table
        # Start and end only at periods that contain rows (empty periods were filled with 0).
        code = de.TREND_FREQUENCIES[freq]
        with_rows = set(parsed.dt.to_period(code))
        has_rows = complete["period"].map(lambda ts: pd.Period(ts, freq=code) in with_rows)
        if has_rows.any():
            first_i, last_i = has_rows.idxmax(), has_rows[::-1].idxmax()
            complete = complete.loc[first_i:last_i].reset_index(drop=True)
        if len(complete) < 2:
            notes.append(f"Fewer than two fully covered {freq.lower()}s in {_col(column)}, so no reliable trend.")
            break
        y = result.value_label
        first, last = complete.iloc[0], complete.iloc[-1]
        start, end = _period_label(first["period"], freq), _period_label(last["period"], freq)
        change = last[y] - first[y]
        steps = complete[y].diff().dropna()
        ups, downs = int((steps > 0).sum()), int((steps < 0).sum())
        if first[y] > 0:
            pct = change / first[y] * 100
            direction = "stayed roughly flat" if abs(pct) < FLAT_CHANGE_PCT else ("rose" if pct > 0 else "fell")
            amount = f"{abs(pct):.1f}%"
            title = f"{_cap(m.label)} {direction}" + ("" if direction.startswith("stayed") else f" {amount}") + f" from {start} to {end}"
            change_text = f"{'+' if change >= 0 else '−'}{fmt(abs(change))} ({'+' if pct >= 0 else '−'}{abs(pct):.1f}%)"
        else:
            direction = "rose" if change > 0 else ("fell" if change < 0 else "stayed the same")
            title = f"{_cap(m.label)} {direction} from {start} to {end}"
            change_text = f"{'+' if change >= 0 else '−'}{fmt(abs(change))} (no percentage: the starting value is {fmt(first[y])})"
            pct = 0.0
        peak = complete.loc[complete[y].idxmax()]
        sentences = [
            f"Per {freq.lower()} of {_col(column)}: {fmt(first[y])} in {start} and {fmt(last[y])} in {end} "
            f"({len(complete)} fully covered {freq.lower()}s; {ups} rise(s) and {downs} fall(s) between them). "
            f"Highest: {_period_label(peak['period'], freq)} ({fmt(peak[y])}).",
        ]
        if len(complete) < len(table):
            sentences.append(f"{len(table) - len(complete)} partly covered period(s) at the start or end were left out.")
        if m.agg in ("sum", "count"):
            sentences.append(f"{freq}s without any rows count as 0.")
        if result.skipped_rows:
            sentences.append(f"{result.skipped_rows} row(s) with a missing or unreadable date were left out.")
        sentences.append(NOT_A_TEST)
        insights.append(Insight(
            key=f"trend:{column}:{m.column or 'rows'}", category=TREND, kind="change", title=title,
            explanation=" ".join(sentences),
            values=(("Start", f"{start}: {fmt(first[y])}"), ("End", f"{end}: {fmt(last[y])}"), ("Change", change_text)),
            evidence=f"data_engine.date_trend(date={column}, value={m.column or 'rows'}, freq={freq}, agg={m.agg}). {m.reason}",
            score=abs(float(pct)) + (1000.0 if m is measure else 0.0),  # the main measure first
            table=complete,
        ))  # fmt: skip
    return insights


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def _quality(df: pd.DataFrame) -> list[Insight]:
    found: list[Insight] = []
    rows = len(df)
    missing = profiling.missing_values(df)
    empty_columns = missing[missing["missing"] == rows]["column"].tolist()
    partial = missing[(missing["missing"] > 0) & (missing["missing"] < rows)]
    if empty_columns:
        found.append(Insight(
            key="quality:empty", category=QUALITY, kind="empty columns",
            title=f"{len(empty_columns)} completely empty column(s)",
            explanation=f"{', '.join(map(_col, empty_columns))} contain no values at all.",
            values=(("Columns", ", ".join(map(str, empty_columns))),),
            evidence="profiling.missing_values: missing count equals the number of rows.", score=100.0,
        ))  # fmt: skip
    if not partial.empty:
        shown = partial.head(MAX_LISTED)
        more = f" ({len(partial) - MAX_LISTED} more)" if len(partial) > MAX_LISTED else ""
        found.append(Insight(
            key="quality:missing", category=QUALITY, kind="missing values",
            title=f"Missing values in {len(partial)} column(s)",
            explanation=(f"{fmt(int(partial['missing'].sum()))} empty cells in total{more}. Missing values are left "
                         "out of calculations, never counted as 0."),
            values=tuple((str(r["column"]), f"{fmt(r['missing'])} ({r['missing_pct']:.1f}%)") for _, r in shown.iterrows()),
            evidence="profiling.missing_values (empty cells per column).", score=float(partial["missing_pct"].max()),
        ))  # fmt: skip
    duplicates = profiling.duplicate_row_count(df)
    if duplicates:
        found.append(Insight(
            key="quality:duplicates", category=QUALITY, kind="duplicate rows",
            title=f"{fmt(duplicates)} duplicate row(s)",
            explanation=(f"{fmt(duplicates)} row(s) are exact copies of an earlier row ({duplicates / rows * 100:.1f}% of "
                         "rows). They may be real repeats or double entries; the data alone can't tell."),
            values=(("Duplicate rows", fmt(duplicates)), ("Share of rows", f"{duplicates / rows * 100:.1f}%")),
            evidence="profiling.duplicate_row_count (rows identical in every column).", score=duplicates / rows * 100,
        ))  # fmt: skip

    kinds = de.column_kinds(df)
    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            continue
        if rows > 1 and values.nunique() == 1:
            found.append(Insight(
                key=f"quality:constant:{column}", category=QUALITY, kind="constant column",
                title=f"{column} has the same value in every row",
                explanation=f"Every non-empty value of {_col(column)} is {fmt(values.iloc[0])}, so it can't explain differences.",
                values=(("Value", fmt(values.iloc[0])), ("Rows", fmt(len(values)))),
                evidence="pandas nunique() == 1.", score=10.0,
            ))  # fmt: skip
            continue
        kind = kinds[column]
        if kind == de.NUMERIC and not looks_like_id(df, column):
            found.extend(_numeric_checks(df, column, values))
        elif kind == de.DATETIME and not pd.api.types.is_datetime64_any_dtype(df[column]):
            unreadable = int(de.to_datetime_series(df[column]).isna().sum() - df[column].isna().sum())
            if unreadable:
                found.append(Insight(
                    key=f"quality:dates:{column}", category=QUALITY, kind="unreadable dates",
                    title=f"{unreadable} unreadable date(s) in {column}",
                    explanation=f"{unreadable} value(s) of {_col(column)} could not be read as dates; trends leave them out.",
                    values=(("Unreadable", fmt(unreadable)), ("Readable", fmt(len(values) - unreadable))),
                    evidence="data_engine.to_datetime_series: values that became NaT.", score=unreadable / rows * 100,
                ))  # fmt: skip
        elif kind == de.CATEGORICAL:
            as_numbers = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
            share = as_numbers.notna().mean()
            if NUMBERS_AS_TEXT_SHARE <= share < 1:
                bad = int(as_numbers.isna().sum())
                found.append(Insight(
                    key=f"quality:text-numbers:{column}", category=QUALITY, kind="numbers stored as text",
                    title=f"{column} is mostly numbers stored as text",
                    explanation=(f"{share * 100:.0f}% of {_col(column)} are numbers, but {bad} value(s) are not, so "
                                 "the column is treated as text and not calculated with. This is a potential issue."),
                    values=(("Numeric values", f"{share * 100:.0f}%"), ("Non-numeric values", fmt(bad))),
                    evidence="pandas.to_numeric(errors='coerce') on the non-empty values.", score=(1 - share) * 100,
                ))  # fmt: skip
    if not found:
        found.append(Insight(
            key="quality:ok", category=QUALITY, kind="no issues",
            title="No data-quality issues found",
            explanation=("Checked: missing values, empty and constant columns, duplicate rows, potential outliers, "
                         "unusual negative values, percentages outside 0-100, unreadable dates and numbers stored as text."),
            values=(("Rows checked", fmt(rows)), ("Columns checked", fmt(df.shape[1]))),
            evidence="All data-quality checks of insights.find_insights returned nothing.", score=0.0,
        ))  # fmt: skip
    return found


def _numeric_checks(df: pd.DataFrame, column: Any, values: pd.Series) -> list[Insight]:
    found = []
    last = _last_word(column)
    rows = len(df)
    if len(values) >= MIN_VALUES_FOR_OUTLIERS:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - OUTLIER_IQR * iqr, q3 + OUTLIER_IQR * iqr
            outliers = values[(values < low) | (values > high)]
            if not outliers.empty:
                extremes = outliers.reindex(outliers.abs().sort_values(ascending=False).index).head(3)
                found.append(Insight(
                    key=f"quality:outliers:{column}", category=QUALITY, kind="potential outliers",
                    title=f"{len(outliers)} potential outlier(s) in {column}",
                    explanation=(f"{len(outliers)} value(s) of {_col(column)} are far from the rest (outside "
                                 f"{fmt(low)} to {fmt(high)}). This is a heuristic check (3 × IQR rule): they may be "
                                 "genuine values or errors; the data alone can't tell."),
                    values=(("Potential outliers", fmt(len(outliers))), ("Most extreme", ", ".join(fmt(v) for v in extremes)),
                            ("Typical range (Q1–Q3)", f"{fmt(q1)} to {fmt(q3)}")),  # fmt: skip
                    evidence=f"Values outside Q1 − {OUTLIER_IQR:g}×IQR and Q3 + {OUTLIER_IQR:g}×IQR (IQR = {fmt(iqr)}).",
                    score=len(outliers) / rows * 100 + 5,
                ))  # fmt: skip
    if last in NON_NEGATIVE_WORDS and (values < 0).any():
        negatives = values[values < 0]
        found.append(Insight(
            key=f"quality:negative:{column}", category=QUALITY, kind="negative values",
            title=f"{len(negatives)} negative value(s) in {column}",
            explanation=(f"{_col(column)} is usually not negative, but {len(negatives)} value(s) are (lowest "
                         f"{fmt(negatives.min())}). They may be returns or corrections, or errors: a potential issue."),
            values=(("Negative values", fmt(len(negatives))), ("Lowest", fmt(negatives.min()))),
            evidence=f"Values below 0 in a column whose name ends with '{last}'.", score=len(negatives) / rows * 100 + 5,
        ))  # fmt: skip
    if set(words(column)) & PERCENT_WORDS and ((values < 0) | (values > 100)).any():
        bad = values[(values < 0) | (values > 100)]
        found.append(Insight(
            key=f"quality:percent:{column}", category=QUALITY, kind="percentages out of range",
            title=f"{len(bad)} percentage(s) outside 0–100 in {column}",
            explanation=f"{_col(column)} looks like a percentage, but {len(bad)} value(s) are below 0 or above 100: a potential issue.",
            values=(("Out of range", fmt(len(bad))), ("Examples", ", ".join(fmt(v) for v in bad.head(3)))),
            evidence="Values below 0 or above 100 in a column named like a percentage.", score=len(bad) / rows * 100 + 5,
        ))  # fmt: skip
    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _ordered(insights: list[Insight]) -> list[Insight]:
    """De-duplicated, highest score first, then by key; at most LIMITS per category."""
    unique = list({i.key: i for i in insights}.values())
    result = []
    for category in CATEGORIES:
        group = sorted((i for i in unique if i.category == category), key=lambda i: (-i.score, i.key))
        result += group[: LIMITS[category]]
    return result


def find_insights(df: pd.DataFrame | None) -> InsightReport:
    """All insights the data supports. Deterministic; never modifies ``df``."""
    notes: dict[str, str] = {}
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        reason = "The dataset has no columns, so no insight can be generated."
        return InsightReport((), {c: reason for c in CATEGORIES}, 0)
    if df.empty:
        reason = "The dataset has no rows, so no insight can be generated."
        return InsightReport((), {c: reason for c in CATEGORIES}, 0)

    performance_notes: list[str] = []
    trend_notes: list[str] = []
    insights: list[Insight] = []
    try:
        measure, data, metric_insight = _choose_measure(df, performance_notes)
        if metric_insight:
            insights.append(metric_insight)
        ranking_dims, list_dims = _dimensions(df)
        for dimension in ranking_dims:
            if (insight := _ranking(data, dimension, measure)) is not None:
                insights.append(insight)
        for dimension in list_dims:
            if (insight := _top_items(data, dimension, measure)) is not None:
                insights.append(insight)
        if not ranking_dims and not list_dims:
            performance_notes.append(
                "No text column with 2–500 different values (that isn't an ID or free text) to compare groups by."
            )
        insights += _trend(data, measure, trend_notes)
    except AnalysisError as exc:  # unexpected shapes: report, never crash
        performance_notes.append(f"Some insights could not be calculated: {exc}")
    insights += _quality(df)

    ordered = _ordered(insights)
    if not [i for i in ordered if i.category == PERFORMANCE]:
        notes[PERFORMANCE] = " ".join(performance_notes) or "No reliable performance insight could be generated."
    elif performance_notes:
        notes[PERFORMANCE] = " ".join(performance_notes)
    if not [i for i in ordered if i.category == TREND]:
        notes[TREND] = " ".join(trend_notes) or "No reliable trend could be generated."
    return InsightReport(tuple(ordered), notes, len(df))
