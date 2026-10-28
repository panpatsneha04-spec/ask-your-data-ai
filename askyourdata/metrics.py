"""Calculated metrics: gross amount and net revenue from existing columns.

    Gross amount = quantity x unit price
    Net revenue  = quantity x unit price x (1 - discount)

Rules:
* Input columns are found by name with ``ColumnMatcher`` (synonyms such as
  qty/units/price work). If a name fits two columns equally well, the user
  is asked; if an input is missing, the error says what is needed.
* How discounts are stored is decided from the DATA, never assumed:
    - values written with "%" ("10%")         -> percentages
    - any value above 1 (all within 0-100)    -> whole-number percentages, 10 = 10%
    - all values 0                            -> no discount
    - all values between 0 and 1 (e.g. 0.15)  -> unclear (15% or 0.15%?) -> ask,
      unless the question says "discount as percent" / "discount as fraction"
    - negative or above 100                   -> invalid, refused
* Missing or non-numeric inputs make that row's metric missing (never 0).
* The result is a NEW Series. The dataset and any existing revenue column
  are never changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from askyourdata import data_engine as de
from askyourdata.data_engine import AnalysisError
from askyourdata.text_matching import ClarificationNeeded, ColumnMatcher, fmt, words

GROSS_AMOUNT = "gross_amount"
NET_REVENUE = "net_revenue"
GENERIC_REVENUE = "generic_revenue"  # "revenue" / "sales": gross or net? -> ask

DISCOUNT_PERCENT = "percent"
DISCOUNT_FRACTION = "fraction"

# A text column must be at least this share numbers to be used as an input.
MIN_NUMERIC_SHARE = 0.5


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    phrases: frozenset[tuple[str, ...]]
    needs_discount: bool


def _phrases(*texts: str) -> frozenset[tuple[str, ...]]:
    return frozenset(words(t) for t in texts)


METRICS = {
    GROSS_AMOUNT: MetricSpec(
        GROSS_AMOUNT,
        "Gross amount (calculated)",
        _phrases(
            "gross amount", "gross revenue", "gross sales", "gross sales amount", "gross value",
            "order value", "line total", "revenue before discount", "sales before discount",
        ),
        needs_discount=False,
    ),
    NET_REVENUE: MetricSpec(
        NET_REVENUE,
        "Net revenue (calculated)",
        _phrases(
            "net revenue", "net sales", "net amount", "net sales amount", "net value",
            "revenue after discount", "sales after discount", "discounted revenue",
        ),
        needs_discount=True,
    ),
}  # fmt: skip
_GENERIC = _phrases("revenue", "sales", "sales amount", "turnover", "income")

_INPUT_PHRASES = {"quantity": words("quantity"), "unit price": words("unit price")}


class MetricUnavailable(AnalysisError):
    """The dataset lacks the columns a metric needs. The message explains which."""


class DiscountScaleUnclear(ClarificationNeeded):
    """Discount values could be fractions or percentages; the user must say which."""


@dataclass(frozen=True)
class MetricResult:
    key: str
    label: str
    values: pd.Series  # one value per row, named ``label``
    formula: str  # with the real column names
    columns: dict[str, Any]  # role -> column used
    notes: tuple[str, ...]  # how discounts were read, invalid values, ...


def match_metric(phrase: Sequence[str]) -> str | None:
    """GROSS_AMOUNT, NET_REVENUE, GENERIC_REVENUE or None for a phrase."""
    phrase = tuple(phrase)
    for spec in METRICS.values():
        if phrase in spec.phrases:
            return spec.key
    return GENERIC_REVENUE if phrase in _GENERIC else None


def _discount_columns(matcher: ColumnMatcher) -> list[Any]:
    return [c for c in matcher.columns if "discount" in words(c)]


_PRICE_WORDS = frozenset(words("price"))
_QUANTITY_WORDS = frozenset(words("quantity qty units volume sold"))


def _fits_role(column: Any, role: str) -> bool:
    """A price input must be named like a price; a quantity input like a quantity
    (and not like a price), so e.g. a lone `units` column is never used as the price."""
    w = set(words(column))
    if role == "unit price":
        return bool(w & _PRICE_WORDS)
    return bool(w & _QUANTITY_WORDS) and not (w & _PRICE_WORDS)


def _find_inputs(matcher: ColumnMatcher, spec: MetricSpec) -> tuple[dict[str, Any], list[str]]:
    """The columns for each input role, plus the roles that are missing."""
    found: dict[str, Any] = {}
    missing: list[str] = []
    for role, phrase in _INPUT_PHRASES.items():
        candidates = [
            c for c in matcher.candidates(phrase)
            if _fits_role(c.column, role) and c.column not in found.values()
        ]  # fmt: skip
        if not candidates:
            missing.append(role)
            continue
        best = [c for c in candidates if c.score == candidates[0].score]
        if len(best) > 1:
            raise ClarificationNeeded(
                f"'{role}' could mean more than one column: {', '.join(f'`{c.column}`' for c in best)}. "
                f"Which one should be used for {spec.label}?",
                [c.column for c in best],
            )
        found[role] = best[0].column
    if spec.needs_discount:
        discounts = _discount_columns(matcher)
        if not discounts:
            missing.append("discount")
        elif len(discounts) > 1:
            raise ClarificationNeeded(
                f"There are several discount columns: {', '.join(f'`{c}`' for c in discounts)}. "
                "I can't tell which one to use for net revenue.",
                discounts,
            )
        else:
            found["discount"] = discounts[0]
    return found, missing


def missing_inputs(df: pd.DataFrame, key: str) -> list[str]:
    """Which inputs a metric lacks in this dataset (empty list = can be tried)."""
    return _find_inputs(ColumnMatcher(df), METRICS[key])[1]


def _to_numbers(series: pd.Series, column: Any) -> tuple[pd.Series, int, bool]:
    """(numbers, count of invalid values, whether values were written with %)."""
    if de.is_numeric(series):
        return series.astype(float), 0, False
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    has_percent = bool(text.str.endswith("%").fillna(False).any())
    numbers = pd.to_numeric(text.str.rstrip("%"), errors="coerce").astype(float)
    present = int(series.notna().sum())
    invalid = int((series.notna() & numbers.isna()).sum())
    if present == 0 or (present - invalid) < MIN_NUMERIC_SHARE * present:
        raise MetricUnavailable(f"Column '{column}' does not contain numbers.")
    return numbers, invalid, has_percent


def _discount_rate(
    values: pd.Series, column: Any, has_percent: bool, scale: str | None
) -> tuple[pd.Series, str, str]:
    """(rate between 0 and 1, scale used, explanation)."""
    present = values.dropna()
    low, high = (present.min(), present.max()) if not present.empty else (0.0, 0.0)
    if (present < 0).any() or (present > 100).any():
        raise MetricUnavailable(
            f"Column '{column}' has values from {fmt(low)} to {fmt(high)}. Discounts must be "
            "between 0 and 100 (percent) or 0 and 1 (fraction), so net revenue can't be calculated."
        )

    if scale == DISCOUNT_FRACTION:
        if (present > 1).any():
            raise MetricUnavailable(
                f"Column '{column}' has values above 1 (up to {fmt(high)}), so they can't be "
                "fractions. Try 'discount as percent' instead."
            )
        return values, DISCOUNT_FRACTION, f"`{column}` was read as fractions (0.1 = 10%), as you asked."
    if scale == DISCOUNT_PERCENT:
        return values / 100, DISCOUNT_PERCENT, f"`{column}` was read as percentages (10 = 10%), as you asked."
    if has_percent:
        return values / 100, DISCOUNT_PERCENT, f"`{column}` values are written with '%', so 10% = 0.10."
    if (present > 1).any():
        return values / 100, DISCOUNT_PERCENT, (
            f"`{column}` values range from {fmt(low)} to {fmt(high)}, so they were read as "
            "whole-number percentages (10 = 10%)."
        )
    if (present == 0).all():
        return values, DISCOUNT_FRACTION, f"All `{column}` values are 0, so no discount was applied."
    raise DiscountScaleUnclear(
        f"All `{column}` values are between 0 and 1 (e.g. {fmt(present[present > 0].iloc[0])}). "
        f"That could mean a fraction (0.15 = 15%) or a percentage (0.15%). Which is it?",
        [DISCOUNT_FRACTION, DISCOUNT_PERCENT],
    )


def compute_metric(
    df: pd.DataFrame,
    key: str,
    matcher: ColumnMatcher | None = None,
    discount_scale: str | None = None,
) -> MetricResult:
    """Calculate a metric for every row of ``df`` (``df`` itself is not changed)."""
    spec = METRICS[key]
    matcher = matcher or ColumnMatcher(df)
    found, missing = _find_inputs(matcher, spec)
    if missing:
        needs = "quantity, unit price and discount" if spec.needs_discount else "quantity and unit price"
        raise MetricUnavailable(
            f"{spec.label.replace(' (calculated)', '')} can't be calculated: it needs {needs} "
            f"columns, and no {' or '.join(missing)} column was found. "
            f"Columns in this dataset: {', '.join(f'`{c}`' for c in df.columns)}."
        )

    notes: list[str] = []
    numbers: dict[str, pd.Series] = {}
    percent_sign = False
    for role, column in found.items():
        values, invalid, has_percent = _to_numbers(df[column], column)
        numbers[role] = values
        percent_sign = percent_sign or (role == "discount" and has_percent)
        if invalid:
            notes.append(f"{invalid} value(s) in `{column}` were not numbers and were treated as missing.")
        if role != "discount" and (values < 0).any():
            notes.append(f"{int((values < 0).sum())} row(s) have a negative `{column}`; they were included as-is.")

    q, p = found["quantity"], found["unit price"]
    result = numbers["quantity"] * numbers["unit price"]
    formula = f"{spec.label} = `{q}` × `{p}`"
    if spec.needs_discount:
        d = found["discount"]
        rate, scale, explanation = _discount_rate(numbers["discount"], d, percent_sign, discount_scale)
        result = result * (1 - rate)
        formula += f" × (1 − `{d}` / 100)" if scale == DISCOUNT_PERCENT else f" × (1 − `{d}`)"
        notes.insert(0, explanation)

    return MetricResult(
        key=key,
        label=spec.label,
        values=result.rename(spec.label),
        formula=formula,
        columns=found,
        notes=tuple(notes),
    )
