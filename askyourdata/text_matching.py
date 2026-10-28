"""Text normalisation and matching of question phrases to real column names.

Shared by the question router, natural-language filters and calculated
metrics. Pure Python + pandas; nothing here runs user text as code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from askyourdata import data_engine as de


class ClarificationNeeded(Exception):
    """The question is unclear; ``message`` asks the user, ``options`` lists choices."""

    def __init__(
        self, message: str, options: Iterable[Any] = (), rephrasings: Iterable[str] = ()
    ) -> None:
        super().__init__(message)
        self.message = message
        self.options = tuple(str(o) for o in options)
        self.rephrasings = tuple(rephrasings)  # complete questions the user can pick


class UnsupportedRequest(Exception):
    """The question asks for something the app does not do."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def words(text: Any) -> tuple[str, ...]:
    """Lower-case singular words: "Unit_Price" / "unit prices" -> ("unit", "price")."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))  # camelCase
    return tuple(_singular(w) for w in _TOKEN_RE.findall(text.lower()))


_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "each", "every", "all", "per", "for", "in", "on", "by",
        "my", "this", "that", "these", "those", "dataset", "data", "table", "column",
        "field", "value", "is", "are", "there", "any", "with", "across", "do", "we",
        "have", "their", "its", "our", "it", "to", "show", "me", "please", "what",
        "which", "how", "s", "doe", "did", "ha", "was", "were", "be", "been", "using",
        "use", "based", "from",
    }
)  # fmt: skip


def _strip(seq: Sequence[str]) -> tuple[str, ...]:
    """Drop filler words at both ends: ("the", "unit", "price", "value") -> ("unit", "price")."""
    items = list(seq)
    while items and items[0] in _STOPWORDS:
        items.pop(0)
    while items and items[-1] in _STOPWORDS:
        items.pop()
    return tuple(items)


def _contains(seq: Sequence[str], sub: Sequence[str]) -> bool:
    n = len(sub)
    return n > 0 and any(tuple(seq[i : i + n]) == tuple(sub) for i in range(len(seq) - n + 1))


def _ends_with(seq: Sequence[str], sub: Sequence[str]) -> bool:
    return 0 < len(sub) <= len(seq) and tuple(seq[-len(sub) :]) == tuple(sub)


def _remove(seq: Sequence[str], sub: Sequence[str]) -> tuple[str, ...]:
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if n and tuple(seq[i : i + n]) == tuple(sub):
            return tuple(seq[:i]) + tuple(seq[i + n :])
    return tuple(seq)


def fmt(value: Any) -> str:
    """Readable numbers: 1234.5 -> "1,234.50", 85.0 -> "85"."""
    if value is None:
        return "no value"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "no value"
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _cap(text: str) -> str:
    """Capitalise the first letter only (str.capitalize would lower-case column names)."""
    return text[:1].upper() + text[1:]


def _col(column: Any) -> str:
    return f"`{column}`"


def _join(items: Iterable[Any]) -> str:
    return ", ".join(_col(i) for i in items) or "none"


# ---------------------------------------------------------------------------
# Column matching
# ---------------------------------------------------------------------------

_SYNONYM_GROUPS = [
    {"city", "town", "location"},
    {"product", "item", "product name", "item name"},
    {"quantity", "qty", "units", "units sold", "quantity sold", "volume"},
    {"price", "unit price", "selling price"},
    {"revenue", "sales", "sales amount", "sale amount", "turnover", "income", "total sales"},
    {"customer", "client", "buyer"},
    {"category", "product category", "product type"},
    {"payment method", "payment type", "payment mode", "payment"},
    {"region", "area", "zone", "territory"},
    {"rating", "score", "review score"},
    {"country", "nation"},
    {"state", "province"},
    {"store", "shop", "branch", "outlet"},
    {"employee", "staff", "salesperson", "sales rep"},
    {"profit", "earnings"},
    {"order id", "order number", "order no"},
]
_SYNONYMS = [frozenset(words(term) for term in group) for group in _SYNONYM_GROUPS]

# Words that mean "count the rows" when no column with that name exists.
ROW_WORDS = frozenset(
    {"row", "record", "entry", "order", "transaction", "observation", "purchase", "line", "case"}
)
_SALES_WORDS = frozenset({("sale",), ("sale", "amount"), ("revenue",), ("total", "sale")})


def _synonyms_for(phrase: tuple[str, ...]) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for group in _SYNONYMS:
        if phrase in group:
            found |= group
    found.discard(phrase)
    return found


@dataclass(frozen=True)
class ColumnCandidate:
    column: Any
    score: int
    reason: str


class ColumnMatcher:
    """Match phrases like "unit prices" or "location" to real column names.

    Scores (highest wins, ties are reported as ambiguous):
      100 exact name (ignoring case, spaces, underscores and plurals)
       90 same letters without spaces ("unitprice" ~ "unit_price")
       80 synonym ("quantity" ~ "units", "location" ~ "city")
       70 the phrase contains the full column name ("total units sold" ~ "units")
       60 the column name ends with the phrase ("date" ~ "order_date")
       50 the column name ends with a synonym ("location" ~ "customer_city")
    Only the end of a column name is matched, so "order" never matches
    "order_id" or "order_date".
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.columns = list(df.columns)
        self.kinds = de.column_kinds(df)
        self._words = {column: words(column) for column in self.columns}

    def candidates(self, phrase: Sequence[str]) -> list[ColumnCandidate]:
        phrase = _strip(phrase)
        if not phrase:
            return []
        label = " ".join(phrase)
        synonyms = _synonyms_for(phrase)
        found = []
        for column, cw in self._words.items():
            if not cw:
                continue
            if cw == phrase:
                score, reason = 100, "exact name"
            elif "".join(cw) == "".join(phrase):
                score, reason = 90, "same name without spaces/underscores"
            elif cw in synonyms and not _ends_with(cw, phrase):
                # A synonym that merely ends with the phrase ("unit price" for
                # "price") gets no priority over other names ending the same
                # way ("list price"); those fall through to the 60 rule.
                score, reason = 80, f"synonym of '{label}'"
            elif _contains(phrase, cw):
                score, reason = 70, f"'{label}' contains the column name"
            elif _ends_with(cw, phrase):
                score, reason = 60, f"column name ends with '{label}'"
            elif any(_ends_with(cw, s) for s in synonyms):
                score, reason = 50, f"column name ends with a synonym of '{label}'"
            else:
                continue
            found.append(ColumnCandidate(column, score, reason))
        return sorted(found, key=lambda c: -c.score)

    def add_column(self, column: Any, kind: str) -> None:
        """Register a calculated column (e.g. net revenue) so it can be matched."""
        if column not in self._words:
            self.columns.append(column)
        self.kinds[column] = kind
        self._words[column] = words(column)

    def names(self, kinds: Iterable[str] | None = None) -> list[Any]:
        kinds = set(kinds) if kinds else None
        return [c for c in self.columns if kinds is None or self.kinds[c] in kinds]

    def resolve(
        self, phrase: Sequence[str], role: str, kinds: Iterable[str] | None = None
    ) -> ColumnCandidate:
        """The single column meant by ``phrase``, or raise to ask the user."""
        phrase = _strip(phrase)
        kinds = set(kinds) if kinds else None
        suitable = self.names(kinds)
        if not phrase:
            raise ClarificationNeeded(f"Which column should I use as the {role}?", suitable)

        label = " ".join(phrase)
        found = self.candidates(phrase)
        if not found:
            hint = ""
            if phrase in _SALES_WORDS:
                hint = (
                    " If you meant a money amount, it is not a column here; it would "
                    "have to be calculated (e.g. quantity x price) before uploading."
                )
            raise ClarificationNeeded(
                f"I couldn't find a column matching '{label}' in this dataset.{hint} "
                f"Available columns: {_join(suitable or self.columns)}.",
                suitable or self.columns,
            )

        best = [c for c in found if c.score == found[0].score]
        if kinds:
            fitting = [c for c in best if self.kinds[c.column] in kinds]
            if not fitting:
                column = best[0].column
                raise ClarificationNeeded(
                    f"{_col(column)} is a {self.kinds[column]} column, but the {role} needs a "
                    f"{' or '.join(sorted(kinds))} column. Suitable columns: {_join(suitable)}.",
                    suitable,
                )
            best = fitting
        if len(best) > 1:
            raise ClarificationNeeded(
                f"'{label}' could mean more than one column: {_join(c.column for c in best)}. "
                "Which one did you mean? Please ask again using the column name.",
                [c.column for c in best],
            )
        return best[0]


_ID_WORDS = frozenset({"id", "code", "number", "no", "key", "zip", "postcode"})


def looks_like_id(df: pd.DataFrame, column: Any) -> bool:
    """IDs are numbers, but averaging or totalling them is meaningless."""
    name = words(column)
    if name and name[-1] in _ID_WORDS:
        return True
    values = df[column].dropna()
    return (
        len(values) > 1
        and pd.api.types.is_integer_dtype(values)
        and values.is_unique
        and values.is_monotonic_increasing
    )
