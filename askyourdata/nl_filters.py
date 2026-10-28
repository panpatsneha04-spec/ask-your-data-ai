"""Simple, safe row filters inside plain-English questions.

Supported filters:
* Text values that really exist in a column: "orders from Pune",
  "Electronics products", "paid by UPI", "where category is Home",
  "from Pune or Mumbai".
* Numbers: "quantity greater than 3", "unit price under 1000",
  "above quantity 2", "quantity between 2 and 4", ">= 3".
* Dates: "between 2026-01-01 and 2026-01-31", "after 2026-02-01",
  "before March 2026", "in February 2026", "since 1 March 2026", "in 2026".

Every filter becomes a ``data_engine.Condition`` and is applied with
``data_engine.filter_rows``: no eval, no SQL. Anything unclear (a value found
in two columns, an unknown value, a number without a column, an ambiguous
date like 03/04/2026) raises ``ClarificationNeeded`` instead of being guessed
or silently ignored.
"""

from __future__ import annotations

import calendar
import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from askyourdata import data_engine as de
from askyourdata.data_engine import Condition
from askyourdata.text_matching import (
    _STOPWORDS,
    ROW_WORDS,
    ClarificationNeeded,
    ColumnMatcher,
    UnsupportedRequest,
    _col,
    _contains,
    _join,
    _singular,
    fmt,
    words,
)

PLACEHOLDER = "datetok"
_PLACEHOLDER_RE = re.compile(rf"^{PLACEHOLDER}(\d+)$")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
MAX_VALUES_PER_COLUMN = 5000

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11,
    "november": 11, "dec": 12, "december": 12,
}  # fmt: skip
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_YEAR = r"(?:19|20)\d{2}"


@dataclass(frozen=True)
class DateSpan:
    start: pd.Timestamp
    end: pd.Timestamp  # last moment of the span (inclusive)
    label: str


def _span(text: str, year: int, month: int | None = None, day: int | None = None) -> DateSpan:
    try:
        if day is not None:
            start = pd.Timestamp(year, month, day)
            end, label = start + pd.Timedelta(days=1), start.strftime("%Y-%m-%d")
        elif month is not None:
            start = pd.Timestamp(year, month, 1)
            end, label = start + pd.offsets.MonthBegin(1), start.strftime("%b %Y")
        else:
            start, end, label = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1), str(year)
    except ValueError:
        raise ClarificationNeeded(
            f"'{text}' is not a valid date. Please write dates like 2026-01-31."
        ) from None
    return DateSpan(start, end - pd.Timedelta(microseconds=1), label)


def _slash_date(m: re.Match) -> DateSpan:
    a, b, year = int(m[1]), int(m[2]), int(m[3])
    if a > 12 >= b:
        return _span(m[0], year, b, a)  # day first: 31/01/2026
    if b > 12 >= a:
        return _span(m[0], year, a, b)  # month first: 01/31/2026
    if a == b:
        return _span(m[0], year, a, a)
    raise ClarificationNeeded(
        f"'{m[0]}' could mean {a} {calendar.month_name[b]} or {calendar.month_name[a]} {b}. "
        "Please write the date as YYYY-MM-DD, e.g. 2026-01-31."
    )


_DATE_PATTERNS = [
    (re.compile(rf"\b({_YEAR})-(\d{{1,2}})-(\d{{1,2}})\b"),
     lambda m: _span(m[0], int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(rf"\b(\d{{1,2}})[/.](\d{{1,2}})[/.]({_YEAR})\b"), _slash_date),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})\.?,?\s+({_YEAR})\b"),
     lambda m: _span(m[0], int(m[3]), _MONTHS[m[2]], int(m[1]))),
    (re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+({_YEAR})\b"),
     lambda m: _span(m[0], int(m[3]), _MONTHS[m[1]], int(m[2]))),
    (re.compile(rf"\b({_MONTH})\.?,?\s+({_YEAR})\b"), lambda m: _span(m[0], int(m[2]), _MONTHS[m[1]])),
    (re.compile(rf"\b({_YEAR})-(\d{{1,2}})\b"), lambda m: _span(m[0], int(m[1]), int(m[2]))),
]  # fmt: skip
# A bare year only counts as a date after a date word ("in 2026"), so numbers
# such as "price greater than 2000" are left alone.
_YEAR_RE = re.compile(rf"\b(?P<pre>in|during|since|after|before|until|till|of|for|from)\s+({_YEAR})\b")
_RELATIVE_RE = re.compile(
    r"\b(?:last|this|previous|past|current)\s+(?:day|week|month|quarter|year)s?\b|\byesterday\b|\btoday\b"
)
_TWO_DATES_RE = re.compile(r"\bbetween\s+(?:two|2)\s+dates?\b")


def extract_dates(raw: str) -> tuple[str, list[DateSpan]]:
    """Replace dates in the (lower-case) question with placeholders like "datetok0"."""
    if _RELATIVE_RE.search(raw):
        raise UnsupportedRequest(
            "Relative dates such as 'last month' are not supported yet. Please give dates, "
            "e.g. 'between 2026-03-01 and 2026-03-31' or 'in March 2026'."
        )
    if _TWO_DATES_RE.search(raw):
        raise ClarificationNeeded(
            "Which two dates? For example: 'orders between 2026-01-01 and 2026-01-31'."
        )
    spans: list[DateSpan] = []

    def keep(span: DateSpan, prefix: str = "") -> str:
        spans.append(span)
        return f"{prefix} {PLACEHOLDER}{len(spans) - 1} "

    for pattern, build in _DATE_PATTERNS:
        raw = pattern.sub(lambda m, build=build: keep(build(m)), raw)
    raw = _YEAR_RE.sub(lambda m: keep(_span(m[2], int(m[2])), m["pre"]), raw)
    return raw, spans


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_OPERATORS = sorted(
    [
        ("greater than or equal to", de.GREATER_EQUAL, True),
        ("more than or equal to", de.GREATER_EQUAL, True),
        ("less than or equal to", de.LESS_EQUAL, True),
        ("at least", de.GREATER_EQUAL, True),
        ("no less than", de.GREATER_EQUAL, True),
        ("at most", de.LESS_EQUAL, True),
        ("no more than", de.LESS_EQUAL, True),
        ("greater than", de.GREATER, True),
        ("more than", de.GREATER, True),
        ("higher than", de.GREATER, True),
        ("larger than", de.GREATER, True),
        ("bigger than", de.GREATER, True),
        ("less than", de.LESS, True),
        ("fewer than", de.LESS, True),
        ("lower than", de.LESS, True),
        ("smaller than", de.LESS, True),
        ("equal to", de.EQUALS, True),
        ("equals", de.EQUALS, True),
        ("above", de.GREATER, False),
        ("over", de.GREATER, False),
        ("exceeding", de.GREATER, False),
        ("below", de.LESS, False),
        ("under", de.LESS, False),
        ("exactly", de.EQUALS, False),
    ],
    key=lambda op: -len(op[0].split()),
)
_OPERATORS = [(tuple(text.split()), op, strong) for text, op, strong in _OPERATORS]

# Words that join a filter to the rest of the question; removed with the filter.
_CONNECTORS = frozenset(
    {
        "in", "from", "for", "with", "where", "whose", "of", "the", "at", "by", "via",
        "using", "paid", "pay", "made", "placed", "is", "are", "was", "were", "that",
        "which", "only", "to", "on", "a", "an", "having", "through", "sold",
    }
)  # fmt: skip
_COLUMN_GLUE = frozenset({"is", "are", "was", "were", "of", "has", "have", "having", "with", "a"})
_AFTER_VALUE = frozenset({"by", "per", "in", "from", "with", "where", "for", "and", "or"})

_QUESTION_WORDS = frozenset(
    words(
        "total sum overall average avg mean median minimum min maximum max highest lowest "
        "largest smallest biggest greatest count number most least fewest top bottom best "
        "worst first last trend trends time day week month quarter year daily weekly monthly "
        "quarterly yearly annual distribution histogram spread breakdown frequency missing null "
        "blank empty correlation correlated relationship related overview summary describe "
        "compare between among across different unique distinct each every all how many much "
        "which what show list find display plot chart graph calculate compute get give tell "
        "value values row rows column columns dataset data table net gross revenue sales "
        "amount discount percent fraction calculated common frequent popular used often "
        "group grouped split broken down there any do we have"
    )
)
_RESERVED = frozenset(_STOPWORDS | ROW_WORDS | _QUESTION_WORDS | _CONNECTORS)
_UNKNOWN_VALUE_TRIGGERS = frozenset({"from", "in", "where", "paid", "via", "at"})


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterSpec:
    """One filter as shown to the user, with the validated conditions behind it."""

    description: str  # e.g. "`City` is Pune"
    conditions: tuple[Condition, ...]

    @property
    def columns(self) -> set:
        return {c.column for c in self.conditions}


@dataclass(frozen=True)
class ParsedFilters:
    remainder: str  # the question without the filter parts
    conditions: tuple[Condition, ...]
    descriptions: tuple[str, ...]  # e.g. "`City` is Pune"
    specs: tuple[FilterSpec, ...] = ()


class FilterParser:
    def __init__(self, df: pd.DataFrame, matcher: ColumnMatcher, dates: Sequence[DateSpan]) -> None:
        self.df = df
        self.matcher = matcher
        self.dates = list(dates)
        self._index: dict[tuple[str, ...], list[tuple[Any, Any]]] | None = None

    # -- helpers ----------------------------------------------------------------

    def _value_index(self) -> dict[tuple[str, ...], list[tuple[Any, Any]]]:
        """Words of every text value -> [(column, original value)]."""
        if self._index is None:
            self._index = {}
            for column in self.df.columns:
                if self.matcher.kinds.get(column) != de.CATEGORICAL:
                    continue
                values = self.df[column].dropna().unique()
                if len(values) > MAX_VALUES_PER_COLUMN:
                    continue
                for value in values:
                    key = words(value)
                    if not key or len("".join(key)) < 2 or all(w in _RESERVED for w in key):
                        continue
                    pairs = self._index.setdefault(key, [])
                    if (column, value) not in pairs:
                        pairs.append((column, value))
        return self._index

    def _mark(self, start: int, end: int) -> None:
        for i in range(start, end):
            self.used[i] = True

    def _free(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.tokens) and not any(self.used[start:end])

    def _absorb_left(self, i: int, column: Any = None) -> None:
        """Mark connector words (and the column's name) just before token ``i``."""
        j = i - 1
        while j >= 0 and not self.used[j] and self.sing[j] in _CONNECTORS:
            self.used[j] = True
            j -= 1
        if column is not None:
            cw = words(column)
            start = j - len(cw) + 1
            if self._free(start, j + 1) and tuple(self.sing[start : j + 1]) == cw:
                self._mark(start, j + 1)
                self._absorb_left(start)

    def _column_mention(self, start: int, end: int) -> Any | None:
        """A column whose name is exactly tokens[start:end]."""
        if not self._free(start, end) or start >= end:
            return None
        found = self.matcher.candidates(tuple(self.sing[start:end]))
        return found[0].column if found and found[0].score >= 90 else None

    def _column_before(self, end: int) -> tuple[list[Any], int, int]:
        """Best column named just before token ``end``: (columns, start, score)."""
        j = end - 1
        while j >= 0 and not self.used[j] and self.sing[j] in _COLUMN_GLUE:
            j -= 1
        best: tuple[list[Any], int, int] = ([], end, 0)
        for length in (3, 2, 1):
            start = j - length + 1
            if not self._free(start, j + 1):
                continue
            found = [c for c in self.matcher.candidates(words(" ".join(self.tokens[start : j + 1])))
                     if c.score >= 60 and c.column in self.df.columns]  # fmt: skip
            if found and found[0].score > best[2]:
                top = [c.column for c in found if c.score == found[0].score]
                best = (top, start, found[0].score)
        return best

    def _numeric_column(self, columns: list[Any], what: str) -> Any:
        numeric = [c for c in columns if self.matcher.kinds.get(c) == de.NUMERIC]
        if not numeric:
            raise ClarificationNeeded(
                f"{_col(columns[0])} is not a numeric column, so it can't be {what}. "
                f"Numeric columns: {_join(self.matcher.names({de.NUMERIC}))}.",
                self.matcher.names({de.NUMERIC}),
            )
        if len(numeric) > 1:
            raise ClarificationNeeded(
                f"Which column should be {what}: {_join(numeric)}?", numeric
            )
        return numeric[0]

    def _ask_for_column(self, what: str) -> None:
        numeric = [c for c in self.matcher.names({de.NUMERIC}) if c in self.df.columns]
        raise ClarificationNeeded(
            f"Which column should be {what}? Numeric columns: {_join(numeric)}.", numeric
        )

    def _add(self, conditions: list[Condition], description: str) -> None:
        self.conditions.extend(conditions)
        self.descriptions.append(description)
        self.specs.append(FilterSpec(description, tuple(conditions)))

    # -- date filters -----------------------------------------------------------

    def _date_column(self) -> Any:
        dates = [c for c in self.df.columns if self.matcher.kinds.get(c) == de.DATETIME]
        if not dates:
            raise UnsupportedRequest(
                "The question contains a date, but this dataset has no date column to filter on."
            )
        if len(dates) == 1:
            return dates[0]
        mentioned = [c for c in dates if _contains(self.sing, words(c))]
        if len(mentioned) == 1:
            return mentioned[0]
        raise ClarificationNeeded(
            f"Which date column should the date filter use: {_join(dates)}? "
            "Please include its name in the question.",
            dates,
        )

    def _span_at(self, i: int) -> DateSpan | None:
        m = _PLACEHOLDER_RE.match(self.tokens[i]) if 0 <= i < len(self.tokens) else None
        return self.dates[int(m[1])] if m else None

    def _parse_dates(self) -> None:
        for i in range(len(self.tokens)):
            span = self._span_at(i)
            if span is None or self.used[i]:
                continue
            column = self._date_column()
            prev = self.sing[i - 1] if i > 0 and not self.used[i - 1] else ""
            nxt = self.sing[i + 1] if i + 1 < len(self.tokens) else ""
            second = self._span_at(i + 2)
            c = _col(column)
            if prev in ("between", "from") and nxt in ("and", "to", "until", "till", "through") and second:
                if second.end < span.start:
                    raise ClarificationNeeded(
                        f"The date range {span.label} to {second.label} ends before it starts."
                    )
                self._add([Condition(column, de.GREATER_EQUAL, span.start),
                           Condition(column, de.LESS_EQUAL, second.end)],
                          f"{c} from {span.label} to {second.label}")  # fmt: skip
                self._mark(i - 1, i + 3)
                start = i - 1
            elif prev == "after":
                self._add([Condition(column, de.GREATER, span.end)], f"{c} after {span.label}")
                self._mark(i - 1, i + 1)
                start = i - 1
            elif prev in ("since", "from", "starting"):
                self._add([Condition(column, de.GREATER_EQUAL, span.start)], f"{c} from {span.label} onwards")
                self._mark(i - 1, i + 1)
                start = i - 1
            elif prev == "before":
                self._add([Condition(column, de.LESS, span.start)], f"{c} before {span.label}")
                self._mark(i - 1, i + 1)
                start = i - 1
            elif prev in ("until", "till", "through"):
                self._add([Condition(column, de.LESS_EQUAL, span.end)], f"{c} up to {span.label}")
                self._mark(i - 1, i + 1)
                start = i - 1
            else:
                self._add([Condition(column, de.GREATER_EQUAL, span.start),
                           Condition(column, de.LESS_EQUAL, span.end)],
                          f"{c} in {span.label}")  # fmt: skip
                self._mark(i, i + 1)
                start = i
            self._absorb_left(start, column)

    # -- number filters -----------------------------------------------------------

    def _number_at(self, i: int) -> float | None:
        if 0 <= i < len(self.tokens) and not self.used[i] and _NUMBER_RE.match(self.tokens[i]):
            value = float(self.tokens[i])
            return int(value) if value.is_integer() else value
        return None

    def _parse_between_numbers(self) -> None:
        for i, word in enumerate(self.sing):
            if word != "between" or self.used[i]:
                continue
            low, high = self._number_at(i + 1), self._number_at(i + 3)
            if low is None or high is None or self.sing[i + 2] != "and":
                continue
            columns, start, _ = self._column_before(i)
            what = f"between {fmt(low)} and {fmt(high)}"
            if not columns:
                self._ask_for_column(what)
            column = self._numeric_column(columns, what)
            if low > high:
                low, high = high, low
            self._add([Condition(column, de.GREATER_EQUAL, low), Condition(column, de.LESS_EQUAL, high)],
                      f"{_col(column)} between {fmt(low)} and {fmt(high)}")  # fmt: skip
            self._mark(start, i + 4)
            self._absorb_left(start)

    def _parse_comparisons(self) -> None:
        i = 0
        while i < len(self.tokens):
            match = next(
                (op for op in _OPERATORS
                 if self._free(i, i + len(op[0])) and tuple(self.sing[i : i + len(op[0])]) == op[0]),
                None,
            )  # fmt: skip
            if match is None:
                i += 1
                continue
            phrase, op, strong = match
            k = i + len(phrase)
            what = f"{op} " + "{value}"

            number = self._number_at(k)
            if number is not None:  # "quantity greater than 3"
                columns, start, _ = self._column_before(i)
                if not columns:
                    self._ask_for_column(what.format(value=fmt(number)))
                column = self._numeric_column(columns, what.format(value=fmt(number)))
                self._add([Condition(column, op, number)], f"{_col(column)} {op} {fmt(number)}")
                self._mark(start, k + 1)
                self._absorb_left(start)
                i = k + 1
                continue

            after = next(  # "above quantity 2"
                ((length, col) for length in (3, 2, 1)
                 if self._number_at(k + length) is not None
                 and (col := self._column_mention(k, k + length)) is not None),
                None,
            )  # fmt: skip
            if after:
                length, column = after
                number = self._number_at(k + length)
                column = self._numeric_column([column], what.format(value=fmt(number)))
                self._add([Condition(column, op, number)], f"{_col(column)} {op} {fmt(number)}")
                self._mark(i, k + length + 1)
                self._absorb_left(i)
                i = k + length + 1
                continue

            columns, _, score = self._column_before(i)
            numeric_before = [c for c in columns if self.matcher.kinds.get(c) == de.NUMERIC]
            if strong and numeric_before:  # "quantity greater than abc"
                raise ClarificationNeeded(
                    f"Please give a number after '{' '.join(phrase)}' for {_col(numeric_before[0])}."
                )
            i = k

    # -- text value filters -----------------------------------------------------------

    def _parse_values(self) -> None:
        index = self._value_index()
        if not index:
            return
        longest = max(len(k) for k in index)
        chosen: dict[Any, list[Any]] = {}
        i = 0
        while i < len(self.tokens):
            hit = next(
                ((length, tuple(self.sing[i : i + length])) for length in range(longest, 0, -1)
                 if self._free(i, i + length) and tuple(self.sing[i : i + length]) in index),
                None,
            )  # fmt: skip
            if hit is None:
                i += 1
                continue
            length, key = hit
            end = i + length
            pairs = index[key]
            columns = list(dict.fromkeys(c for c, _ in pairs))
            if len(columns) > 1:
                named = [c for c in columns if self._names_column(c, i, end)]
                if len(named) != 1:
                    raise ClarificationNeeded(
                        f"'{' '.join(self.tokens[i:end])}' appears in more than one column: "
                        f"{_join(columns)}. Which one did you mean? Please include the column name.",
                        columns,
                    )
                columns = named
            column = columns[0]
            value = next(v for c, v in pairs if c == column)
            chosen.setdefault(column, [])
            if value not in chosen[column]:
                chosen[column].append(value)
            self._mark(i, end)
            self._absorb_right(end, column)
            self._absorb_left(i, column)
            i = end

        for column, values in chosen.items():
            if len(values) == 1:
                self._add([Condition(column, de.EQUALS, values[0])], f"{_col(column)} is {values[0]}")
            else:
                listed = " or ".join(map(str, values))
                self._add([Condition(column, de.IS_ONE_OF, list(map(str, values)))],
                          f"{_col(column)} is {listed}")  # fmt: skip
        # "Pune or Mumbai": the joining word belongs to the filter too
        for j in range(1, len(self.tokens) - 1):
            if self.sing[j] in ("or", "and") and self.used[j - 1] and self.used[j + 1]:
                self.used[j] = True

    def _names_column(self, column: Any, start: int, end: int) -> bool:
        """Is the column named next to the value ("work city is Pune", "Pune category")?"""
        cw = words(column)
        n = len(cw)
        while start > 0 and self.sing[start - 1] in ("is", "are", "was", "were", "equal", "equals"):
            start -= 1
        before = tuple(self.sing[max(0, start - n) : start])
        after = tuple(self.sing[end : end + n])
        return cw in (before, after)

    def _absorb_right(self, end: int, column: Any) -> None:
        cw = words(column)
        if self._free(end, end + len(cw)) and tuple(self.sing[end : end + len(cw)]) == cw:
            self._mark(end, end + len(cw))  # "Electronics category"
            return
        if self._free(end, end + 1):
            nxt = self.sing[end]
            follower = self.sing[end + 1] if end + 1 < len(self.tokens) else None
            noun = nxt in ROW_WORDS or any(
                words(c) == (nxt,) and self.matcher.kinds.get(c) == de.CATEGORICAL
                for c in self.df.columns
            )
            if noun and (follower is None or follower in _AFTER_VALUE):
                self.used[end] = True  # "Electronics products", "Pune orders"

    # -- leftovers -------------------------------------------------------------------

    def _check_leftovers(self) -> None:
        """Refuse filter-looking words that could not be understood."""
        known = set(_RESERVED)
        for column in self.df.columns:
            known |= set(words(column))
        for key in self._value_index():
            known |= set(key)
        for j in range(len(self.tokens) - 1):
            if self.used[j] or self.used[j + 1] or self.sing[j] not in _UNKNOWN_VALUE_TRIGGERS:
                continue
            nxt = self.sing[j + 1]
            if nxt in known or _NUMBER_RE.match(nxt) or _PLACEHOLDER_RE.match(nxt):
                continue
            values = {" ".join(k): k for k in self._value_index()}
            close = difflib.get_close_matches(nxt, list(values), n=3, cutoff=0.7)
            originals = [str(self._value_index()[values[c]][0][1]) for c in close]
            hint = f" Did you mean {' or '.join(originals)}?" if originals else ""
            raise ClarificationNeeded(
                f"I couldn't find '{self.tokens[j + 1]}' as a value in any text column, so I "
                f"can't filter by it.{hint}",
                originals,
            )

    # -- entry point --------------------------------------------------------------------

    def parse(self, q: str) -> ParsedFilters:
        self.tokens = q.split()
        self.sing = [_singular(t) for t in self.tokens]
        self.used = [False] * len(self.tokens)
        self.conditions: list[Condition] = []
        self.descriptions: list[str] = []
        self.specs: list[FilterSpec] = []

        self._parse_dates()
        self._parse_between_numbers()
        self._parse_comparisons()
        self._parse_values()
        self._check_leftovers()

        rest = [t for t, u in zip(self.tokens, self.used) if not u]
        return ParsedFilters(
            _tidy(rest), tuple(self.conditions), tuple(self.descriptions), tuple(self.specs)
        )


def _tidy(tokens: list[str]) -> str:
    """Remove words left dangling by removing filters."""
    tokens = list(tokens)
    edge = _CONNECTORS | {"and", "or"}
    while tokens and _singular(tokens[-1]) in edge:
        tokens.pop()
    while tokens and tokens[0] in ("and", "or"):
        tokens.pop(0)
    if (
        len(tokens) > 2
        and tokens[-2] in ("for", "of", "in", "from", "among", "across")
        and _singular(tokens[-1]) in ROW_WORDS
    ):
        tokens = tokens[:-2]  # "compare payment methods for orders"
    return " ".join(tokens)
