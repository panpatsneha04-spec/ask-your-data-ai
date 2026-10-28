"""Answer plain-English questions about a dataset, deterministically.

How a question is answered (``answer_question``):
1. Safety screen: requests to change data, code-like or SQL-like text and
   forecasting questions are refused before anything else happens.
2. The question is normalised (lower case, punctuation removed, polite
   openings such as "please show me" dropped).
3. Fixed patterns decide the intent: overview, missing values, correlation,
   trend, distribution, top N, "which X has the most Y", count, group-by or
   a single aggregate.
4. Column phrases from the question are matched to the dataset's REAL
   columns (``ColumnMatcher``). Unclear matches lead to a clarifying
   question instead of a guess; missing columns are reported.
5. ONE approved ``data_engine`` function runs with the validated columns,
   and the answer text is built only from the numbers it returned.

No LLM, eval/exec, generated code or SQL is involved: the question text can
only choose between the handlers below and supply validated column names
and numbers.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from askyourdata import charts, exports, metrics, nl_filters, profiling
from askyourdata import data_engine as de
from askyourdata.data_engine import AnalysisError, ChartData

logger = logging.getLogger(__name__)

# Answer statuses
ANSWER = "answer"
CLARIFY = "clarify"
UNSUPPORTED = "unsupported"
ERROR = "error"
NO_DATA = "no_data"

MAX_QUESTION_LENGTH = 300
MAX_TOP_N = 100
DEFAULT_TOP_N = 5
RANKING_ROWS = 10
CATEGORY_CHART_LIMIT = 20

CAPABILITIES = (
    "I can answer questions about: the size and columns of the dataset, totals / "
    "averages / minimum / maximum of a column, comparisons by group, top N rankings, "
    "trends over time, distributions, missing values and correlations."
)


@dataclass(frozen=True, eq=False)
class Answer:
    """The result of a question: text plus optional table, chart and details."""

    status: str
    text: str
    intent: str | None = None
    table: pd.DataFrame | None = None
    figure: go.Figure | None = None
    details: tuple[str, ...] = ()  # which columns and calculation were used
    options: tuple[str, ...] = ()  # column choices when clarification is needed
    suggestions: tuple[str, ...] = ()  # example questions that would work
    rephrasings: tuple[str, ...] = ()  # complete questions to pick when clarifying
    context: AnswerContext | None = None  # structured facts for follow-up questions
    plan: Any = None  # planner.PlanReport: the steps behind a multi-step answer

    @property
    def ok(self) -> bool:
        return self.status == ANSWER


@dataclass(frozen=True, eq=False)
class AnswerContext:
    """What an answer was based on, so a follow-up question can build on it.

    Everything here comes from validated router decisions (real column names,
    validated conditions, the table that was shown), never from answer text.
    """

    question: str  # the question as asked
    core: str  # normalised question without its filters (can be answered again)
    intent: str
    filters: tuple = ()  # nl_filters.FilterSpec objects applied first
    table: pd.DataFrame | None = None  # the table that was shown
    entity: Any = None  # the column being ranked / grouped / counted
    groups: tuple = ()  # grouping columns (group comparisons)
    value: Any = None  # the numeric column measured (may be a calculated metric)
    metric: str | None = None  # metric key when ``value`` is calculated
    agg: str | None = None  # sum / mean / count / ...
    n: int | None = None  # N of a top-N result
    largest: bool = True  # largest-first ranking?
    winner: Any = None  # the single top value of a ranking (None when tied)
    date: Any = None  # date column of a trend
    freq: str | None = None  # time period of a trend (Day / Week / Month / ...)
    chart_x: Any = None  # columns of the chart that was (or could be) drawn
    chart_y: Any = None
    chart_kind: str | None = None  # "bar", "line", "histogram", ...


from askyourdata.text_matching import (  # noqa: F401  (re-exported for callers and tests)
    _STOPWORDS,
    ROW_WORDS,
    ClarificationNeeded,
    ColumnMatcher,
    UnsupportedRequest,
    _cap,
    _col,
    _contains,
    _join,
    _remove,
    _strip,
    fmt,
    looks_like_id,
    words,
)

_Clarify = ClarificationNeeded
_Unsupported = UnsupportedRequest


# ---------------------------------------------------------------------------
# Question patterns
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(
    r"__|;|`|\bimport\s|\beval\s*\(|\bexec\s*\(|\blambda\b|\bsubprocess\b|\bos\.\w+"
    r"|\bselect\s.+\sfrom\s|--|/\*|<\s*script"
)
_WRITE_RE = re.compile(
    r"\b(?:drop|delete|truncate|insert|update|alter|remove|overwrite|rename|modify|edit"
    r"|replace|save|create table)\b"
)
_FORECAST_RE = re.compile(
    r"\b(?:predict\w*|forecast\w*|future|next (?:day|week|month|quarter|year)"
    r"|will (?:be|sell|increase|decrease|grow)|why)\b"
)
_LEAD_RE = re.compile(
    r"^(?:please|kindly|can you|could you|would you|will you|show me|show|tell me|give me"
    r"|find me|find|list|display|plot|draw|chart|calculate|compute|get me|get|what is|what s"
    r"|whats|what are|what were|what was|i want to know|i want to see|i d like to see"
    r"|i would like to see|let me see|and|so|ok|okay|hey|hi)\b\s*"
)

_GROUP_SEP = (
    r"(?:grouped by|broken down by|split by|for each|for every|in each|by|per|across)"
)
_GROUP_SEP_RE = re.compile(rf"\s{_GROUP_SEP}\s")
_GROUP_RE = re.compile(rf"^(?P<measure>.+?)\s+{_GROUP_SEP}\s+(?P<group>.+)$")

_MISSING_RE = re.compile(
    r"\b(?:missing|null|nulls|blank|blanks|nan|empty (?:value|values|cell|cells)|incomplete)\b"
)
_MISSING_COLUMN_RE = re.compile(
    r"\b(?:missing|null|nulls|blank|blanks|empty)\s+(?:values?\s+|cells?\s+|entries\s+)?"
    r"(?:in|for|of)\s+(?P<col>.+)$"
)
_CORR_RE = re.compile(r"\bcorrelat\w*|\brelationships? between\b|\brelated (?:to|with)\b")
_CORR_PAIR_RES = (
    re.compile(r"\bbetween (?P<a>.+?) and (?P<b>.+)$"),
    re.compile(r"^(?:is |are )?(?:the )?(?P<a>.+?) (?:correlated|related) (?:with|to) (?P<b>.+)$"),
)
_ROWCOUNT_RE = re.compile(
    r"\bhow many (?:rows|records|entries|lines|observations)\b"
    r"|\b(?:number|count) of (?:rows|records|entries|observations)\b|\brow count\b"
    r"|\bhow (?:big|large) is\b|\bsize of\b|\bshape of\b|\bhow many columns\b"
    r"|\b(?:what|which) columns\b|\blist (?:the |all )?columns\b|\bcolumn names\b"
)
_DESCRIBE_RE = re.compile(r"\b(?:overview|summary|summari[sz]e|describe)\b(?:\s+of)?(?P<rest>.*)$")
_DESCRIBE_FILLER = frozenset(
    {"dataset", "data", "table", "file", "it", "statistic", "stat", "info", "information"}
)

_FREQ_WORDS = {
    "day": "Day", "daily": "Day", "week": "Week", "weekly": "Week", "month": "Month",
    "monthly": "Month", "quarter": "Quarter", "quarterly": "Quarter", "year": "Year",
    "yearly": "Year", "annual": "Year", "annually": "Year",
}  # fmt: skip
_TREND_RE = re.compile(
    r"\btrends?\b|\bover time\b|\b(?:by|per|each|every|across) (?:the )?"
    r"(?:day|week|month|quarter|year)s?\b|\b(?:daily|weekly|monthly|quarterly|yearly|annual|annually)\b"
)
_TREND_REMOVE_RE = re.compile(
    r"\b(?:trends?|over time|how (?:has|have|did|does|do)|chang(?:e|ed|es|ing)|evolv\w*"
    r"|(?:by|per|each|every|across) (?:the )?(?:day|week|month|quarter|year)s?"
    r"|daily|weekly|monthly|quarterly|yearly|annual|annually|of|in|the|for)\b"
)

_DIST_RE = re.compile(
    r"\b(?:distribution|histogram|spread|breakdown|frequency|frequencies|split)"
    r"(?:\s+(?:of|for|in))?\s+(?P<col>.+)$"
)
_DIST_SUFFIX_RE = re.compile(r"^(?P<col>.+?)\s+(?:distribution|histogram|breakdown)$")

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}  # fmt: skip
_SMALL_FIRST = frozenset({"bottom", "worst", "lowest", "smallest", "last", "least", "fewest"})
_TOP_RE = re.compile(
    r"\b(?P<dir>top|bottom|best|worst|highest|lowest|largest|smallest|biggest|first|last)\s+"
    rf"(?P<n>-?\d+(?:\.\d+)?|{'|'.join(_NUMBER_WORDS)})\s+(?P<rest>.+)$"
)
_TOP_DEFAULT_RE = re.compile(r"\b(?P<dir>top|bottom)\s+(?P<rest>[a-z].+)$")
_TOP_SPLIT_RE = re.compile(
    r"\s+(?:by|in terms of|based on|ranked by|according to|with the (?:most|highest)|for)\s+"
)

_WHICH_RE = re.compile(
    r"\b(?:which|what)\s+(?P<entity>.+?)\s+(?:has|have|had|with|sold|sells|gets|got|"
    r"generated|generates|made|makes|brought|is|are|was|were)?\s*(?:the\s+)?"
    r"(?P<dir>most|highest|largest|biggest|greatest|least|lowest|fewest|smallest)\b\s*(?P<rest>.*)$"
)
_MOST_COMMON_RE = re.compile(
    r"\b(?P<dir>most|least)\s+(?:common|frequent|popular|used|often)\s+(?P<entity>.+)$"
)
_HOW_MANY_RE = re.compile(
    r"^(?:how many|total number of|number of|count of|count)\s+"
    r"(?:different\s+|unique\s+|distinct\s+)?(?P<x>.+?)"
    r"(?:\s+(?:are there|is there|are in .+|in (?:the|this) (?:data|dataset|table)|do we have"
    r"|exist|does it have|there are))?$"
)
_AGG_RE = re.compile(
    r"^(?:the\s+)?(?P<agg>average|avg|mean|median|total|sum|overall|minimum|min|lowest|smallest"
    r"|maximum|max|highest|largest|biggest)\s+(?:of\s+)?(?:the\s+)?(?P<measure>.+)$"
)

_COMPARE_RE = re.compile(
    r"^compare\s+(?:the\s+)?(?P<measure>.+?)\s+(?:across|between|among|by|per|for each|for every"
    r"|in each|for all)\s+(?:the\s+|all\s+|different\s+)*(?P<group>.+)$"
)
_COMPARE_ONLY_RE = re.compile(r"^compare\s+(?:the\s+|all\s+|different\s+)*(?P<group>.+)$")
_DISCOUNT_HINT_RE = re.compile(
    r"\b(?:with\s+|treating\s+|treat\s+)?(?:the\s+)?discounts?\s+(?:values?\s+)?(?:as|in|is|are)\s+"
    r"(?:a\s+|an\s+)?(?P<scale>percent|percentages?|fractions?|decimals?)\b"
)
_REVENUE_WORD_RE = re.compile(r"\b(?:sales amount|revenue|sales|sale|turnover|income)\b", re.I)

_LIST_WORDS = frozenset({"show", "list", "find", "display", "get", "all", "only", "me", "give"})

_AGG_WORDS = {
    "total": "sum", "sum": "sum", "overall": "sum", "average": "mean", "avg": "mean",
    "mean": "mean", "median": "median", "minimum": "min", "min": "min", "lowest": "min",
    "smallest": "min", "maximum": "max", "max": "max", "highest": "max", "largest": "max",
    "biggest": "max", "count": "count", "number": "count",
}  # fmt: skip
_AGG_LABELS = {
    "sum": "total", "mean": "average", "median": "median", "min": "minimum",
    "max": "maximum", "count": "count of", "nunique": "number of different values in",
}  # fmt: skip


def _split_agg(phrase: Sequence[str]) -> tuple[str | None, tuple[str, ...]]:
    """("average", "unit", "price") -> ("mean", ("unit", "price"))."""
    items = list(_strip(phrase))
    agg = None
    if items and items[0] in _AGG_WORDS:
        agg = _AGG_WORDS[items.pop(0)]
    return agg, _strip(items)


def _parse_top_n(raw: str) -> int:
    value = _NUMBER_WORDS.get(raw)
    if value is None:
        try:
            number = float(raw)
        except ValueError:
            number = math.nan
        if not number.is_integer() or not 1 <= number <= MAX_TOP_N:
            raise AnalysisError(
                f"N must be a whole number between 1 and {MAX_TOP_N} (got '{raw}')."
            )
        value = int(number)
    return value


@dataclass(frozen=True)
class _Prepared:
    q: str  # normalised question text
    discount_scale: str | None = None  # "percent" / "fraction" if the user said so
    dates: list = field(default_factory=list)  # nl_filters.DateSpan for each "datetokN"


def _prepare(question: Any) -> _Prepared:
    if not isinstance(question, str) or not question.strip():
        raise _Unsupported("Please type a question about your data.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise _Unsupported(f"Please keep questions under {MAX_QUESTION_LENGTH} characters.")

    raw = question.strip().lower()
    if _CODE_RE.search(raw):
        raise _Unsupported(
            "That looks like code or SQL. I only answer plain-English questions and never run "
            "code or SQL. To run a read-only SELECT query, use the Database Explorer tab."
        )
    if _WRITE_RE.search(raw):
        raise _Unsupported(
            "I can only read and analyse the data. I can't change, delete or save anything."
        )
    if _FORECAST_RE.search(raw):
        raise _Unsupported(
            "I can only describe the data that is there. I can't predict the future or "
            "explain causes."
        )

    discount_scale = None
    if m := _DISCOUNT_HINT_RE.search(raw):
        scale = m.group("scale")
        discount_scale = (
            metrics.DISCOUNT_PERCENT if scale.startswith("percent") else metrics.DISCOUNT_FRACTION
        )
        raw = raw[: m.start()] + " " + raw[m.end() :]

    # Comparison symbols become words; "1,000" becomes "1000".
    for symbol, text in ((">=", " at least "), ("<=", " at most "), (">", " greater than "),
                         ("<", " less than "), ("=", " equals ")):  # fmt: skip
        raw = raw.replace(symbol, text)
    raw = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw)
    raw, dates = nl_filters.extract_dates(raw)  # before punctuation is removed

    q = re.sub(r"[^a-z0-9_.\-\s]", " ", raw.replace("'", " "))
    q = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", q)  # keep only decimal points, e.g. "top 2.5"
    q = q.replace("_", " ")
    q = re.sub(r"(?<!\s)-|-(?!\d)", " ", q)  # keep "-" only for negative numbers
    q = re.sub(r"\s+", " ", q).strip()
    while (stripped := _LEAD_RE.sub("", q, count=1)) != q:
        q = stripped
    return _Prepared(q=q, discount_scale=discount_scale, dates=dates)


# ---------------------------------------------------------------------------
# Example questions built from the current dataset's columns
# ---------------------------------------------------------------------------


def _spoken(column: Any) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", str(column)).split())


_looks_like_id = looks_like_id


def _filter_example_value(df: pd.DataFrame, column: Any) -> str | None:
    """The most common value of a text column, if it can be used in a filter."""
    counts = df[column].dropna().astype(str).value_counts()
    if counts.empty:
        return None
    value = counts.index[0]
    key = words(value)
    if not key or len("".join(key)) < 2 or all(w in nl_filters._RESERVED for w in key):
        return None
    return value


def _metric_example(df: pd.DataFrame) -> str | None:
    """"net revenue" or "gross amount" if it can be calculated without questions."""
    for key, label in ((metrics.NET_REVENUE, "net revenue"), (metrics.GROSS_AMOUNT, "gross amount")):
        try:
            metrics.compute_metric(df, key)
        except Exception:
            continue
        return label
    return None


def suggest_questions(df: pd.DataFrame | None, limit: int = 10) -> list[str]:
    """Example questions that use the dataset's real column names."""
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return []
    all_numbers = de.numeric_columns(df)
    numbers = [c for c in all_numbers if not _looks_like_id(df, c)] or all_numbers
    categories = de.categorical_columns(df)
    dates = de.date_columns(df)
    num = _spoken(numbers[0]) if numbers else None
    cat = _spoken(categories[0]) if categories else None

    questions = ["How many rows are in this dataset?"]
    if num:
        questions.append(f"What is the average {num}?")
    if num and cat:
        questions.append(f"Show total {num} by {cat}")
        if (value := _filter_example_value(df, categories[0])) is not None:
            questions.append(f"Show total {num} for {value}")
    if cat and (metric := _metric_example(df)) and not any(
        metrics.match_metric(words(c)) for c in df.columns
    ):
        questions.append(f"Show {metric} by {cat}")
    if num and cat:
        questions.append(f"Find the top 5 {cat} by {num}")
    if cat:
        questions.append(f"Which {cat} has the most rows?")
    if dates:
        questions.append("What is the trend by month?")
    if cat or num:
        questions.append(f"Show the distribution of {cat or num}")
    questions.append("Are there any missing values?")
    if len(numbers) >= 2:
        questions.append("Show the correlation between numeric columns")
    return questions[:limit]


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


class _Router:
    def __init__(
        self, df: pd.DataFrame, question: str = "", discount_scale: str | None = None
    ) -> None:
        self.full_df = df  # never modified
        self.df = df  # rows in use; becomes a private copy if a metric is added
        self.question = question
        self.discount_scale = discount_scale
        self.matcher = ColumnMatcher(df)
        self.notes: list[str] = []  # how phrases were matched to columns
        self.metric_details: list[str] = []  # formulas and notes of calculated metrics
        self.metric_columns: dict[str, Any] = {}  # metric key -> column name in self.df
        self.metric_tables: dict[Any, pd.DataFrame] = {}  # metric column -> inputs + result
        self._own_copy = False
        self.filters: tuple[str, ...] = ()  # descriptions of filters applied first
        self.filter_specs: tuple = ()  # the FilterSpec objects behind them
        self.core = ""  # the question without its filters
        self.ctx: dict[str, Any] = {}  # structured facts recorded by the handler

    # -- helpers ----------------------------------------------------------------

    def _resolve(self, phrase: Sequence[str], role: str, kinds: Iterable[str] | None = None) -> Any:
        found = self.matcher.resolve(phrase, role, kinds)
        if found.score < 100:
            self.notes.append(
                f"'{' '.join(_strip(phrase))}' was matched to column {_col(found.column)} "
                f"({found.reason})."
            )
        return found.column

    def _measure(self, phrase: Sequence[str], role: str = "value to calculate") -> Any | None:
        """A numeric column, or None when the phrase means "count the rows".

        Metric names ("net revenue", "gross amount") become calculated columns,
        unless the dataset already has a column with that exact name.
        """
        phrase = _strip(phrase)
        if not phrase:
            return None
        found = self.matcher.candidates(phrase)
        metric = metrics.match_metric(phrase)
        existing = bool(found) and found[0].score >= 90
        if metric == metrics.GENERIC_REVENUE and not existing:
            if found and found[0].score >= 80:  # e.g. a "Revenue" or "Sales" column
                return self._resolve(phrase, role, {de.NUMERIC})
            self._revenue_clarify(phrase)
        if metric and not existing:
            return self._metric_column(metric)
        if metric:
            self.notes.append(
                f"The existing column {_col(found[0].column)} was used; nothing was recalculated."
            )
        strong = bool(found) and found[0].score >= 80
        if not strong and len(phrase) == 1 and phrase[0] in ROW_WORDS:
            self.notes.append(f"'{phrase[0]}' is not a column, so each row was counted as one.")
            return None
        return self._resolve(phrase, role, {de.NUMERIC})

    def _metric_column(self, key: str) -> Any:
        """Add a calculated metric column to a private copy of the rows in use."""
        if key in self.metric_columns:
            return self.metric_columns[key]
        try:
            # Calculated on the full dataset, so how discounts are stored is judged
            # from all values, then lined up with the rows in use.
            result = metrics.compute_metric(self.full_df, key, discount_scale=self.discount_scale)
        except metrics.DiscountScaleUnclear as exc:
            base = self.question.strip().rstrip(" ?.!")
            raise ClarificationNeeded(
                exc.message,
                exc.options,
                rephrasings=[f"{base} with discount as fraction", f"{base} with discount as percent"],
            ) from None

        name = result.label
        while name in self.full_df.columns:
            name += " *"
        if not self._own_copy:
            self.df = self.df.copy()
            self._own_copy = True
        self.df[name] = result.values.reindex(self.df.index)
        self.matcher.add_column(name, de.NUMERIC)
        self.metric_columns[key] = name
        self.metric_tables[name] = self.df[[*dict.fromkeys(result.columns.values()), name]]

        self.metric_details.append("Formula: " + result.formula)
        self.metric_details.extend(result.notes)
        unknown = int(self.df[name].isna().sum())
        if unknown:
            self.metric_details.append(
                f"{unknown} row(s) have a missing or invalid input, so their {name} is unknown; "
                "they are left out of totals (not counted as 0)."
            )
        similar = [c for c in self.full_df.columns if {"revenue", "sale", "amount"} & set(words(c))]
        if similar:
            self.metric_details.append(f"Existing column(s) {_join(similar)} were not used or changed.")
        return name

    def _revenue_clarify(self, phrase: Sequence[str]) -> None:
        """'revenue' without a revenue column: offer the metrics that can be calculated."""
        label = " ".join(phrase)
        labels = {metrics.NET_REVENUE: "net revenue", metrics.GROSS_AMOUNT: "gross amount"}
        formulas = {
            "net revenue": "net revenue = quantity × unit price × (1 − discount)",
            "gross amount": "gross amount = quantity × unit price",
        }
        options = []
        for key, option in labels.items():
            try:
                available = not metrics.missing_inputs(self.full_df, key)
            except ClarificationNeeded:
                available = True  # the inputs exist but need clarifying later
            if available:
                options.append(option)
        if not options:
            numeric = self.matcher.names({de.NUMERIC})
            raise ClarificationNeeded(
                f"I couldn't find a column matching '{label}' in this dataset. If you meant a "
                "money amount, it is not a column here, and it can't be calculated because "
                f"quantity and unit price columns are needed. Available numeric columns: {_join(numeric)}.",
                numeric,
            )
        rephrasings = []
        for option in options:
            new = _REVENUE_WORD_RE.sub(option, self.question, count=1)
            rephrasings.append(new if new != self.question else option)
        raise ClarificationNeeded(
            f"'{label}' is not a column here. I can calculate it, but it could mean "
            + " or ".join(formulas[o] for o in options)
            + ". Which one do you mean?",
            options,
            rephrasings=rephrasings,
        )

    def _answer(self, intent: str, text: str, *, calculation: str, columns: Iterable[str] = (),
                table: pd.DataFrame | None = None, figure: go.Figure | None = None) -> Answer:  # fmt: skip
        details = []
        used = [c for c in columns if c]
        if used:
            details.append("Columns used: " + "; ".join(used))
        details.append("Calculation: " + calculation)
        details += self.metric_details
        details += self.notes
        shown = exports.display_table(table)
        return Answer(
            ANSWER, text, intent=intent, table=shown, figure=figure, details=tuple(details),
            context=self._context(intent, shown),
        )  # fmt: skip

    def _context(self, intent: str, table: pd.DataFrame | None) -> AnswerContext:
        value = self.ctx.get("value")
        metric = next((key for key, column in self.metric_columns.items() if column == value), None)
        return AnswerContext(
            question=self.question, core=self.core, intent=intent, filters=self.filter_specs,
            table=table, metric=metric, **self.ctx,
        )  # fmt: skip

    # -- filters, then routing --------------------------------------------------------

    def answer(self, prepared: _Prepared, base_filters: Sequence = ()) -> Answer:
        """Apply filters (carried-over ones first), then answer the rest of the question."""
        parsed = nl_filters.FilterParser(self.full_df, self.matcher, prepared.dates).parse(prepared.q)
        self.core = parsed.remainder
        specs = (*base_filters, *parsed.specs)
        if not specs:
            return self.route(parsed.remainder)

        conditions = [condition for spec in specs for condition in spec.conditions]
        self.df = de.filter_rows(self.full_df, conditions)  # validated, no eval
        self._own_copy = True  # filter_rows returns a new DataFrame
        self.filters = tuple(spec.description for spec in specs)
        self.filter_specs = tuple(specs)
        summary = (
            f"**Filters:** {'; '.join(self.filters)} → **{fmt(len(self.df))} of "
            f"{fmt(len(self.full_df))} rows**."
        )
        filter_detail = (
            f"Filters applied before the calculation: {'; '.join(self.filters)} "
            f"({fmt(len(self.df))} of {fmt(len(self.full_df))} rows match; data_engine.filter_rows)."
        )
        if self.df.empty:
            empty = exports.display_table(self.df)
            return Answer(
                ANSWER, summary + " No rows match, so there is nothing to analyse.",
                intent="filter", table=empty, details=(filter_detail,),
                context=self._context("filter", empty),
            )  # fmt: skip
        if self._is_filter_only(parsed.remainder):
            return self._answer(
                "filter", summary, table=self.df,
                calculation="data_engine.filter_rows - the matching rows are listed",
            )  # fmt: skip

        answer = self.route(parsed.remainder)
        return replace(answer, text=f"{summary}\n\n{answer.text}", details=(filter_detail, *answer.details))

    def _is_filter_only(self, remainder: str) -> bool:
        """True for "show orders", "list products" etc. once filters are removed."""
        rest = [w for w in _strip(words(remainder)) if w not in _LIST_WORDS]
        if not rest or all(w in ROW_WORDS for w in rest):
            return True
        found = self.matcher.candidates(tuple(rest))
        return bool(found) and found[0].score >= 100 and self.matcher.kinds[found[0].column] == de.CATEGORICAL

    # -- routing --------------------------------------------------------------------

    def route(self, q: str) -> Answer:
        if _MISSING_RE.search(q):
            return self.missing(q)
        if _CORR_RE.search(q):
            return self.correlation(q)
        if _ROWCOUNT_RE.search(q) and not _GROUP_SEP_RE.search(q):
            return self.overview()
        if m := _DESCRIBE_RE.search(q):
            rest = _strip(words(m.group("rest")))
            if not [w for w in rest if w not in _DESCRIBE_FILLER]:
                return self.overview()
            return self.distribution(rest)
        if _TREND_RE.search(q):
            return self.trend(q)
        if m := (_DIST_RE.search(q) or _DIST_SUFFIX_RE.search(q)):
            return self.distribution(words(m.group("col")))
        if m := _TOP_RE.search(q):
            return self.top(m.group("dir"), _parse_top_n(m.group("n")), m.group("rest"))
        if m := _TOP_DEFAULT_RE.search(q):
            return self.top(m.group("dir"), DEFAULT_TOP_N, m.group("rest"), default_n=True)
        if m := _MOST_COMMON_RE.search(q):
            return self.ranking(words(m.group("entity")), m.group("dir"), ())
        if m := _WHICH_RE.search(q):
            return self.ranking(words(m.group("entity")), m.group("dir"), words(m.group("rest")))
        if m := _HOW_MANY_RE.match(q):
            return self.how_many(m.group("x"))
        if m := _COMPARE_RE.match(q):
            return self.group(words(m.group("measure")), m.group("group"))
        if m := _COMPARE_ONLY_RE.match(q):
            return self.group((), m.group("group"))
        if m := _GROUP_RE.match(q):
            return self.group(words(m.group("measure")), m.group("group"))
        if m := _AGG_RE.match(q):
            return self.aggregate(_AGG_WORDS[m.group("agg")], words(m.group("measure")))
        if metrics.match_metric(_strip(words(q))):  # "calculate net revenue"
            return self.aggregate("sum", words(q))
        return self.fallback(q)

    # -- handlers -------------------------------------------------------------------

    def overview(self) -> Answer:
        profile = de.profile_dataset(self.df)
        o = profile.overview
        subject = "The filtered data" if self.filters else "This dataset"
        text = (
            f"{subject} has **{fmt(o.rows)} rows** and **{fmt(o.columns)} columns**, with "
            f"{fmt(o.missing_cells)} missing cells and {fmt(o.duplicate_rows)} duplicate rows."
        )
        return self._answer(
            "overview", text, calculation="data_engine.profile_dataset (row and column counts)",
            table=profile.columns,
        )  # fmt: skip

    def missing(self, q: str) -> Answer:
        if m := _MISSING_COLUMN_RE.search(q):
            phrase = _strip(words(m.group("col")))
            if phrase and phrase not in (("dataset",), ("data",)):
                column = self._resolve(phrase, "column to check")
                count = int(self.df[column].isna().sum())
                share = count / len(self.df) * 100 if len(self.df) else 0
                text = (
                    f"{_col(column)} has **{fmt(count)} missing values** "
                    f"({share:.1f}% of {fmt(len(self.df))} rows)."
                )
                return self._answer(
                    "missing values", text, columns=[_col(column)],
                    calculation="count of empty cells in the column",
                )  # fmt: skip

        table = profiling.missing_values(self.df)
        total = int(table["missing"].sum())
        with_missing = table[table["missing"] > 0]
        if total == 0:
            text = "**No missing values** were found in any column."
            return self._answer(
                "missing values", text, table=table,
                calculation="profiling.missing_values (empty cells per column)",
            )  # fmt: skip
        worst = with_missing.iloc[0]
        text = (
            f"**Yes.** There are **{fmt(total)} missing cells** in {len(with_missing)} of "
            f"{self.df.shape[1]} columns. The most affected is {_col(worst['column'])} with "
            f"{fmt(worst['missing'])} missing ({worst['missing_pct']:.1f}%)."
        )
        figure = charts.bar_chart(
            ChartData(with_missing, x="column", y="missing", title="Missing values per column")
        )
        self.ctx.update(chart_x="column", chart_y="missing", chart_kind="bar")
        return self._answer(
            "missing values", text, table=table, figure=figure,
            calculation="profiling.missing_values (empty cells per column)",
        )  # fmt: skip

    def correlation(self, q: str) -> Answer:
        for pattern in _CORR_PAIR_RES:
            m = pattern.search(q)
            if not m:
                continue
            a_words, b_words = _strip(words(m.group("a"))), _strip(words(m.group("b")))
            generic = {"numeric", "number", "all", "column", "variable", "field"}
            if set(a_words + b_words) & generic or not a_words or not b_words:
                break
            a = self._resolve(a_words, "first column", {de.NUMERIC})
            b = self._resolve(b_words, "second column", {de.NUMERIC})
            if a == b:
                raise _Clarify("Please name two different numeric columns to compare.")
            corr = de.correlation_matrix(self.df, [a, b])
            r = corr.loc[a, b]
            if pd.isna(r):
                text = (
                    f"There are not enough rows with values in both {_col(a)} and {_col(b)} "
                    "(at least 3 are needed), so no correlation could be calculated."
                )
            else:
                text = (
                    f"The correlation between {_col(a)} and {_col(b)} is **{r:.2f}** "
                    f"({de.correlation_strength(r)}). Correlation shows whether two numbers "
                    "move together; it does not prove that one causes the other."
                )
            return self._answer(
                "correlation", text, columns=[_col(a), _col(b)], table=corr.round(3),
                figure=charts.correlation_heatmap(corr),
                calculation="data_engine.correlation_matrix (Pearson; rows missing either value are skipped)",
            )  # fmt: skip

        corr = de.correlation_matrix(self.df)
        pairs = de.strongest_correlations(corr, top=5)
        if pairs.empty:
            text = "There is not enough data in the numeric columns to calculate correlations."
        else:
            top = pairs.iloc[0]
            text = (
                f"I compared {len(corr.columns)} numeric columns. The strongest relationship is "
                f"between {_col(top['column_a'])} and {_col(top['column_b'])}: "
                f"**{top['correlation']:.2f}** ({top['strength']})."
            )
        return self._answer(
            "correlation", text, columns=[_join(corr.columns)], table=pairs.round(3),
            figure=charts.correlation_heatmap(corr),
            calculation="data_engine.correlation_matrix + strongest_correlations (Pearson)",
        )  # fmt: skip

    def _date_column(self, q_words: Sequence[str]) -> Any:
        dates = de.date_columns(self.df)
        if not dates:
            raise _Unsupported(
                "This dataset has no date column, so I can't show a trend over time. "
                "Dates should look like 2026-01-31 to be recognised."
            )
        if len(dates) == 1:
            return dates[0]
        mentioned = [c for c in dates if _contains(q_words, words(c))]
        if len(mentioned) == 1:
            return mentioned[0]
        raise _Clarify(
            f"This dataset has several date columns: {_join(dates)}. Which one should I use? "
            "Please include its name in the question.",
            dates,
        )

    def trend(self, q: str) -> Answer:
        q_words = words(q)
        freq = next((_FREQ_WORDS[w] for w in q_words if w in _FREQ_WORDS), "Month")
        date_column = self._date_column(q_words)

        rest = _remove(words(_TREND_REMOVE_RE.sub(" ", q)), words(date_column))
        agg, measure_words = _split_agg(rest)
        value = self._measure(measure_words, "value to track")
        if value is None:
            if agg not in (None, "count"):
                raise _Clarify("Rows can only be counted. Name a numeric column to use a total or average.")
            agg_used = "count"
            calculation = f"number of rows per {freq.lower()}"
        else:
            agg_used = agg or "sum"
            calculation = f"{agg_used} of {value} per {freq.lower()} (missing values ignored)"
        self.ctx["agg"] = agg_used
        chart = de.line_chart_data(self.df, date_column, value, freq=freq, agg=agg_used)
        return self._trend_answer(chart, date_column, value, freq, calculation)

    def _trend_answer(self, chart: ChartData, date_column: Any, value: Any, freq: str,
                      calculation: str) -> Answer:  # fmt: skip
        data = chart.data
        label = chart.y
        valid = data.dropna(subset=[label])
        period_format = {"Day": "%Y-%m-%d", "Week": "week of %Y-%m-%d", "Month": "%b %Y",
                         "Quarter": "%Y-%m", "Year": "%Y"}[freq]  # fmt: skip

        def when(ts: pd.Timestamp) -> str:
            if freq == "Quarter":
                return f"Q{ts.quarter} {ts.year}"
            return ts.strftime(period_format)

        text = f"{_cap(label)} per {freq.lower()} from {when(data['period'].iloc[0])} to " \
               f"{when(data['period'].iloc[-1])} ({len(data)} {freq.lower()}s)."  # fmt: skip
        if not valid.empty:
            peak = valid.loc[valid[label].idxmax()]
            low = valid.loc[valid[label].idxmin()]
            text += (
                f" Highest: **{when(peak['period'])}** with **{fmt(peak[label])}**. "
                f"Lowest: {when(low['period'])} with {fmt(low[label])}."
            )
        if chart.note:
            text += f" Note: {chart.note}"
        columns = [f"{_col(date_column)} (date)"]
        if value is not None:
            columns.append(f"{_col(value)} (value)")
        self.ctx.setdefault("agg", "count")
        self.ctx.update(
            value=value, date=date_column, freq=freq, chart_x="period", chart_y=label, chart_kind="line"
        )
        return self._answer(
            "trend", text, columns=columns, table=data, figure=charts.line_chart(chart),
            calculation=f"data_engine.line_chart_data - {calculation}",
        )  # fmt: skip

    def distribution(self, phrase: Sequence[str]) -> Answer:
        column = self._resolve(phrase, "column to describe")
        kind = self.matcher.kinds[column]

        if kind == de.DATETIME:
            chart = de.line_chart_data(self.df, column, None, freq="Month")
            return self._trend_answer(chart, column, None, "Month", "number of rows per month")

        if kind == de.NUMERIC:
            chart = de.histogram_data(self.df, column)
            stats = de.describe_numeric(self.df, [column])
            s = stats.loc[column]
            text = (
                f"{_col(column)} ranges from **{fmt(s['min'])}** to **{fmt(s['max'])}**, with an "
                f"average of **{fmt(s['mean'])}** and a median of {fmt(s['median'])} "
                f"({fmt(s['count'])} values"
                + (f", {fmt(s['missing'])} missing ignored" if s["missing"] else "")
                + ")."
            )
            self.ctx.update(value=column, chart_kind="histogram")
            return self._answer(
                "distribution", text, columns=[_col(column)], table=stats.round(2),
                figure=charts.histogram_chart(chart),
                calculation="data_engine.histogram_data + describe_numeric",
            )  # fmt: skip

        chart = de.bar_chart_data(self.df, column, top=CATEGORY_CHART_LIMIT)
        distinct = int(self.df[column].nunique(dropna=False))
        top = chart.data.iloc[0]
        text = (
            f"{_col(column)} has **{fmt(distinct)} different values**. The most common is "
            f"**{fmt(top[column])}** with {fmt(top['count'])} rows ({top['percent']:.1f}%)."
        )
        if distinct > CATEGORY_CHART_LIMIT:
            text += f" The chart shows the {CATEGORY_CHART_LIMIT} most common values."
        self.ctx.update(entity=column, agg="count", chart_x=column, chart_y="count", chart_kind="bar")
        return self._answer(
            "distribution", text, columns=[_col(column)], table=chart.data,
            figure=charts.bar_chart(chart),
            calculation="data_engine.bar_chart_data (rows per value; missing shown as '(missing)')",
        )  # fmt: skip

    def top(self, direction: str, n: int, rest: str, default_n: bool = False) -> Answer:
        largest = direction not in _SMALL_FIRST
        parts = _TOP_SPLIT_RE.split(rest, maxsplit=1)
        entity_words = _strip(words(parts[0]))
        agg, measure_words = _split_agg(words(parts[1])) if len(parts) > 1 else (None, ())

        entity = self._resolve(entity_words, "thing to rank")
        word = "largest" if largest else "smallest"
        if self.matcher.kinds[entity] == de.NUMERIC and not measure_words:
            table = de.top_rows(self.df, entity, n=n, largest=largest)
            text = f"These are the **{len(table)} rows** with the {word} {_col(entity)} values."
            self.ctx.update(value=entity, n=n, largest=largest)
            return self._answer(
                "top N", text, columns=[_col(entity)], table=table,
                calculation=f"data_engine.top_rows (rows sorted by {entity}; missing values skipped)",
            )  # fmt: skip

        value = self._measure(measure_words, "value to rank by")
        if value is None and agg not in (None, "count"):
            raise _Clarify("Rows can only be counted. Name a numeric column to rank by a total or average.")
        agg_used = agg if value is not None and agg else "sum"
        table = de.top_n(self.df, entity, n=n, value_column=value, agg=agg_used, largest=largest)
        y = table.columns[-1] if value is not None else "count"
        listing = ", ".join(f"{fmt(r[entity])} ({fmt(r[y])})" for _, r in table.head(5).iterrows())
        ranked_by = f"{_AGG_LABELS[agg_used]} {_col(value)}" if value is not None else "number of rows"
        side = "Top" if largest else "Bottom"
        text = f"{side} {len(table)} {_col(entity)} by {ranked_by}: {listing}."
        if len(table) < n:
            text += f" (Only {len(table)} different values exist.)"
        if default_n:
            text += f" I showed {n} because no number was given."
        figure = charts.bar_chart(ChartData(table, x=entity, y=y, title=f"{side} {n} {entity} by {y}"))
        columns = [f"{_col(entity)} (ranked)"] + ([f"{_col(value)} (value)"] if value is not None else [])
        calc = f"{agg_used} of {value} per {entity}" if value is not None else f"rows per {entity}"
        self.ctx.update(
            entity=entity, value=value, agg=agg_used if value is not None else "count", n=n,
            largest=largest, chart_x=entity, chart_y=y, chart_kind="bar",
        )  # fmt: skip
        return self._answer(
            "top N", text, columns=columns, table=table, figure=figure,
            calculation=f"data_engine.top_n - {calc}, sorted {'largest' if largest else 'smallest'} first",
        )  # fmt: skip

    def ranking(self, entity_words: Sequence[str], direction: str, rest: Sequence[str]) -> Answer:
        largest = direction not in _SMALL_FIRST
        entity = self._resolve(entity_words, "thing to compare")
        agg, measure_words = _split_agg(rest)
        value = self._measure(measure_words, "value to compare")
        if value is None and agg not in (None, "count"):
            raise _Clarify("Rows can only be counted. Name a numeric column to compare totals or averages.")
        agg_used = (agg or "sum") if value is not None else "sum"

        table = de.top_n(
            self.df, entity, n=RANKING_ROWS, value_column=value, agg=agg_used, largest=largest
        )
        y = table.columns[-1] if value is not None else "count"
        winner = table.iloc[0]
        ties = table[table[y] == winner[y]][entity].tolist()
        what = (f"{_AGG_LABELS[agg_used]} {_col(value)}" if value is not None else "rows")
        word = direction if direction in ("most", "least", "fewest") else ("highest" if largest else "lowest")
        if len(ties) > 1:
            text = f"There is a tie for the {word} {what}: {', '.join(f'**{fmt(t)}**' for t in ties)} " \
                   f"each have **{fmt(winner[y])}**."  # fmt: skip
        else:
            text = f"**{fmt(winner[entity])}** has the {word} {what}: **{fmt(winner[y])}**."
        if value is not None and agg is None:
            text += f" (Ranked by the total. Ask for the 'average {_spoken(value)}' to compare averages.)"
        figure = charts.bar_chart(ChartData(table, x=entity, y=y, title=f"{y} by {entity}"))
        columns = [f"{_col(entity)} (compared)"] + ([f"{_col(value)} (value)"] if value is not None else [])
        calc = f"{agg_used} of {value} per {entity}" if value is not None else f"rows per {entity}"
        self.ctx.update(
            entity=entity, value=value, agg=agg_used if value is not None else "count",
            n=RANKING_ROWS, largest=largest, winner=winner[entity] if len(ties) == 1 else None,
            chart_x=entity, chart_y=y, chart_kind="bar",
        )  # fmt: skip
        return self._answer(
            "ranking", text, columns=columns, table=table, figure=figure,
            calculation=f"data_engine.top_n - {calc}",
        )  # fmt: skip

    def how_many(self, x: str) -> Answer:
        if _GROUP_SEP_RE.search(f" {x} "):
            measure, group = _GROUP_SEP_RE.split(f" {x} ", maxsplit=1)
            return self.group(words(measure), group.strip(), forced_agg="count")
        phrase = _strip(words(x))
        if len(phrase) == 1 and phrase[0] in ROW_WORDS and self._measure(phrase) is None:
            return self.overview()  # "how many orders" -> number of rows
        column = self._resolve(phrase, "column to count")
        result = de.aggregate_column(self.df, column, "nunique")
        text = f"{_col(column)} has **{fmt(result.value)} different values**"
        text += f" ({fmt(result.missing)} missing values not counted)." if result.missing else "."
        chart = de.bar_chart_data(self.df, column, top=CATEGORY_CHART_LIMIT)
        self.ctx.update(entity=column, agg="count", chart_x=column, chart_y="count")
        return self._answer(
            "count", text, columns=[_col(column)], table=chart.data,
            calculation="data_engine.aggregate_column (number of distinct non-missing values)",
        )  # fmt: skip

    def group(self, measure_words: Sequence[str], group_text: str, forced_agg: str | None = None) -> Answer:
        agg, measure = _split_agg(measure_words)
        agg = forced_agg or agg
        value = self._measure(measure)
        parts = [p for p in re.split(r"\s+and\s+|\s*,\s*", group_text) if p.strip()]
        if not parts:
            raise _Clarify("Which column should I group by?", self.matcher.names())
        if len(parts) > 2:
            raise _Unsupported("I can group by at most two columns at a time.")
        groups = [self._resolve(words(p), "grouping column") for p in parts]

        if value is None:
            if agg not in (None, "count"):
                raise _Clarify("Rows can only be counted. Name a numeric column to use a total or average.")
            result = de.group_summary(self.df, groups)
            y, calc = "rows", "number of rows in each group"
        else:
            agg = agg or "sum"
            result = de.group_summary(self.df, groups, value, [agg])
            y = result.columns[-1]
            result = result.sort_values(y, ascending=False, na_position="last", kind="stable")
            result = result.reset_index(drop=True)
            calc = f"{agg} of {value} for each group (missing values ignored)"

        top = result.iloc[0]
        group_label = " / ".join(fmt(top[g]) for g in groups)
        measure_label = f"{_AGG_LABELS[agg]} {_col(value)}" if value is not None else "number of rows"
        text = (
            f"{_cap(measure_label)} by {_join(groups)} ({len(result)} groups). "
            f"Highest: **{group_label}** with **{fmt(top[y])}**."
        )
        figure = None
        if len(groups) == 1:
            figure = charts.bar_chart(ChartData(result, x=groups[0], y=y, title=f"{y} by {groups[0]}"))
        self.ctx.update(
            groups=tuple(groups), entity=groups[0] if len(groups) == 1 else None, value=value,
            agg=agg if value is not None else "count",
            chart_x=groups[0] if len(groups) == 1 else None, chart_y=y,
            chart_kind="bar" if figure is not None else None,
        )  # fmt: skip
        columns = [f"{_join(groups)} (groups)"] + ([f"{_col(value)} (value)"] if value is not None else [])
        return self._answer(
            "group comparison", text, columns=columns, table=result, figure=figure,
            calculation=f"data_engine.group_summary - {calc}; missing groups shown as '(missing)'",
        )  # fmt: skip

    def aggregate(self, agg: str, measure_words: Sequence[str]) -> Answer:
        _, measure = _split_agg(measure_words)  # "total number of ..." etc.
        value = self._measure(measure)
        if value is None:
            if agg != "sum":
                raise _Clarify("Rows can only be counted. Name a numeric column to calculate that.")
            return self.overview()
        result = de.aggregate_column(self.df, value, agg)
        self.ctx.update(value=value, agg=agg)
        text = f"The {_AGG_LABELS[agg]} of {_col(value)} is **{fmt(result.value)}**"
        text += f" (from {fmt(result.used_values)} values"
        text += f"; {fmt(result.missing)} missing values ignored)." if result.missing else ")."
        return self._answer(
            "aggregate", text, columns=[_col(value)], table=self.metric_tables.get(value),
            calculation=f"data_engine.aggregate_column - {agg} of {value}",
        )  # fmt: skip

    def fallback(self, q: str) -> Answer:
        q_words = words(q)
        mentioned = [c for c in self.matcher.columns if _contains(q_words, words(c))]
        message = "I'm not sure how to answer that."
        if mentioned:
            message += f" I recognised {_join(mentioned)}, but not what you want to know about it."
        raise _Unsupported(f"{message} {CAPABILITIES}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def answer_question(df: pd.DataFrame | None, question: Any, base_filters: Sequence = ()) -> Answer:
    """Answer one plain-English question about ``df``. Never raises.

    ``base_filters`` are ``nl_filters.FilterSpec`` objects from an earlier answer
    (used by follow-up questions); they are applied before the question's own filters.
    """
    if not isinstance(df, pd.DataFrame):
        return Answer(NO_DATA, "No dataset is selected. Upload a file, choose the sample "
                      "dataset or send a database result first.")  # fmt: skip
    if df.shape[1] == 0:
        return Answer(NO_DATA, "The selected dataset has no columns.")

    try:
        prepared = _prepare(question)
        router = _Router(df, question=str(question).strip(), discount_scale=prepared.discount_scale)
        return router.answer(prepared, base_filters)
    except _Clarify as exc:
        return Answer(CLARIFY, exc.message, options=exc.options, rephrasings=exc.rephrasings)
    except _Unsupported as exc:
        return Answer(UNSUPPORTED, exc.message, suggestions=tuple(suggest_questions(df)))
    except AnalysisError as exc:
        return Answer(ERROR, f"I couldn't calculate that: {exc}")
    except Exception:  # never show internal details to the user
        logger.exception("Unexpected error while answering a question")
        return Answer(ERROR, "Something went wrong while answering. Please try rephrasing the question.")


def screen_question(df: pd.DataFrame, question: Any) -> Answer | None:
    """The router's safety screen on its own: an Answer when the question must be
    refused or clarified before anything else (code, SQL, writes, unclear dates), else None."""
    try:
        _prepare(question)
    except _Clarify as exc:
        return Answer(CLARIFY, exc.message, options=exc.options, rephrasings=exc.rephrasings)
    except _Unsupported as exc:
        return Answer(UNSUPPORTED, exc.message, suggestions=tuple(suggest_questions(df)))
    return None


def parse_filters(df: pd.DataFrame, text: str) -> tuple[tuple, str]:
    """Filters in ``text`` as (FilterSpec objects, leftover words). May raise
    ClarificationNeeded / UnsupportedRequest exactly like a full question."""
    prepared = _prepare(text)
    parsed = nl_filters.FilterParser(df, ColumnMatcher(df), prepared.dates).parse(prepared.q)
    return parsed.specs, parsed.remainder
