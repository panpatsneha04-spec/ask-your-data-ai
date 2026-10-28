"""Follow-up questions in Ask Your Data: deterministic, local, no LLM.

A follow-up builds on the previous successful answer's ``AnswerContext`` (the
validated columns, filters and result table recorded by the question router).
It is turned into one of two safe operations:

1. A new question for the existing router, with the previous filters carried
   over as validated ``FilterSpec`` objects:
   * reference   "Show the top 5 products in that city" -> filter City = <winner>
                 "Show the trend for those products"    -> Product is one of <listed>
   * regroup     "Now group them by category"           -> "count by category" + filters
   * swap value  "What about Mumbai?"                    -> same question, City = Mumbai
   * top N       "Show the top 10 instead"               -> same top-N question, new N
2. A validated operation on the previous result table:
   * sort        "Sort it from highest to lowest", "lowest first", "sort by <column>"
   * top N       "Show only the top 3" (slice of the rows already shown)
   * chart       "Plot it as a bar / line / pie chart", "show it as a table"

References are resolved only from values in the previous result or its filters.
If that is not possible with certainty, the user is asked (with ready-made
rephrasings) instead of guessing. Nothing is executed as code or SQL: every
calculation goes through ``question_router`` / ``data_engine`` validation.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from askyourdata import charts, planner
from askyourdata import data_engine as de
from askyourdata import question_router as qr
from askyourdata.data_engine import AnalysisError, ChartData, Condition
from askyourdata.nl_filters import FilterSpec
from askyourdata.question_router import (
    ANSWER,
    CLARIFY,
    ERROR,
    UNSUPPORTED,
    Answer,
    AnswerContext,
    answer_question,
)
from askyourdata.text_matching import (
    ClarificationNeeded,
    ColumnMatcher,
    UnsupportedRequest,
    _singular,
    fmt,
    looks_like_id,
    words,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 20  # conversation memory is bounded
MAX_REFERENCE_VALUES = 100  # "those products" may stand for at most this many values
MAX_PIE_SLICES = 20
LISTED_VALUES = 5

# Follow-up kinds (shown to the user as "follow-up: ...")
REFERENCE = "reference"
SORT = "sort"
TOP_N = "top N"
CHART = "chart"
REGROUP = "regroup"
SWAP_VALUE = "what about"
PLAN = "plan"  # a multi-step request with its own calculation
PLAN_FOLLOW_UP = "multi-step follow-up"  # several presentation steps on the previous answer

NO_CONTEXT_MESSAGE = (
    "That sounds like a follow-up, but there is no earlier answer to build on in this "
    "conversation (it starts fresh when the dataset changes or is cleared). Please ask a "
    "complete question first."
)


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Turn:
    question: str
    answer: Answer
    follow_up: str | None = None  # the follow-up kind, or None for a fresh question


class Conversation:
    """The current Ask Your Data conversation for ONE dataset, kept in memory only.

    Holds at most ``max_turns`` turns. Only the latest successful answer's
    context is used for follow-ups. Nothing is written to disk.
    """

    def __init__(self, fingerprint: str | None = None, max_turns: int = MAX_TURNS) -> None:
        self.fingerprint = fingerprint
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.last: AnswerContext | None = None

    def clear(self) -> None:
        self.turns.clear()
        self.last = None

    def reset_if_dataset_changed(self, fingerprint: str | None) -> bool:
        """Forget everything that belongs to another dataset."""
        if fingerprint == self.fingerprint:
            return False
        self.clear()
        self.fingerprint = fingerprint
        return True

    def ask(self, df: pd.DataFrame, question: Any) -> Turn:
        answer, kind = respond(df, question, self.last)
        if answer.ok and answer.context is not None:
            self.last = answer.context
        turn = Turn(str(question), answer, kind)
        self.turns.append(turn)
        return turn


# ---------------------------------------------------------------------------
# Detecting follow-ups
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"^(?:ok|okay|now|and|then|also|so|please|can you|could you|would you|great|thanks)\b\s*"
)
_N = r"-?\d+(?:\.\d+)?|" + "|".join(qr._NUMBER_WORDS)
_TOP_N_RE = re.compile(
    r"^(?:show|give|list|keep|display|return)?\s*(?:me\s+)?(?:only\s+|just\s+)?(?:the\s+)?"
    rf"(?P<dir>top|bottom|first|last)\s+(?P<n>{_N})"
    r"(?:\s+(?:only|instead|rows|results|ones|items|of them|of those|of these|please))*$"
)
# "sort it ...", "order by ...": the verb must be followed by a pronoun or a sort
# word, so questions such as "order trend by month" are not mistaken for sorting.
_SORT_RE = re.compile(
    r"^(?:sort|order|rank|re ?sort|re ?order|arrange)"
    r"(?:\s+(?:it|them|this|that|these|those|the results?|the table|the rows|the list)\b)?"
    r"(?P<rest>(?:\s+(?:by|from|in|ascending|descending|alphabetical(?:ly)?|highest|lowest|largest"
    r"|smallest|biggest|desc|asc|decreasing|increasing)\b.*)?)$"
)
_REVERSE_RE = re.compile(r"^(?:reverse|flip)(?:\s+(?:the|it|this|that))?(?:\s+(?:order|sort|sorting))?$")
_DIRECTION_ONLY_RE = re.compile(
    r"^(?:show\s+)?(?:the\s+)?(?:(?:highest|largest|biggest|most|lowest|smallest|least)\s+first"
    r"|ascending|descending|from (?:highest|lowest|largest|smallest)(?: to (?:lowest|highest|smallest|largest))?)$"
)
_DESC_WORDS = ("highest", "largest", "biggest", "most", "descending", "desc", "decreasing", "high to low", "big to small")
_ASC_WORDS = ("lowest", "smallest", "least", "ascending", "asc", "increasing", "low to high", "small to big",
              "alphabetical", "alphabetically", "a to z")  # fmt: skip
_CHART_KIND_RE = re.compile(
    r"\b(?:as|into|to|in|use|using)\s+(?:a\s+|an\s+)?(?P<kind>bar|column|line|pie|donut|doughnut)\b"
    r"|\b(?:make|turn|change|switch|convert|redraw|draw|plot|show)\s+(?:it|this|that|them)\s+"
    r"(?:into\s+|to\s+)?(?:a\s+|an\s+)?(?P<kind2>bar|column|line|pie|donut|doughnut)\b"
    r"|^(?:a\s+)?(?P<alone>bar|column|line|pie|donut|doughnut)\s*(?:chart|graph|plot)?(?:\s+(?:instead|please))*$"
)
_TABLE_RE = re.compile(
    r"\b(?:as\s+a\s+table|table\s+only|just\s+the\s+table|only\s+the\s+table|without\s+(?:a\s+|the\s+)?chart)\b"
)
_PRONOUN_RE = re.compile(r"\b(?:it|this|that|them|these|those|results?|same|answer|instead)\b")
# "group them by X", "break it down by X", "split by X"; "show/summarise/compare" only
# with a pronoun ("show them by X"), so "count by category" stays a fresh question.
_REGROUP_RE = re.compile(
    r"^(?:(?:group|break|split)\s+"
    r"(?P<pronoun>(?:them|it|these|those|this|that|the results?|the rows|the orders|the data)\s+)?"
    r"|(?:show|summari[sz]e|compare)\s+"
    r"(?P<pronoun2>(?:them|it|these|those|this|that|the results?|the rows|the orders|the data)\s+))"
    r"(?:down\s+|up\s+)?(?:by|per|across)\s+(?P<group>.+)$"
)
_SWAP_RE = re.compile(
    r"^(?:what|how)\s+about\s+(?P<a>.+?)(?:\s+instead)?$"
    r"|^(?:the\s+)?same\s+(?:for|with|in)\s+(?P<b>.+?)(?:\s+instead)?$"
    r"|^(?:for|in)\s+(?P<c>.+?)\s+instead$"
)
_REFERENCE_RE = re.compile(
    r"(?:\b(?P<prep>in|for|from|of|on|at|with|among|within|across)\s+)?"
    r"\b(?P<det>that|this|those|these|the same)\s+(?P<noun>[a-z0-9]+(?:\s+[a-z0-9]+){0,2})"
)
_AGG_WORDS = {"sum": "total", "mean": "average", "median": "median", "min": "minimum", "max": "maximum"}
_METRIC_PHRASES = {"net_revenue": "net revenue", "gross_amount": "gross amount"}


def _normalise(question: str) -> str:
    q = re.sub(r"[^a-z0-9.\-\s]", " ", question.lower().replace("'", " "))
    q = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    while (stripped := _FILLER_RE.sub("", q, count=1)) != q:
        q = stripped
    return q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(column: Any) -> str:
    return f"`{column}`"


def _spoken(column: Any) -> str:
    return qr._spoken(column)


def _measure_column(last: AnswerContext) -> Any | None:
    """The numeric column of the previous result table (what "it" is sorted by)."""
    table = last.table
    if table is None:
        return None
    if last.chart_y is not None and last.chart_y in table.columns and de.is_numeric(table[last.chart_y]):
        return last.chart_y
    numeric = [c for c in table.columns if de.is_numeric(table[c]) and c not in ("row", "percent")]
    return numeric[-1] if numeric else None


def _filter_lines(last: AnswerContext) -> tuple[str, str]:
    if not last.filters:
        return "", ""
    text = "; ".join(spec.description for spec in last.filters)
    return f"**Filters (still applied):** {text}.\n\n", f"Filters still applied: {text}."


def _figure_for(context: AnswerContext, table: pd.DataFrame, kind: str | None):
    x, y = context.chart_x, context.chart_y
    if kind is None or x is None or y is None or x not in table.columns or y not in table.columns:
        return None
    chart = ChartData(table, x=x, y=y, title=f"{y} by {x}")
    if kind == "line":  # always drawn in date order, even after sorting by value
        return charts.line_chart(replace(chart, data=table.sort_values(x)))
    if kind == "pie":
        return charts.pie_chart(chart)
    return charts.bar_chart(chart)


def _table_answer(
    last: AnswerContext, question: str, table: pd.DataFrame, text: str, calculation: str,
    chart_kind: str | None, **changes: Any,
) -> Answer:  # fmt: skip
    """An answer built from a transformed copy of the previous result table."""
    prefix, filter_detail = _filter_lines(last)
    context = replace(last, question=question, table=table, chart_kind=chart_kind, **changes)
    details = [f"Follow-up to: “{last.question}”", f"Calculation: {calculation}"]
    if filter_detail:
        details.append(filter_detail)
    return Answer(
        ANSWER, prefix + text, intent=f"follow-up: {last.intent}", table=table,
        figure=_figure_for(context, table, chart_kind), details=tuple(details), context=context,
    )  # fmt: skip


def _with_note(answer: Answer, last: AnswerContext | None, note: str, question: str) -> Answer:
    """Mark a re-run answer as a follow-up and explain what was carried over.

    The context keeps the user's own wording, so later follow-ups quote it."""
    if last is not None:
        note = f"Follow-up to: “{last.question}”. {note}"
    context = replace(answer.context, question=question) if answer.context is not None else None
    return replace(answer, details=(note, *answer.details), context=context)


def _values_text(values: list) -> str:
    shown = ", ".join(map(str, values[:LISTED_VALUES]))
    return shown + (f" and {len(values) - LISTED_VALUES} more" if len(values) > LISTED_VALUES else "")


# ---------------------------------------------------------------------------
# Follow-up handlers
# ---------------------------------------------------------------------------


def _top_n(df: pd.DataFrame, question: str, last: AnswerContext, direction: str, raw_n: str) -> Answer:
    n = qr._parse_top_n(raw_n)  # validated: whole number 1-100
    largest = direction in ("top", "first")
    table = last.table
    if table is None or table.empty:
        raise ClarificationNeeded("The previous answer has no table to take the top rows from.")

    # A new top-N for the same question: answer it again when the rows shown are not enough.
    if last.intent == "top N" and last.core and (n > len(table) or largest != last.largest):
        new_core, count = re.subn(
            rf"\b(?:top|bottom|best|worst|highest|lowest|largest|smallest|biggest|first|last)\s+(?:{_N})\b",
            f"{'top' if largest else 'bottom'} {n}", last.core, count=1,
        )  # fmt: skip
        if not count:
            new_core, count = re.subn(r"\b(?:top|bottom)\b", f"{'top' if largest else 'bottom'} {n}", last.core, count=1)
        if count:
            answer = answer_question(df, new_core, base_filters=last.filters)
            return _with_note(answer, last, f"Answered again as “{new_core}” with the same filters.", question)

    measure = _measure_column(last)
    if measure is None or last.intent in ("filter", "follow-up: filter"):
        rows = table.head(n) if largest else table.tail(n)
        word = "first" if largest else "last"
        text = f"The {word} {len(rows)} of the {len(table)} rows in the previous result."
        return _table_answer(last, question, rows, text, f"{word} {n} rows of the previous table", last.chart_kind)

    ordered = de.sort_rows(table, measure, ascending=not largest)
    rows = ordered.head(n).reset_index(drop=True)
    side = "Top" if largest else "Bottom"
    label = last.chart_x if last.chart_x in rows.columns else rows.columns[0]
    listing = ", ".join(f"{fmt(r[label])} ({fmt(r[measure])})" for _, r in rows.head(LISTED_VALUES).iterrows())
    text = f"{side} {len(rows)} by {_col(measure)}: {listing}."
    if n > len(table):
        text += (
            f" The previous result only had {len(table)} rows; ask a new question such as "
            f"“top {n} … by …” to look further."
        )
    return _table_answer(
        last, question, rows, text,
        f"previous result sorted by {measure} ({'largest' if largest else 'smallest'} first, "
        f"data_engine.sort_rows), first {n} rows kept",
        last.chart_kind, n=len(rows), largest=largest,
    )  # fmt: skip


def _direction(text: str) -> bool | None:
    """True for descending, False for ascending, None if not stated (first word wins)."""
    hits = [(text.find(w), True) for w in _DESC_WORDS if re.search(rf"\b{w}\b", text)]
    hits += [(text.find(w), False) for w in _ASC_WORDS if re.search(rf"\b{w}\b", text)]
    return min(hits)[1] if hits else None


def _sort(question: str, last: AnswerContext, text: str, reverse: bool = False) -> Answer:
    table = last.table
    if table is None or len(table) < 2:
        raise ClarificationNeeded("The previous answer has no table with several rows to sort.")

    column = None
    if m := re.search(r"\bby\s+(?P<col>.+?)(?:\s+(?:from|in|descending|ascending|highest|lowest|largest|smallest)\b.*)?$", text):
        found = ColumnMatcher(table).resolve(words(m.group("col")), "column to sort by")
        column = found.column
    alphabetical = bool(re.search(r"\balphabetical(?:ly)?\b|\ba to z\b", text))
    if column is None and alphabetical:
        column = last.chart_x if last.chart_x in table.columns else table.columns[0]
    if column is None:
        column = _measure_column(last)
    if column is None:
        raise ClarificationNeeded(
            "Sort by which column?", [str(c) for c in table.columns]
        )

    descending = _direction(text)
    if reverse:
        values = table[column].dropna()
        descending = not values.is_monotonic_decreasing if len(values) > 1 else True
    is_date = pd.api.types.is_datetime64_any_dtype(table[column])
    if descending is None:
        descending = not (alphabetical or is_date)  # numbers: largest first; dates: oldest first

    ordered = de.sort_rows(table, column, ascending=not descending).reset_index(drop=True)
    label = last.chart_x if last.chart_x in ordered.columns and last.chart_x != column else None
    words_dir = "highest to lowest" if descending else "lowest to highest"
    if is_date:
        words_dir = "newest to oldest" if descending else "oldest to newest"
    elif not de.is_numeric(table[column]):
        words_dir = "Z to A" if descending else "A to Z"
    first = ordered.iloc[0]
    lead = f" First: **{fmt(first[label])}** ({fmt(first[column])})." if label else ""
    text_out = f"The previous result sorted by {_col(column)} from {words_dir}.{lead}"
    return _table_answer(
        last, question, ordered, text_out,
        f"previous result sorted by {column}, {words_dir} (data_engine.sort_rows; missing values last)",
        last.chart_kind,
    )  # fmt: skip


def _chart(question: str, last: AnswerContext, kind: str) -> Answer:
    kind = {"column": "bar", "donut": "pie", "doughnut": "pie"}.get(kind, kind)
    table = last.table
    if table is None or table.empty:
        raise ClarificationNeeded("The previous answer has no table to chart.")
    if kind == "table":
        return _table_answer(last, question, table, "The previous result as a table (no chart).",
                             "previous result shown as a table", None)  # fmt: skip

    x, y = last.chart_x, last.chart_y
    if x is None or y is None or x not in table.columns or y not in table.columns:
        raise UnsupportedRequest(
            f"The previous answer can't be drawn as a {kind} chart: it has no category column with "
            "a value per row (for example a single number, statistics or a correlation matrix). "
            "Ask a comparison such as “total quantity by category” first."
        )
    if kind == "line" and not pd.api.types.is_datetime64_any_dtype(table[x]):
        raise UnsupportedRequest(
            f"A line chart needs dates (or another ordered axis), but {_col(x)} is a list of "
            "categories. Try “plot it as a bar chart” instead."
        )
    data = table
    note = ""
    if kind == "pie":
        additive = y in ("count", "rows") or last.agg in ("sum", "count")
        if not additive:
            raise UnsupportedRequest(
                f"A pie chart shows parts of a whole, which doesn't work for {last.agg or 'these'} "
                f"values like {_col(y)}. Try “plot it as a bar chart” instead."
            )
        values = table[y].dropna()
        if (values < 0).any():
            raise UnsupportedRequest(f"{_col(y)} has negative values, so it can't be shown as a pie chart.")
        if len(values) > MAX_PIE_SLICES:
            raise UnsupportedRequest(
                f"The previous result has {len(values)} rows; a pie chart is only drawn for up to "
                f"{MAX_PIE_SLICES}. Try “show only the top 10” first."
            )
        if len(values) < len(table):
            data = table.dropna(subset=[y])
            note = f" {len(table) - len(data)} row(s) without a value are not shown."
    text = f"The previous result as a {kind} chart of {_col(y)} by {_col(x)}.{note}"
    return _table_answer(last, question, data, text, f"previous result redrawn as a {kind} chart", kind)


def _regroup(df: pd.DataFrame, question: str, last: AnswerContext | None, group_text: str) -> Answer:
    if last is not None and last.value is not None and last.agg not in (None, "count"):
        phrase = _METRIC_PHRASES.get(last.metric) or _spoken(last.value)
        core = f"{_AGG_WORDS.get(last.agg, 'total')} {phrase} by {group_text}"
    else:
        core = f"count by {group_text}"
    filters = last.filters if last is not None else ()
    answer = answer_question(df, core, base_filters=filters)
    carried = "the same filters" if filters else "no filters"
    return _with_note(answer, last, f"Answered as “{core}” with {carried}.", question)


def _swap(df: pd.DataFrame, question: str, last: AnswerContext, value_text: str) -> Answer | None:
    """"What about Mumbai?": the same question with that filter instead.
    None when the text holds no filter at all ("what about missing values?")."""
    specs, rest = qr.parse_filters(df, value_text)
    if not specs:
        # A bare value ("Kolkata") is checked like "rows from Kolkata", so unknown
        # values get a clear message (with suggestions) instead of a vague answer.
        specs, rest = qr.parse_filters(df, f"rows from {value_text}")
    leftover = [w for w in words(rest) if w not in qr._STOPWORDS and w not in qr.ROW_WORDS]
    if not specs:
        return None
    if leftover:
        raise ClarificationNeeded(
            f"I couldn't use “{value_text}” as a filter: it should be a value that appears in a text "
            "column (e.g. a city), a number condition or a date."
        )
    replaced = set().union(*(spec.columns for spec in specs))
    kept = [spec for spec in last.filters if not (spec.columns & replaced)]
    core = last.core or ""
    answer = answer_question(df, core, base_filters=(*kept, *specs))
    return _with_note(answer, last, f"Same question with the filter changed to: "
                      f"{'; '.join(s.description for s in specs)}.", question)  # fmt: skip


def _reference_values(last: AnswerContext, column: Any, plural: bool) -> list:
    """Values that "that <column>" / "those <column>s" stand for, from real results only."""
    table = last.table
    in_table = table is not None and column in table.columns
    present = []
    if in_table:
        present = [v for v in pd.unique(table[column].dropna()) if v != de.MISSING_LABEL]
    from_filters = [
        c.value if isinstance(c.value, list) else [c.value]
        for spec in last.filters for c in spec.conditions
        if c.column == column and c.operator in (de.EQUALS, de.IS_ONE_OF)
    ]  # fmt: skip
    filter_values = [v for values in from_filters for v in values]

    if plural:
        return present if in_table and last.intent != "filter" else (filter_values or present)
    if column == last.entity and last.intent == "ranking":
        # "that city" after "which city has the most ...": only the winner itself counts.
        if last.winner == de.MISSING_LABEL:
            return []  # the "winner" is the group of missing values: nothing to filter on
        if last.winner is not None:
            return [last.winner]
        y = last.chart_y
        if in_table and y in table.columns:  # a tie: the tied values, so the user can pick
            top = table[y].max()
            return [v for v in table.loc[table[y] == top, column] if v != de.MISSING_LABEL]
        return present
    if len(filter_values) == 1:
        return filter_values
    return present


# Words after "this/that" that are not references to a value ("this dataset", "this month").
_NOT_REFERENCES = frozenset(
    words(
        "dataset data table file sheet question answer result results chart graph plot list "
        "column columns row rows time period day week month quarter year one way is was"
    )
)


def _reference(df: pd.DataFrame, question: str, q: str, last: AnswerContext | None) -> Answer | None:
    """Resolve "that city" / "those products"; None if the text has no such reference.

    A reference that can't be matched to a column is never silently dropped:
    the user is asked instead."""
    matcher = ColumnMatcher(df)
    for m in _REFERENCE_RE.finditer(q):
        noun_words = m.group("noun").split()
        for length in range(len(noun_words), 0, -1):
            phrase = noun_words[:length]
            found = matcher.candidates(words(" ".join(phrase)))
            if not found or found[0].score < 80:
                continue
            best = [c for c in found if c.score == found[0].score]
            if len(best) > 1:
                raise ClarificationNeeded(
                    f"“{m.group('det')} {' '.join(phrase)}” could mean {', '.join(_col(c.column) for c in best)}.",
                    [c.column for c in best],
                )
            column = best[0].column
            break
        else:
            first = words(noun_words[0])
            if m.group("det") != "this" and first and first[0] not in _NOT_REFERENCES:
                raise ClarificationNeeded(
                    f"I can't tell what “{m.group('det')} {noun_words[0]}” refers to: this dataset has no "
                    f"column like '{noun_words[0]}'"
                    + (" and there is no earlier answer to take it from." if last is None else
                       ", so it can't be taken from the previous answer.")
                    + " Please name the value instead, e.g. “… in Pune”."
                )
            continue
        if last is None:
            raise ClarificationNeeded(NO_CONTEXT_MESSAGE)

        last_noun = phrase[-1]
        plural = m.group("det") in ("those", "these") or _singular(last_noun) != last_noun
        span = q[m.start() : m.start("noun") + len(" ".join(phrase))]
        values = _reference_values(last, column, plural)
        noun = " ".join(phrase)

        if not values:
            raise ClarificationNeeded(
                f"The previous answer doesn't show which {noun} you mean, so I can't resolve "
                f"“{span.strip()}”. Please name it, e.g. “… in <{_spoken(column)}>”."
            )
        if not plural and len(values) > 1:
            prep = f"{m.group('prep')} " if m.group("prep") else ""
            rephrasings = [
                re.sub(re.escape(span.strip()), f"{prep}{v}", q, count=1) for v in values[:LISTED_VALUES]
            ]
            raise ClarificationNeeded(
                f"The previous answer lists several values of {_col(column)} ({_values_text(values)}). "
                f"Which {noun} do you mean?",
                [str(v) for v in values[:LISTED_VALUES]],
                rephrasings=rephrasings,
            )
        if len(values) > MAX_REFERENCE_VALUES:
            raise ClarificationNeeded(
                f"“{span.strip()}” would mean {len(values)} values; please narrow the previous result first."
            )

        values = [str(v) for v in values]
        if len(values) == 1:
            spec = FilterSpec(f"{_col(column)} is {values[0]}", (Condition(column, de.EQUALS, values[0]),))
        else:
            spec = FilterSpec(
                f"{_col(column)} is one of {_values_text(values)}", (Condition(column, de.IS_ONE_OF, values),)
            )
        kept = [s for s in last.filters if column not in s.columns]
        remaining = (q[: m.start()] + " " + q[m.start("noun") + len(noun) :]).strip()
        answer = answer_question(df, remaining or "show rows", base_filters=(*kept, spec))
        return _with_note(
            answer, last, f"“{span.strip()}” was read as {spec.description} (from the previous answer).",
            question,
        )
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def respond(df: pd.DataFrame, question: Any, last: AnswerContext | None) -> tuple[Answer, str | None]:
    """Answer ``question``, as a follow-up to ``last`` when it is one. Never raises."""
    if not isinstance(question, str) or not question.strip():
        return answer_question(df, question), None
    screened = qr.screen_question(df, question)  # code, SQL, writes, ... are refused first
    if screened is not None:
        return screened, None

    if planner.is_multi_step(question):
        chain = planner.follow_up_chain(question)
        if chain is None:
            return planner.answer_plan(df, question), PLAN  # a new multi-step calculation
        if last is None:
            return Answer(CLARIFY, NO_CONTEXT_MESSAGE), PLAN_FOLLOW_UP
        return _run_chain(df, question, chain, last), PLAN_FOLLOW_UP

    q = _normalise(question)
    kind: str | None = None
    try:
        if m := _TOP_N_RE.match(q):
            kind = TOP_N
            if last is None:
                raise ClarificationNeeded(NO_CONTEXT_MESSAGE)
            return _top_n(df, question, last, m.group("dir"), m.group("n")), kind
        if (m := _SORT_RE.match(q)) or _REVERSE_RE.match(q) or _DIRECTION_ONLY_RE.match(q):
            kind = SORT
            if last is None:
                raise ClarificationNeeded(NO_CONTEXT_MESSAGE)
            return _sort(question, last, q, reverse=bool(_REVERSE_RE.match(q))), kind
        chart_kind = "table" if _TABLE_RE.search(q) else None
        if chart_kind is None and (m := _CHART_KIND_RE.search(q)) and (
            m.group("alone") or (_PRONOUN_RE.search(q) and " by " not in f" {q} ")
        ):
            chart_kind = m.group("kind") or m.group("kind2") or m.group("alone")
        if chart_kind:
            kind = CHART
            if last is None:
                raise ClarificationNeeded(NO_CONTEXT_MESSAGE)
            return _chart(question, last, chart_kind), kind
        if m := _REGROUP_RE.match(q):
            kind = REGROUP
            if last is None and (m.group("pronoun") or m.group("pronoun2")):
                raise ClarificationNeeded(NO_CONTEXT_MESSAGE)
            return _regroup(df, question, last, m.group("group")), kind
        if m := _SWAP_RE.match(q):
            value_text = m.group("a") or m.group("b") or m.group("c")
            if last is not None and (answer := _swap(df, question, last, value_text)) is not None:
                return answer, SWAP_VALUE
            if last is None and qr.parse_filters(df, value_text)[0]:
                kind = SWAP_VALUE
                raise ClarificationNeeded(NO_CONTEXT_MESSAGE)
            return answer_question(df, value_text), None  # a fresh question after all
        if (answer := _reference(df, question, q, last)) is not None:
            return answer, REFERENCE
    except ClarificationNeeded as exc:
        return Answer(CLARIFY, exc.message, options=exc.options, rephrasings=exc.rephrasings), kind
    except UnsupportedRequest as exc:
        return Answer(UNSUPPORTED, exc.message, suggestions=tuple(suggest_followups(last, df))), kind
    except AnalysisError as exc:
        return Answer(ERROR, f"I couldn't calculate that: {exc}"), kind
    except Exception:  # never show internal details
        logger.exception("Unexpected error in a follow-up question")
        return Answer(ERROR, "Something went wrong with that follow-up. Please ask a complete question."), kind

    answer = answer_question(df, question)
    if answer.ok:  # show the steps when one question combined several operations
        answer = replace(answer, plan=planner.plan_from_context(answer.context))
    return answer, None


_CHAIN_OPS = {"sort": "sort", "limit": "limit", "chart": "chart", "explain": "summarize"}


def _run_chain(df: pd.DataFrame, question: str, chain: list, last: AnswerContext) -> Answer:
    """Several presentation follow-ups in one request, applied one after another.

    Each step is an ordinary, validated follow-up on the previous step's result.
    The chain stops at the first step that can't run, and then shows no result."""
    steps = [planner.PlanStep(_CHAIN_OPS[kind], {}, f"Follow-up: {text}") for kind, text in chain]
    plan = planner.Plan(question, "Follow-up steps on the previous answer: " + "; ".join(t for _, t in chain) + ".",
                        tuple(steps), last)  # fmt: skip
    outcomes: list = []
    current, answer = last, None
    for number, ((kind, text), step) in enumerate(zip(chain, steps), 1):
        try:
            if kind == "explain":
                explanation = planner.explain_context(current)
                answer = Answer(
                    ANSWER, (answer.text + "\n\n" if answer else "") + explanation, intent="follow-up: explain",
                    table=current.table, figure=answer.figure if answer else None,
                    details=answer.details if answer else (), context=current,
                )  # fmt: skip
            else:
                step_answer, _ = respond(df, text, current)
                if not step_answer.ok:
                    raise _ChainStop(step_answer)
                answer, current = step_answer, step_answer.context or current
            outcomes.append(planner.StepOutcome(step, planner.DONE, "Done."))
        except (_ChainStop, UnsupportedRequest) as exc:
            failed = exc.answer if isinstance(exc, _ChainStop) else Answer(UNSUPPORTED, exc.message)
            outcomes.append(planner.StepOutcome(step, planner.FAILED, failed.text))
            outcomes += [planner.StepOutcome(s, planner.SKIPPED, "Not run because an earlier step stopped.")
                         for s in steps[number:]]  # fmt: skip
            report = planner.PlanReport(plan, tuple(outcomes), planner.FAILED, source="follow-up")
            return Answer(
                failed.status if failed.status != ANSWER else ERROR,
                f"Step {number} (“{text}”) could not run: {failed.text} The remaining steps were skipped "
                "and no result is shown.",
                options=failed.options, plan=report,
            )  # fmt: skip
    report = planner.PlanReport(plan, tuple(outcomes), planner.COMPLETED, source="follow-up")
    return replace(answer, plan=report)


class _ChainStop(Exception):
    def __init__(self, answer: Answer) -> None:
        super().__init__(answer.text)
        self.answer = answer


# ---------------------------------------------------------------------------
# Suggested follow-ups
# ---------------------------------------------------------------------------


def _other_category(df: pd.DataFrame, exclude: set) -> Any | None:
    for column in de.categorical_columns(df):
        if column in exclude or de.column_kind(df[column]) != de.CATEGORICAL or looks_like_id(df, column):
            continue
        if 2 <= df[column].nunique() <= 50:
            return column
    return None


def suggest_followups(last: AnswerContext | None, df: pd.DataFrame | None, limit: int = 4) -> list[str]:
    """Follow-up questions that fit the last answer's actual result."""
    if last is None or df is None or last.table is None or last.table.empty:
        return []
    table = last.table
    filtered_columns = set().union(*(spec.columns for spec in last.filters)) if last.filters else set()
    suggestions: list[str] = []

    if last.entity is not None and last.winner is not None:
        other = _other_category(df, {last.entity})
        if other is not None:
            suggestions.append(f"Show the top 5 {_spoken(other)} in that {_spoken(last.entity)}")

    measure = _measure_column(last)
    rankable = last.intent in ("top N", "ranking", "group comparison", "distribution", "count") or last.intent.startswith("follow-up")
    if measure is not None and rankable and len(table) > 1 and last.chart_x is not None:
        ascending_now = table[measure].dropna().is_monotonic_increasing
        suggestions.append("Sort it from highest to lowest" if ascending_now else "Sort it from lowest to highest")
        if len(table) > 3:
            suggestions.append("Show only the top 3")

    if last.chart_x in table.columns and last.chart_y in table.columns:
        if last.chart_kind == "line":
            suggestions.append("Plot it as a bar chart")
        elif (last.chart_y in ("count", "rows") or last.agg in ("sum", "count")) and len(table) <= MAX_PIE_SLICES \
                and (table[last.chart_y].dropna() >= 0).all() and last.chart_kind != "pie":  # fmt: skip
            suggestions.append("Plot it as a pie chart")
        elif last.chart_kind is None:
            suggestions.append("Plot it as a bar chart")

    if last.filters:
        other = _other_category(df, filtered_columns | {last.entity})
        if other is not None:
            suggestions.append(f"Now group them by {_spoken(other)}")
        single = [
            c for spec in last.filters for c in spec.conditions if c.operator == de.EQUALS
            and de.column_kind(df[c.column]) == de.CATEGORICAL
        ]  # fmt: skip
        if len(single) == 1:
            counts = df[single[0].column].dropna().astype(str).value_counts()
            others = [v for v in counts.index if v.casefold() != str(single[0].value).casefold()]
            if others:
                suggestions.append(f"What about {others[0]}?")
    return list(dict.fromkeys(suggestions))[:limit]
