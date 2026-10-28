"""Deterministic multi-step analysis planner (no LLM, no generated code).

A request such as "Find the top 5 cities by net revenue for Electronics and show
a bar chart" is turned into a PLAN: an ordered tuple of ``PlanStep`` objects.
Each step has one operation type from a fixed allowlist and plain-data
arguments (column names, numbers, validated filter conditions). The plan is
validated as a whole, then executed step by step with existing, validated
functions from ``data_engine``, ``metrics`` and ``charts``.

How a request becomes a plan (``build_plan``):
1. Split the request into clauses at ", " / " then " / " and <verb> ".
2. Classify each clause with fixed patterns: filter ("filter to Pune"), sort
   ("sort descending"), limit ("keep the top 3"), chart ("show a bar chart"),
   explain ("explain the result") or the ONE analysis clause.
   A clause that fits none of them stops planning: nothing is silently dropped.
3. The analysis clause is interpreted by the existing question router
   (read-only); its structured ``AnswerContext`` names the real columns,
   filters, metric and aggregation.
4. Steps are generated in a fixed order: validate_columns -> filter -> metric
   -> aggregate -> sort -> limit -> chart -> summarize.

Execution (``execute_plan``) re-validates the plan and every column, works on
copies (the dataset is never modified), stops at the first failed step and
never presents a result for a failed plan. Nothing touches SQL or databases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from askyourdata import charts, exports, metrics
from askyourdata import data_engine as de
from askyourdata import question_router as qr
from askyourdata.data_engine import AnalysisError, ChartData
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
    fmt,
    words,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The plan model
# ---------------------------------------------------------------------------

OPERATIONS = ("validate_columns", "filter", "metric", "aggregate", "sort", "limit", "chart", "summarize")
_ORDER = {op: i for i, op in enumerate(OPERATIONS)}
# The only arguments each operation accepts (required, optional).
_ARGS = {
    "validate_columns": ({"columns"}, set()),
    "filter": ({"spec"}, set()),
    "metric": ({"metric"}, {"discount_scale"}),
    "aggregate": ({"group_by", "value", "agg"}, {"freq"}),
    "sort": ({"by", "descending"}, set()),
    "limit": ({"n"}, set()),
    "chart": ({"kind"}, set()),
    "summarize": ({"explain"}, set()),
}
CHART_KINDS = ("bar", "line", "pie")
MEASURE = "measure"  # sort "by" value meaning the calculated value column
MAX_STEPS = 12
MAX_GROUP_COLUMNS = 2
PLANNABLE_INTENTS = ("group comparison", "top N", "ranking", "distribution", "count", "aggregate", "trend", "filter")

# Step and plan statuses
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
COMPLETED = "completed"
EMPTY = "empty"


@dataclass(frozen=True, eq=False)
class PlanStep:
    """One operation with plain-data arguments (never code)."""

    op: str
    args: dict
    description: str


@dataclass(frozen=True, eq=False)
class Plan:
    request: str  # the user's words
    interpretation: str  # what the planner understood, in plain English
    steps: tuple[PlanStep, ...]
    context: AnswerContext | None = None  # the router's reading of the analysis clause


class PlanValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class PlanStepError(Exception):
    """A step could not be executed; the message is shown to the user."""


@dataclass(frozen=True, eq=False)
class StepOutcome:
    step: PlanStep
    status: str  # done / failed / skipped
    message: str


@dataclass(frozen=True, eq=False)
class PlanReport:
    """What the UI shows: the plan, and what actually happened to each step."""

    plan: Plan
    outcomes: tuple[StepOutcome, ...]
    status: str  # completed / failed / empty
    source: str = "planner"  # "planner" (executed here) or "router" (single question)


@dataclass(eq=False)
class _State:
    data: pd.DataFrame  # rows in use (a copy)
    table: pd.DataFrame | None = None
    measure: Any = None  # value column of ``table``
    label: Any = None  # category / period column of ``table``
    figure: Any = None
    summary: str = ""
    notes: list = field(default_factory=list)
    formulas: list = field(default_factory=list)
    filter_lines: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _valid_int(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def validate_plan(plan: Plan, df: pd.DataFrame) -> None:
    """Raise PlanValidationError listing every problem; return quietly if valid."""
    errors: list[str] = []
    steps = plan.steps
    if not steps:
        errors.append("The plan has no steps.")
    if len(steps) > MAX_STEPS:
        errors.append(f"The plan has too many steps (max {MAX_STEPS}).")
    columns = set(df.columns)
    metric_label = None
    aggregate: dict | None = None
    seen: dict[str, int] = {}
    last_rank = -1
    for number, step in enumerate(steps, 1):
        if step.op not in _ORDER:
            errors.append(f"Step {number}: operation '{step.op}' is not allowed.")
            continue
        if not isinstance(step.args, dict):
            errors.append(f"Step {number}: arguments must be plain data.")
            continue
        required, optional = _ARGS[step.op]
        unknown = set(step.args) - required - optional
        missing = required - set(step.args)
        if unknown:
            errors.append(f"Step {number} ({step.op}): unknown argument(s) {', '.join(sorted(unknown))}.")
        if missing:
            errors.append(f"Step {number} ({step.op}): missing argument(s) {', '.join(sorted(missing))}.")
        if _ORDER[step.op] < last_rank:
            errors.append(f"Step {number} ({step.op}) is out of order.")
        last_rank = max(last_rank, _ORDER[step.op])
        seen[step.op] = seen.get(step.op, 0) + 1
        if step.op != "filter" and seen[step.op] > 1:
            errors.append(f"Operation '{step.op}' may appear only once.")
        if unknown or missing:
            continue
        a = step.args

        if step.op == "validate_columns":
            if not isinstance(a["columns"], (list, tuple)) or not a["columns"]:
                errors.append("validate_columns needs a list of columns.")
            else:
                for column in a["columns"]:
                    if column not in columns:
                        errors.append(f"Column '{column}' does not exist in this dataset.")
        elif step.op == "filter":
            spec = a["spec"]
            if not isinstance(spec, FilterSpec) or not spec.conditions:
                errors.append(f"Step {number}: a filter must be a validated filter specification.")
            else:
                for condition in spec.conditions:
                    if condition.column not in columns:
                        errors.append(f"Filter column '{condition.column}' does not exist in this dataset.")
                    if condition.operator not in de.OPERATORS:
                        errors.append(f"Filter operator '{condition.operator}' is not allowed.")
        elif step.op == "metric":
            if a["metric"] not in metrics.METRICS:
                errors.append(f"Metric '{a['metric']}' is not an approved metric.")
            else:
                metric_label = metrics.METRICS[a["metric"]].label
            if a.get("discount_scale") not in (None, metrics.DISCOUNT_PERCENT, metrics.DISCOUNT_FRACTION):
                errors.append("Unknown discount scale.")
        elif step.op == "aggregate":
            aggregate = a
            group_by = a["group_by"]
            if not isinstance(group_by, (list, tuple)) or len(group_by) > MAX_GROUP_COLUMNS:
                errors.append(f"Aggregation can group by at most {MAX_GROUP_COLUMNS} columns.")
            else:
                for column in group_by:
                    if column not in columns:
                        errors.append(f"Grouping column '{column}' does not exist in this dataset.")
            value = a["value"]
            if value is not None and value not in columns and value != metric_label:
                errors.append(f"Value column '{value}' does not exist and is not a calculated metric of this plan.")
            if a["agg"] not in de.AGGREGATIONS:
                errors.append(f"Aggregation '{a['agg']}' is not allowed.")
            if a.get("freq") is not None and a["freq"] not in de.TREND_FREQUENCIES:
                errors.append(f"Time period '{a['freq']}' is not allowed.")
            if a.get("freq") is not None and len(group_by) != 1:
                errors.append("A trend groups by exactly one date column.")
        elif step.op == "sort":
            if not isinstance(a["descending"], bool):
                errors.append("Sort direction must be true or false.")
            if not isinstance(a["by"], str) and a["by"] not in columns:
                errors.append("Sort column is not valid.")
        elif step.op == "limit":
            if not _valid_int(a["n"], 1, qr.MAX_TOP_N):
                errors.append(f"The number of rows must be a whole number between 1 and {qr.MAX_TOP_N}.")
        elif step.op == "chart":
            kind = a["kind"]
            if kind not in CHART_KINDS:
                errors.append(f"Chart type '{kind}' is not allowed.")
            elif aggregate is None or not aggregate["group_by"]:
                errors.append("A chart needs a result with one value per category or period.")
            elif kind == "line" and aggregate.get("freq") is None:
                errors.append("A line chart needs dates; use a bar chart for categories.")
            elif kind in ("bar", "pie") and aggregate.get("freq") is None and len(aggregate["group_by"]) != 1:
                errors.append("A chart needs exactly one grouping column.")
            elif kind == "pie" and aggregate["agg"] not in ("sum", "count"):
                errors.append(f"A pie chart shows parts of a whole, which doesn't work for {aggregate['agg']} values.")
        elif step.op == "summarize":
            if not isinstance(a["explain"], bool):
                errors.append("summarize.explain must be true or false.")
    if steps and steps[-1].op != "summarize":
        errors.append("A plan must end with a summary step.")
    if errors:
        raise PlanValidationError(errors)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _numeric_or_fail(data: pd.DataFrame, column: Any) -> None:
    if column not in data.columns:
        raise PlanStepError(f"Column '{column}' is not available at this step.")
    if not de.is_numeric(data[column]):
        raise PlanStepError(f"Column '{column}' is not numeric, so it can't be calculated with.")


def _run_step(step: PlanStep, state: _State, df: pd.DataFrame) -> str:
    """Execute one validated step on ``state``; return what happened."""
    a = step.args
    if step.op == "validate_columns":
        de.validate_columns(state.data, list(a["columns"]))  # re-checked at execution time
        return f"All {len(a['columns'])} column(s) exist."

    if step.op == "filter":
        before = len(state.data)
        state.data = de.filter_rows(state.data, list(a["spec"].conditions))
        state.filter_lines.append(a["spec"].description)
        return f"{a['spec'].description}: {fmt(len(state.data))} of {fmt(before)} rows kept."

    if step.op == "metric":
        result = metrics.compute_metric(df, a["metric"], discount_scale=a.get("discount_scale"))
        state.data = state.data.copy()
        state.data[result.label] = result.values.reindex(state.data.index)
        state.formulas.append(result.formula)
        state.notes.extend(result.notes)
        unknown = int(state.data[result.label].isna().sum())
        if unknown:
            state.notes.append(f"{unknown} row(s) have a missing or invalid input and are left out (not counted as 0).")
        return f"{result.formula}."

    if step.op == "aggregate":
        group_by, value, agg = list(a["group_by"]), a["value"], a["agg"]
        if value is not None:
            _numeric_or_fail(state.data, value)
        if a.get("freq"):
            trend = de.date_trend(state.data, group_by[0], value, freq=a["freq"], agg=agg)
            state.table, state.label, state.measure = trend.table, "period", trend.value_label
            if trend.skipped_rows:
                state.notes.append(f"{trend.skipped_rows} row(s) with a missing or unreadable date were left out.")
            return f"{trend.value_label} per {a['freq'].lower()} of {group_by[0]}: {len(trend.table)} periods."
        if group_by:
            if value is None:
                table = de.group_summary(state.data, group_by)
                measure = "rows"
            else:
                table = de.group_summary(state.data, group_by, value, [agg])
                measure = table.columns[-1]
            state.table, state.label, state.measure = table, group_by[0], measure
            return f"{measure} for each {' / '.join(map(str, group_by))}: {len(table)} groups."
        if value is None:
            state.table = pd.DataFrame({"measure": ["rows"], "value": [len(state.data)]})
            state.measure, state.label = "value", "measure"
            return f"{fmt(len(state.data))} rows."
        result = de.aggregate_column(state.data, value, agg)
        state.table = pd.DataFrame({"measure": [f"{agg} of {value}"], "value": [result.value]})
        state.measure, state.label = "value", "measure"
        if result.missing:
            state.notes.append(f"{result.missing} missing value(s) of {value} were ignored.")
        return f"{agg} of {value} = {fmt(result.value)} (from {fmt(result.used_values)} values)."

    if step.op == "sort":
        table = state.table if state.table is not None else state.data
        by = state.measure if a["by"] == MEASURE else a["by"]
        if by is None or by not in table.columns:
            raise PlanStepError(f"The result has no column '{a['by']}' to sort by.")
        sorted_table = de.sort_rows(table, by, ascending=not a["descending"]).reset_index(drop=True)
        if state.table is not None:
            state.table = sorted_table
        else:
            state.data = sorted_table
        direction = "highest to lowest" if a["descending"] else "lowest to highest"
        return f"Sorted by {by}, {direction} (missing values last)."

    if step.op == "limit":
        if state.table is not None:
            state.table = state.table.head(a["n"]).reset_index(drop=True)
            return f"Kept the first {len(state.table)} row(s)."
        state.data = state.data.head(a["n"])
        return f"Kept the first {len(state.data)} row(s)."

    if step.op == "chart":
        table, x, y, kind = state.table, state.label, state.measure, a["kind"]
        if table is None or x not in table.columns or y not in table.columns:
            raise PlanStepError("There is no category and value column to chart.")
        data = table.dropna(subset=[y])
        if data.empty:
            raise PlanStepError("Every value is missing, so there is nothing to chart.")
        chart = ChartData(data, x=x, y=y, title=f"{y} by {x}")
        if kind == "line":
            if not pd.api.types.is_datetime64_any_dtype(data[x]):
                raise PlanStepError("A line chart needs dates; use a bar chart for categories.")
            state.figure = charts.line_chart(replace(chart, data=data.sort_values(x)))
        elif kind == "pie":
            if (data[y] < 0).any():
                raise PlanStepError(f"'{y}' has negative values, so it can't be a pie chart.")
            state.figure = charts.pie_chart(chart)
        else:
            state.figure = charts.bar_chart(chart)
        return f"{kind.capitalize()} chart of {y} by {x}."

    if step.op == "summarize":
        state.summary = _summary(state, a["explain"])
        return "Summary written from the calculated numbers."

    raise PlanStepError(f"Operation '{step.op}' is not allowed.")  # unreachable after validation


def _summary(state: _State, explain: bool) -> str:
    """Plain-English facts taken only from the final table."""
    table, y, x = state.table, state.measure, state.label
    if table is None:
        return f"**{fmt(len(state.data))} rows** match."
    if x == "measure":
        return f"The {table.iloc[0]['measure']} is **{fmt(table.iloc[0]['value'])}**."
    values = table[y].dropna()
    if values.empty:
        return "Every calculated value is missing, so there is nothing to summarise."
    top = table.loc[values.idxmax()]
    low = table.loc[values.idxmin()]
    label = lambda row: fmt(row[x]) if x != "period" else row[x].strftime("%b %Y")  # noqa: E731
    lines = [f"Highest: **{label(top)}** with **{fmt(top[y])}**; lowest: {label(low)} with {fmt(low[y])} "
             f"({len(table)} rows in the result)."]  # fmt: skip
    if explain:
        total = values.sum()
        additive = y in ("rows",) or str(y).startswith(("sum of", "count of"))
        if additive and total > 0:
            lines.append(f"{label(top)} accounts for {top[y] / total * 100:.1f}% of the {fmt(total)} total shown.")
        if low[y] > 0 and top[y] != low[y]:
            lines.append(f"The highest value is {top[y] / low[y]:.1f}× the lowest.")
        if len(values) > 2:
            lines.append(f"The middle (median) value is {fmt(values.median())}.")
        if de.MISSING_LABEL in table[x].astype(str).tolist():
            lines.append("One group is '(missing)': rows where that column is empty.")
        lines.append(f"These figures come from {fmt(len(state.data))} row(s) after filtering.")
    return " ".join(lines)


def execute_plan(plan: Plan, df: pd.DataFrame) -> tuple[PlanReport, _State]:
    """Run a plan. Stops at the first failed step; never modifies ``df``."""
    validate_plan(plan, df)
    state = _State(data=df.copy())
    outcomes: list[StepOutcome] = []
    status = COMPLETED
    for index, step in enumerate(plan.steps):
        if status != COMPLETED:
            outcomes.append(StepOutcome(step, SKIPPED, "Not run because an earlier step stopped the plan."))
            continue
        try:
            message = _run_step(step, state, df)
        except (AnalysisError, PlanStepError, ClarificationNeeded, UnsupportedRequest) as exc:
            outcomes.append(StepOutcome(step, FAILED, str(exc)))
            status = FAILED
            continue
        outcomes.append(StepOutcome(step, DONE, message))
        if step.op == "filter" and state.data.empty:
            status = EMPTY  # nothing left to analyse: the rest is skipped, not failed
    return PlanReport(plan, tuple(outcomes), status), state


# ---------------------------------------------------------------------------
# Building a plan from a request
# ---------------------------------------------------------------------------

_VERBS = (
    r"show|display|plot|draw|chart|visuali[sz]e|make|create|add|sort|order|rank|explain|summari[sz]e"
    r"|describe|interpret|keep|limit|give|return|list|group|calculate|compute|filter|find|count|compare|total"
)
_SPLIT_RE = re.compile(rf",(?!\s*\d)\s*(?:and\s+|then\s+)?|\s+(?:and\s+)?then\s+|\s+and\s+(?=(?:{_VERBS})\b)")
_CHART_RE = re.compile(
    r"^(?:show|display|plot|draw|make|create|add|give|render|visuali[sz]e|with|include)?\s*(?:me\s+)?"
    r"(?:it\s+|them\s+|the\s+results?\s+)?(?:as\s+|in\s+|on\s+|with\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?P<kind>bar|column|line|pie|donut)?\s*(?:chart|graph|plot|visuali[sz]ation)s?(?:\s+of\s+(?:it|them|the\s+results?))?$"
)
_SORT_CLAUSE_RE = re.compile(
    r"^(?:sort|order|rank)(?:\s+(?:it|them|this|the\s+results?))?(?:\s+by\s+(?P<col>.+?))?"
    r"(?:\s+(?:in\s+)?(?P<dir>descending|ascending|desc|asc|(?:from\s+)?(?:highest|largest|biggest)\s+to\s+(?:lowest|smallest)"
    r"|(?:from\s+)?(?:lowest|smallest)\s+to\s+(?:highest|largest|biggest)|highest first|largest first|lowest first"
    r"|smallest first)(?:\s+order)?)?$"
)
_NUMBER = r"-?\d+(?:\.\d+)?|" + "|".join(qr._NUMBER_WORDS)
_LIMIT_CLAUSE_RE = re.compile(
    r"^(?:keep|show|limit(?:\s+it)?\s+to|return|list|give\s+me)?\s*(?:only\s+)?(?:the\s+)?"
    rf"(?P<dir>top|bottom|first)\s+(?P<n>{_NUMBER})(?:\s+(?:rows|results|ones))?$"
)
_EXPLAIN_RE = re.compile(
    r"^(?:explain|summari[sz]e|describe|interpret)(?:\s+(?:it|this|that|them|the\s+(?:results?|answer|findings)))?$"
)
_FILTER_CLAUSE_RE = re.compile(
    r"^(?:filter|restrict|narrow|limit)(?:\s+(?:it|the\s+data|the\s+rows|rows|orders))?\s+(?:down\s+)?"
    r"(?:to|for|on|by)\s+(?P<f>.+)$|^only\s+(?P<g>.+)$"
)
_GROUP_LEAD_RE = re.compile(r"^(?:group|aggregate)\s+(?P<rest>.+)$")
_REFERENCE_RE = re.compile(r"\b(?:that|those|these)\s+[a-z]")
_DESCENDING_WORDS = ("descending", "desc", "highest to", "largest to", "biggest to", "highest first", "largest first")


@dataclass
class _Clauses:
    analysis: str | None = None
    filters: list = field(default_factory=list)  # FilterSpec
    filter_texts: list = field(default_factory=list)
    sort: tuple | None = None  # (column phrase or None, descending or None)
    limit: tuple | None = None  # (n, largest)
    chart: str | None = None  # kind or "auto"
    explain: bool = False
    understood: list = field(default_factory=list)  # (kind, text) in request order


def _clean(text: str) -> str:
    text = re.sub(r"[^a-z0-9.\-,\s/]", " ", text.lower().replace("'", " "))
    return re.sub(r"\s+", " ", text).strip(" .,")


def split_clauses(question: str) -> list[str]:
    return [c.strip(" .") for c in _SPLIT_RE.split(_clean(question)) if c and c.strip(" .")]


def _filter_specs(df: pd.DataFrame, text: str) -> list:
    specs, rest = qr.parse_filters(df, text)
    if not specs:
        specs, rest = qr.parse_filters(df, f"rows from {text}")
    leftover = [w for w in words(rest) if w not in qr._STOPWORDS and w not in qr.ROW_WORDS]
    if not specs or leftover:
        raise ClarificationNeeded(
            f"I couldn't use “{text}” as a filter. Filters can be a value from a text column "
            "(e.g. Pune), a number condition (quantity above 2) or a date (in March 2026)."
        )
    return list(specs)


def _classify(df: pd.DataFrame, clauses: list[str]) -> _Clauses:
    found = _Clauses()
    for clause in clauses:
        clause = re.sub(r"^(?:and|also|then|please)\s+", "", clause)
        if m := _CHART_RE.match(clause):
            found.chart = {"column": "bar", "donut": "pie"}.get(m.group("kind"), m.group("kind")) or "auto"
            found.understood.append(("chart", clause))
        elif m := _SORT_CLAUSE_RE.match(clause):
            direction = m.group("dir")
            descending = None if direction is None else any(w in direction for w in _DESCENDING_WORDS)
            found.sort = (m.group("col"), descending)
            found.understood.append(("sort", clause))
        elif m := _LIMIT_CLAUSE_RE.match(clause):
            found.limit = (qr._parse_top_n(m.group("n")), m.group("dir") != "bottom")
            found.understood.append(("limit", clause))
        elif _EXPLAIN_RE.match(clause):
            found.explain = True
            found.understood.append(("explain", clause))
        elif m := _FILTER_CLAUSE_RE.match(clause):
            text = m.group("f") or m.group("g")
            found.filters.extend(_filter_specs(df, text))
            found.filter_texts.append(text)
            found.understood.append(("filter", clause))
        elif (specs := _only_filters(df, clause)) is not None:  # "orders from Pune paid by UPI"
            found.filters.extend(specs)
            found.filter_texts.append(clause)
            found.understood.append(("filter", clause))
        elif found.analysis is None:
            if m := _GROUP_LEAD_RE.match(clause):  # "group quantity by category"
                rest = m.group("rest")
                clause = f"count {rest}" if rest.startswith("by ") else f"total {rest}"
            found.analysis = clause
            found.understood.append(("analysis", clause))
        elif answer_question(df, clause).ok:
            raise UnsupportedRequest(
                f"I can only plan ONE calculation per request, but found a second one: “{clause}”. "
                "Please ask it separately."
            )
        else:
            raise UnsupportedRequest(
                f"I couldn't understand the part “{clause}”, so I won't run a plan without it. "
                "Supported extra steps: filters (“filter to Pune”), sorting (“sort descending”), "
                "top N (“keep the top 3”), charts (“show a bar chart”) and “explain the result”."
            )
    return found


def _only_filters(df: pd.DataFrame, clause: str) -> list | None:
    """The filters of a clause that contains nothing but filters (and row words), else None."""
    try:
        specs, rest = qr.parse_filters(df, clause)
    except (ClarificationNeeded, UnsupportedRequest):
        return None  # let the analysis step report the problem in context
    leftover = [w for w in words(rest) if w not in qr._STOPWORDS and w not in qr.ROW_WORDS
                and w not in ("show", "list", "find", "only", "all", "get")]  # fmt: skip
    return list(specs) if specs and not leftover else None


def is_multi_step(question: Any) -> bool:
    """True when the request has 2+ clauses: the planner, not the router, handles it."""
    return isinstance(question, str) and len(split_clauses(question)) >= 2


def follow_up_chain(question: str) -> list[tuple[str, str]] | None:
    """For "sort it descending and plot it as a pie chart": [(kind, canonical follow-up)].

    None unless EVERY clause is a presentation step (sort / limit / chart / explain),
    i.e. the request builds on the previous answer instead of asking a new calculation.
    """
    chain = []
    for clause in split_clauses(question):
        clause = re.sub(r"^(?:and|also|then|please|now)\s+", "", clause)
        if m := _CHART_RE.match(clause):
            kind = {"column": "bar", "donut": "pie"}.get(m.group("kind"), m.group("kind")) or "bar"
            chain.append(("chart", f"plot it as a {kind} chart"))
        elif m := _SORT_CLAUSE_RE.match(clause):
            direction = m.group("dir")
            text = "sort it"
            if m.group("col"):
                text += f" by {m.group('col')}"
            if direction is not None:
                descending = any(w in direction for w in _DESCENDING_WORDS)
                text += " from highest to lowest" if descending else " from lowest to highest"
            chain.append(("sort", text))
        elif m := _LIMIT_CLAUSE_RE.match(clause):
            chain.append(("limit", f"show only the {'bottom' if m.group('dir') == 'bottom' else 'top'} {m.group('n')}"))
        elif _EXPLAIN_RE.match(clause):
            chain.append(("explain", "explain"))
        else:
            return None
    return chain if len(chain) >= 2 else None


def explain_context(ctx: AnswerContext) -> str:
    """Explanation of a previous answer's table (for "… and explain it" follow-ups)."""
    table = ctx.table
    if table is None or table.empty or ctx.chart_x not in table.columns or ctx.chart_y not in table.columns:
        raise UnsupportedRequest("The previous answer has no table with one value per category to explain.")
    state = _State(data=table, table=table, measure=ctx.chart_y, label=ctx.chart_x)
    text = _summary(state, explain=True)
    return text.rsplit(" These figures come from", 1)[0]  # row counts of the original data are unknown here


def _interpretation(ctx: AnswerContext, clauses: _Clauses, chart_kind: str | None) -> str:
    parts = []
    if ctx.filters:
        parts.append("filter to " + "; ".join(s.description for s in ctx.filters))
    if ctx.metric:
        parts.append(f"calculate {metrics.METRICS[ctx.metric].label}")
    if ctx.intent == "filter":
        parts.append("list the matching rows")
    elif ctx.intent == "trend":
        parts.append(f"{ctx.agg} of {ctx.value or 'rows'} per {(ctx.freq or 'Month').lower()}")
    elif ctx.intent == "aggregate":
        parts.append(f"{ctx.agg} of {ctx.value}")
    else:
        what = f"{ctx.agg} of {ctx.value}" if ctx.value is not None else "number of rows"
        parts.append(f"{what} for each {' / '.join(map(str, ctx.groups or (ctx.entity,)))}")
    if clauses.sort or ctx.intent in ("top N", "ranking"):
        parts.append("sort")
    if clauses.limit or ctx.intent in ("top N", "ranking"):
        parts.append("keep the top rows")
    if chart_kind:
        parts.append(f"draw a {chart_kind} chart")
    parts.append("explain the result" if clauses.explain else "summarise the result")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."  # keep column names and values as they are


def _steps_from_context(ctx: AnswerContext, clauses: _Clauses, df: pd.DataFrame,
                        discount_scale: str | None) -> tuple[list[PlanStep], str | None]:  # fmt: skip
    """The ordered steps for a router-interpreted analysis plus the extra clauses."""
    if ctx.intent not in PLANNABLE_INTENTS or (ctx.intent == "distribution" and ctx.entity is None):
        raise UnsupportedRequest(
            "Multi-step plans support totals, averages, counts, group comparisons, rankings, top-N, "
            "trends and row listings. This analysis can't be combined with other steps; ask it on its own."
        )
    metric_label = metrics.METRICS[ctx.metric].label if ctx.metric else None
    value = metric_label if ctx.metric else ctx.value
    groups = list(ctx.groups) or ([ctx.entity] if ctx.entity is not None else [])
    if ctx.intent == "trend":
        groups = [ctx.date]
    if ctx.intent == "aggregate":
        groups = []
    agg = ctx.agg or ("count" if value is None else "sum")

    required = [c for spec in ctx.filters for c in spec.columns] + groups
    if value is not None and not ctx.metric:
        required.append(value)
    if ctx.metric:
        metric_result = metrics.compute_metric(df, ctx.metric, discount_scale=discount_scale)
        required += list(metric_result.columns.values())
    steps = [PlanStep("validate_columns", {"columns": list(dict.fromkeys(required))},
                      "Check that the required columns exist")] if required else []  # fmt: skip
    steps += [PlanStep("filter", {"spec": spec}, f"Filter: {spec.description}") for spec in ctx.filters]
    if ctx.metric:
        args = {"metric": ctx.metric}
        if discount_scale:
            args["discount_scale"] = discount_scale
        steps.append(PlanStep("metric", args, f"Calculate {metric_label}"))

    if ctx.intent != "filter":
        args = {"group_by": groups, "value": value, "agg": agg}
        if ctx.intent == "trend":
            args["freq"] = ctx.freq or "Month"
        what = f"{agg} of {value}" if value is not None else "number of rows"
        by = f" for each {' / '.join(map(str, groups))}" if groups else ""
        steps.append(PlanStep("aggregate", args, f"Calculate {what}{by}"))

    # Sorting: an explicit "sort ..." clause, or the ranking a top-N question implies.
    if clauses.sort is not None:
        column_phrase, descending = clauses.sort
        by = MEASURE
        if column_phrase:
            by = _sort_column(column_phrase, df, groups, value, ctx)
        if descending is None:
            descending = by == MEASURE
        steps.append(PlanStep("sort", {"by": by, "descending": descending},
                              f"Sort by {'the calculated value' if by == MEASURE else by}, "
                              f"{'highest' if descending else 'lowest'} first"))  # fmt: skip
    elif ctx.intent in ("top N", "ranking"):
        steps.append(PlanStep("sort", {"by": MEASURE, "descending": ctx.largest},
                              f"Sort by the calculated value, {'highest' if ctx.largest else 'lowest'} first"))  # fmt: skip

    limit = clauses.limit
    if limit is None and ctx.intent in ("top N", "ranking") and ctx.n:
        limit = (ctx.n, ctx.largest)
    if limit is not None:
        steps.append(PlanStep("limit", {"n": limit[0]}, f"Keep the first {limit[0]} rows"))

    chart_kind = clauses.chart
    if chart_kind == "auto":
        chart_kind = "line" if ctx.intent == "trend" else "bar"
    if chart_kind:
        steps.append(PlanStep("chart", {"kind": chart_kind}, f"Draw a {chart_kind} chart"))
    steps.append(PlanStep("summarize", {"explain": clauses.explain},
                          "Explain the result" if clauses.explain else "Summarise the result"))  # fmt: skip
    return steps, chart_kind


def _sort_column(phrase: str, df: pd.DataFrame, groups: list, value: Any, ctx: AnswerContext) -> Any:
    phrase_words = words(phrase)
    if value is not None and (set(phrase_words) & set(words(value)) or set(phrase_words) & {"value", "total", "result"}):
        return MEASURE
    if ctx.metric and set(phrase_words) & {"revenue", "amount", "net", "gross"}:
        return MEASURE
    found = ColumnMatcher(df).resolve(phrase_words, "column to sort by")
    if found.column in groups:
        return found.column
    if found.column == value:
        return MEASURE
    raise ClarificationNeeded(
        f"The result only has {', '.join(map(str, groups))} and the calculated value, so it can't be "
        f"sorted by {found.column}."
    )


def build_plan(df: pd.DataFrame, question: str) -> Plan:
    """Turn a multi-step request into a validated plan (raises to ask or refuse)."""
    clauses = _classify(df, split_clauses(question))
    if _REFERENCE_RE.search(clauses.analysis or ""):
        raise UnsupportedRequest(
            "References such as “that city” can't be combined with other steps yet. Ask the "
            "follow-up on its own first (e.g. “Show the top 5 products in that city”), then e.g. "
            "“plot it as a bar chart”."
        )
    analysis = clauses.analysis or ("show rows" if clauses.filters else None)
    if analysis is None:
        raise UnsupportedRequest(
            "This request has no calculation to plan (only "
            + ", ".join(kind for kind, _ in clauses.understood)
            + "). Ask a question such as “total quantity by category and show a bar chart”."
        )
    scale_match = qr._DISCOUNT_HINT_RE.search(question.lower())
    discount_scale = None
    if scale_match:
        discount_scale = (metrics.DISCOUNT_PERCENT if scale_match.group("scale").startswith("percent")
                          else metrics.DISCOUNT_FRACTION)  # fmt: skip

    interpreted = answer_question(df, analysis, base_filters=tuple(clauses.filters))  # read-only
    if not interpreted.ok or interpreted.context is None:
        raise _Interpretation(interpreted)
    ctx = interpreted.context
    if ctx.intent == "filter" and ctx.table is not None and ctx.table.empty and ctx.core.strip():
        # The filters leave no rows, so the router could not show which calculation was
        # asked for. Read the calculation once more without filters, only to learn its
        # shape; the filters are still applied first when the plan runs.
        shape = answer_question(df, ctx.core)
        if shape.ok and shape.context is not None and shape.context.intent != "filter":
            ctx = replace(shape.context, filters=ctx.filters, question=ctx.question)
    steps, chart_kind = _steps_from_context(ctx, clauses, df, discount_scale)
    plan = Plan(question, _interpretation(ctx, clauses, chart_kind), tuple(steps), ctx)
    validate_plan(plan, df)
    return plan


class _Interpretation(Exception):
    """The router could not interpret the analysis clause; carries its answer."""

    def __init__(self, answer: Answer) -> None:
        super().__init__(answer.text)
        self.answer = answer


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _context_for(plan: Plan, state: _State) -> AnswerContext | None:
    ctx = plan.context
    if ctx is None:
        return None
    chart = next((s for s in plan.steps if s.op == "chart"), None)
    limit = next((s for s in plan.steps if s.op == "limit"), None)
    return replace(
        ctx,
        question=plan.request,
        table=exports.display_table(state.table if state.table is not None else state.data),
        chart_x=state.label if state.label not in (None, "measure") else ctx.chart_x,
        chart_y=state.measure if state.label not in (None, "measure") else ctx.chart_y,
        chart_kind=chart.args["kind"] if chart else ctx.chart_kind,
        n=limit.args["n"] if limit else ctx.n,
    )


def answer_plan(df: pd.DataFrame, question: str) -> Answer:
    """Plan, validate and execute a multi-step request. Never raises."""
    try:
        plan = build_plan(df, question)
        report, state = execute_plan(plan, df)
    except _Interpretation as exc:
        answer = exc.answer
        return replace(answer, text="I couldn't plan the calculation part of this request. " + answer.text)
    except ClarificationNeeded as exc:
        return Answer(CLARIFY, exc.message, options=exc.options, rephrasings=exc.rephrasings)
    except UnsupportedRequest as exc:
        return Answer(UNSUPPORTED, exc.message, suggestions=tuple(qr.suggest_questions(df)))
    except PlanValidationError as exc:
        return Answer(ERROR, "The plan could not be validated: " + "; ".join(exc.errors))
    except AnalysisError as exc:
        return Answer(ERROR, f"I couldn't plan that: {exc}")
    except Exception:  # never show internal details
        logger.exception("Unexpected error while planning")
        return Answer(ERROR, "Something went wrong while planning. Please try a simpler request.")

    details = [f"Interpreted as: {plan.interpretation}"]
    details += [f"Formula: {f}" for f in state.formulas]
    if state.filter_lines:
        details.append("Filters applied: " + "; ".join(state.filter_lines))
    details += state.notes

    if report.status == FAILED:
        failed = next(o for o in report.outcomes if o.status == FAILED)
        number = report.outcomes.index(failed) + 1
        return Answer(
            ERROR,
            f"The plan stopped at step {number} ({failed.step.description}): {failed.message} "
            "No result is shown, because the remaining steps could not run.",
            intent="plan", details=tuple(details), plan=report,
        )  # fmt: skip

    prefix = ""
    if state.filter_lines:
        prefix = f"**Filters:** {'; '.join(state.filter_lines)} → **{fmt(len(state.data))} of {fmt(len(df))} rows**.\n\n"
    if report.status == EMPTY:
        return Answer(
            ANSWER, prefix + "No rows match the filters, so the remaining steps were skipped.",
            intent="plan", table=exports.display_table(state.data), details=tuple(details),
            plan=report, context=_context_for(plan, state),
        )  # fmt: skip
    table = state.table if state.table is not None else state.data
    return Answer(
        ANSWER, prefix + state.summary, intent="plan", table=exports.display_table(table),
        figure=state.figure, details=tuple(details), plan=report, context=_context_for(plan, state),
    )  # fmt: skip


def plan_from_context(ctx: AnswerContext | None) -> PlanReport | None:
    """The steps a single router answer already performed, for display.

    Only for answers that combined several operations (filters and/or a metric
    with a calculation); simple answers return None."""
    if ctx is None or ctx.intent not in PLANNABLE_INTENTS or not (ctx.filters or ctx.metric):
        return None
    steps = [PlanStep("filter", {"spec": spec}, f"Filter: {spec.description}") for spec in ctx.filters]
    if ctx.metric:
        steps.append(PlanStep("metric", {"metric": ctx.metric}, f"Calculate {metrics.METRICS[ctx.metric].label}"))
    if ctx.intent != "filter":
        groups = ctx.groups or ((ctx.entity,) if ctx.entity is not None else ())
        what = f"{ctx.agg} of {ctx.value}" if ctx.value is not None else "number of rows"
        by = f" for each {' / '.join(map(str, groups))}" if groups and ctx.intent != "aggregate" else ""
        steps.append(PlanStep("aggregate", {"group_by": list(groups), "value": ctx.value, "agg": ctx.agg or "count"},
                              f"Calculate {what}{by}"))  # fmt: skip
    if ctx.intent in ("top N", "ranking"):
        steps.append(PlanStep("sort", {"by": MEASURE, "descending": ctx.largest}, "Sort by the calculated value"))
        if ctx.n:
            steps.append(PlanStep("limit", {"n": ctx.n}, f"Keep the first {ctx.n} rows"))
    steps.append(PlanStep("summarize", {"explain": False}, "Summarise the result"))
    plan = Plan(ctx.question, "", tuple(steps), ctx)
    outcomes = tuple(StepOutcome(step, DONE, "") for step in steps)
    return PlanReport(plan, outcomes, COMPLETED, source="router")
