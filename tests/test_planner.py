"""Tests for the deterministic multi-step analysis planner."""

import ast
import builtins
from pathlib import Path

import pandas as pd
import pytest

from askyourdata import planner
from askyourdata import data_engine as de
from askyourdata.conversation import PLAN, PLAN_FOLLOW_UP, Conversation
from askyourdata.data_engine import Condition
from askyourdata.nl_filters import FilterSpec
from askyourdata.planner import (
    COMPLETED,
    DONE,
    EMPTY,
    FAILED,
    SKIPPED,
    Plan,
    PlanStep,
    PlanValidationError,
    answer_plan,
    build_plan,
    execute_plan,
    validate_plan,
)
from askyourdata.question_router import ANSWER, CLARIFY, ERROR, UNSUPPORTED

ROOT = Path(__file__).parent.parent


@pytest.fixture
def shop():
    return pd.DataFrame(
        {
            "Order_Date": [
                "2026-01-05", "2026-01-20", "2026-02-03", "2026-02-14", "2026-03-01",
                "2026-03-15", "2026-03-20", "2026-03-28", "2026-03-30", "2026-03-31",
            ],
            "City": ["Pune", "Pune", "Pune", "Delhi", "Delhi", "Mumbai", "Pune", "Delhi", None, "Mumbai"],
            "Product": ["Phone", "Shirt", "Lamp", "Phone", "Lamp", "Shirt", "Phone", "Mug", "Phone", "Lamp"],
            "Category": ["Electronics", "Clothing", "Home", "Electronics", "Home", "Clothing",
                         "Electronics", "Home", "Electronics", "Home"],
            "Quantity": [3, 1, 4, 2, 5, 6, 2, 1, 7, None],
            "Unit_Price": [100.0, 20.0, 40.0, 100.0, 40.0, 20.0, 100.0, 5.0, 100.0, 40.0],
            "Discount_Pct": [10, 0, 5, 20, 0, 5, 10, 0, 0, 5],
            "Payment_Method": ["UPI", "Card", "UPI", "Cash", "Card", "UPI", "Card", "UPI", "Cash", "UPI"],
        }
    )  # fmt: skip


def net(df):
    return df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Pct"] / 100)


def ops(plan_or_report):
    steps = plan_or_report.steps if isinstance(plan_or_report, Plan) else [o.step for o in plan_or_report.outcomes]
    return [s.op for s in steps]


# --- splitting and recognising multi-step requests ------------------------------------------


@pytest.mark.parametrize(
    ("question", "clauses"),
    [
        ("Find the top 5 cities by net revenue for Electronics and show a bar chart.",
         ["find the top 5 cities by net revenue for electronics", "show a bar chart"]),
        ("Filter to Pune, group quantity by category, and sort descending.",
         ["filter to pune", "group quantity by category", "sort descending"]),
        ("Calculate total quantity by category and explain the result.",
         ["calculate total quantity by category", "explain the result"]),
        ("orders between 2026-01-01 and 2026-02-28", ["orders between 2026-01-01 and 2026-02-28"]),
        ("average unit price per city and category", ["average unit price per city and category"]),
        ("orders on March 31, 2026", ["orders on march 31, 2026"]),
    ],
)  # fmt: skip
def test_split_clauses(question, clauses):
    assert planner.split_clauses(question) == clauses


@pytest.mark.parametrize(
    "question",
    ["Show total quantity by category", "Compare net revenue by city for orders paid by UPI.",
     "Is quantity correlated with unit price", "Plot it as a pie chart"],
)  # fmt: skip
def test_single_questions_are_not_plans(question):
    assert not planner.is_multi_step(question)


# --- plan construction ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_ops"),
    [
        ("Find the top 2 cities by net revenue for Electronics and show a bar chart.",
         ["validate_columns", "filter", "metric", "aggregate", "sort", "limit", "chart", "summarize"]),
        ("Filter to Pune, group quantity by category, and sort descending.",
         ["validate_columns", "filter", "aggregate", "sort", "summarize"]),
        ("Show the top 3 products by quantity and display a chart.",
         ["validate_columns", "aggregate", "sort", "limit", "chart", "summarize"]),
        ("Calculate total quantity by category and explain the result.",
         ["validate_columns", "aggregate", "summarize"]),
        ("Filter to UPI and show the order trend by month and show a line chart",
         ["validate_columns", "filter", "aggregate", "chart", "summarize"]),
        ("Filter to Pune and show a bar chart", None),  # rows can't be charted -> invalid
    ],
)  # fmt: skip
def test_plan_steps_and_order(shop, question, expected_ops):
    if expected_ops is None:
        with pytest.raises(PlanValidationError, match="A chart needs"):
            build_plan(shop, question)
        return
    plan = build_plan(shop, question)
    assert ops(plan) == expected_ops
    assert all(step.op in planner.OPERATIONS for step in plan.steps)


def test_plan_arguments_are_plain_data(shop):
    plan = build_plan(shop, "Find the top 2 cities by net revenue for Electronics and show a pie chart")
    allowed = (str, int, bool, type(None), list, tuple, FilterSpec)
    for step in plan.steps:
        for value in step.args.values():
            assert isinstance(value, allowed), (step.op, value)
    aggregate = next(s for s in plan.steps if s.op == "aggregate")
    assert aggregate.args == {"group_by": ["City"], "value": "Net revenue (calculated)", "agg": "sum"}
    assert next(s for s in plan.steps if s.op == "limit").args == {"n": 2}
    assert "Filter to `Category` is Electronics" in plan.interpretation


# --- validation: allowed and disallowed plans ----------------------------------------------------


def _plan(*steps):
    return Plan("test", "test", tuple(steps))


SUMMARY = PlanStep("summarize", {"explain": False}, "summary")
AGG = PlanStep("aggregate", {"group_by": ["Category"], "value": "Quantity", "agg": "sum"}, "agg")


def test_valid_hand_made_plan(shop):
    validate_plan(_plan(AGG, PlanStep("chart", {"kind": "bar"}, "chart"), SUMMARY), shop)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((PlanStep("eval", {"code": "1+1"}, "x"), SUMMARY), "operation 'eval' is not allowed"),
        ((PlanStep("sql", {"query": "DROP TABLE x"}, "x"), SUMMARY), "operation 'sql' is not allowed"),
        ((PlanStep("aggregate", {"group_by": ["Category"], "value": "Quantity", "agg": "sum", "code": "x"}, "x"),
          SUMMARY), "unknown argument"),
        ((PlanStep("aggregate", {"group_by": ["Category"]}, "x"), SUMMARY), "missing argument"),
        ((AGG, PlanStep("filter", {"spec": FilterSpec("x", (Condition("City", de.EQUALS, "Pune"),))}, "f"),
          SUMMARY), "out of order"),
        ((AGG, AGG, SUMMARY), "may appear only once"),
        ((AGG,), "must end with a summary"),
        ((PlanStep("validate_columns", {"columns": ["Nope"]}, "v"), SUMMARY), "'Nope' does not exist"),
        ((PlanStep("aggregate", {"group_by": ["Category"], "value": "Quantity", "agg": "exec"}, "a"), SUMMARY),
         "Aggregation 'exec' is not allowed"),
        ((PlanStep("filter", {"spec": "City == 'Pune'"}, "f"), SUMMARY), "validated filter"),
        ((PlanStep("filter", {"spec": FilterSpec("x", (Condition("City", "__import__", "os"),))}, "f"), SUMMARY),
         "operator '__import__' is not allowed"),
        ((PlanStep("metric", {"metric": "profit"}, "m"), SUMMARY), "not an approved metric"),
        ((AGG, PlanStep("chart", {"kind": "3d"}, "c"), SUMMARY), "Chart type '3d' is not allowed"),
        ((AGG, PlanStep("chart", {"kind": "line"}, "c"), SUMMARY), "line chart needs dates"),
        ((PlanStep("aggregate", {"group_by": ["Category"], "value": "Quantity", "agg": "mean"}, "a"),
          PlanStep("chart", {"kind": "pie"}, "c"), SUMMARY), "parts of a whole"),
        ((PlanStep("chart", {"kind": "bar"}, "c"), SUMMARY), "A chart needs"),
        ((PlanStep("aggregate", {"group_by": ["City", "Category", "Product"], "value": None, "agg": "count"}, "a"),
          SUMMARY), "at most 2"),
        ((PlanStep("aggregate", {"group_by": ["Category"], "value": "Net revenue (calculated)", "agg": "sum"}, "a"),
          SUMMARY), "not a calculated metric of this plan"),
    ],
)  # fmt: skip
def test_invalid_plans_are_rejected(shop, steps, message):
    with pytest.raises(PlanValidationError) as info:
        validate_plan(_plan(*steps), shop)
    assert any(message in error for error in info.value.errors), info.value.errors


@pytest.mark.parametrize("n", [0, -3, 101, 2.5, "5", True, None])
def test_invalid_limits_are_rejected(shop, n):
    with pytest.raises(PlanValidationError, match="whole number between 1 and 100"):
        validate_plan(_plan(AGG, PlanStep("limit", {"n": n}, "l"), SUMMARY), shop)


@pytest.mark.parametrize("bad", ["keep the top 0", "keep the top 500", "keep the top 2.5"])
def test_invalid_top_n_in_a_request(shop, bad):
    answer = answer_plan(shop, f"Show total quantity by category and {bad}")
    assert answer.status == ERROR
    assert "whole number between 1 and 100" in answer.text


# --- execution: results match pandas ----------------------------------------------------------------


def test_top_cities_by_net_revenue_for_electronics(shop):
    answer = answer_plan(shop, "Find the top 2 cities by net revenue for Electronics and show a bar chart.")
    assert answer.status == ANSWER and answer.plan.status == COMPLETED
    electronics = shop[shop["Category"] == "Electronics"]
    city = electronics["City"].fillna("(missing)")  # rows without a city form their own group
    expected = net(electronics).groupby(city).sum().nlargest(2)
    table = answer.table
    assert table["City"].tolist() == list(expected.index) == ["(missing)", "Pune"]
    assert table["sum of Net revenue (calculated)"].tolist() == pytest.approx(expected.tolist())
    assert answer.figure.data[0].type == "bar"
    assert any(d.startswith("Formula: Net revenue (calculated) = `Quantity`") for d in answer.details)
    assert "**Filters:** `Category` is Electronics" in answer.text


def test_filter_group_sort(shop):
    answer = answer_plan(shop, "Filter to Pune, group quantity by category, and sort descending.")
    pune = shop[shop["City"] == "Pune"]
    expected = pune.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
    assert answer.table["Category"].tolist() == list(expected.index)
    assert answer.table["sum of Quantity"].tolist() == expected.tolist()
    assert [o.status for o in answer.plan.outcomes] == [DONE] * 5


def test_sort_ascending_by_category_name(shop):
    answer = answer_plan(shop, "Show total quantity by category, and sort by category ascending")
    assert answer.table["Category"].tolist() == ["Clothing", "Electronics", "Home"]


def test_top_products_with_chart(shop):
    answer = answer_plan(shop, "Show the top 3 products by quantity and display a chart.")
    expected = shop.groupby("Product")["Quantity"].sum().nlargest(3)
    assert answer.table["Product"].tolist() == list(expected.index)
    assert answer.figure is not None


def test_explain_uses_only_calculated_numbers(shop):
    answer = answer_plan(shop, "Calculate total quantity by category and explain the result.")
    totals = shop.groupby("Category")["Quantity"].sum()
    share = totals.max() / totals.sum() * 100
    assert f"{totals.idxmax()} accounts for {share:.1f}% of the {totals.sum():.0f} total shown" in answer.text
    assert f"{len(shop)} row(s) after filtering" in answer.text


def test_trend_plan_with_line_chart(shop):
    answer = answer_plan(shop, "Filter to UPI, show the order trend by month and show a line chart")
    upi = shop[shop["Payment_Method"] == "UPI"]
    assert answer.table["rows"].sum() == len(upi)
    assert answer.figure.data[0].type == "scatter"


def test_filter_phrase_clause_then_calculation(shop):
    answer = answer_plan(shop, "orders from Pune paid by UPI, total quantity by category and show a bar chart")
    rows = shop[(shop["City"] == "Pune") & (shop["Payment_Method"] == "UPI")]
    assert answer.table.set_index("Category")["sum of Quantity"].to_dict() == \
        rows.groupby("Category")["Quantity"].sum().to_dict()  # fmt: skip


def test_steps_run_in_plan_order(shop):
    answer = answer_plan(shop, "Find the top 2 cities by net revenue for Electronics and show a pie chart")
    assert ops(answer.plan) == ops(answer.plan.plan)
    assert answer.plan.outcomes[1].message.startswith("`Category` is Electronics: 4 of 10 rows kept")


def test_single_question_answers_show_router_steps(shop):
    c = Conversation("fp")
    turn = c.ask(shop, "Compare net revenue by city for orders paid by UPI.")
    report = turn.answer.plan
    assert turn.answer.ok and report.source == "router"
    assert ops(report) == ["filter", "metric", "aggregate", "summarize"]
    assert c.ask(shop, "Show total quantity by category").answer.plan is None  # nothing combined


# --- failures stop the plan ------------------------------------------------------------------------


def test_failed_step_stops_the_plan_and_shows_no_result(shop, monkeypatch):
    original = planner._run_step

    def failing_aggregate(step, state, df):  # only the execution of step 3 fails
        if step.op == "aggregate":
            raise de.AnalysisError("simulated failure")
        return original(step, state, df)

    monkeypatch.setattr(planner, "_run_step", failing_aggregate)
    answer = answer_plan(shop, "Filter to Pune, group quantity by category, and show a bar chart")
    assert answer.status == ERROR
    assert answer.table is None and answer.figure is None
    statuses = [o.status for o in answer.plan.outcomes]
    assert statuses == [DONE, DONE, FAILED, SKIPPED, SKIPPED]
    assert "stopped at step 3" in answer.text and "simulated failure" in answer.text


def test_execution_revalidates_types(shop):
    plan = _plan(PlanStep("aggregate", {"group_by": ["City"], "value": "Product", "agg": "sum"}, "a"), SUMMARY)
    report, _ = execute_plan(plan, shop)
    assert report.status == FAILED
    assert "not numeric" in report.outcomes[0].message
    assert report.outcomes[1].status == SKIPPED


def test_negative_values_stop_a_pie_chart(shop):
    df = shop.assign(Quantity=[3, 1, 4, 2, 5, -50, 2, 1, 7, 1])  # Mumbai's total is negative
    answer = answer_plan(df, "Show total quantity by city and show a pie chart")
    assert answer.status == ERROR and "negative" in answer.text and answer.figure is None


def test_validation_runs_again_before_execution(shop):
    plan = build_plan(shop, "Show total quantity by category and show a bar chart")
    with pytest.raises(PlanValidationError):
        execute_plan(plan, shop.drop(columns="Category"))  # the data changed since planning


# --- empty results, missing values ------------------------------------------------------------------


def test_empty_filter_result_skips_the_rest(shop):
    answer = answer_plan(shop, "Filter to Mumbai and Cash, show total quantity by category and show a bar chart")
    assert answer.status == ANSWER and answer.plan.status == EMPTY
    assert "No rows match" in answer.text and answer.figure is None
    assert [o.status for o in answer.plan.outcomes][-3:] == [SKIPPED, SKIPPED, SKIPPED]


def test_missing_values_are_reported_not_hidden(shop):
    answer = answer_plan(shop, "Show net revenue by city and sort descending")
    assert "(missing)" in answer.table["City"].tolist()  # the row without a city is its own group
    assert any("left out (not counted as 0)" in d for d in answer.details)  # Quantity is missing once


# --- clarification and unsupported requests ------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "status", "message"),
    [
        ("Filter to Kolkata and show total quantity by category", CLARIFY, "kolkata"),
        ("Show total revenue_x by category and show a bar chart", CLARIFY, "couldn't plan the calculation"),
        ("Total quantity by category, and then dance", UNSUPPORTED, "couldn't understand the part “dance”"),
        ("Total quantity by category and show net revenue by city", UNSUPPORTED, "ONE calculation"),
        ("Show the correlation between numeric columns and show a bar chart", UNSUPPORTED, "can't be combined"),
        ("Sort descending and show a bar chart", UNSUPPORTED, "no calculation to plan"),
    ],
)
def test_unclear_or_unsupported_requests(shop, question, status, message):
    answer = answer_plan(shop, question)
    assert answer.status == status, answer.text
    assert message in answer.text
    assert answer.table is None and answer.figure is None


def test_ambiguous_column_in_a_plan_asks():
    df = pd.DataFrame({"category": ["a", "b"], "unit_price": [1.0, 2.0], "list_price": [3.0, 4.0]})
    answer = answer_plan(df, "Average price by category and show a bar chart")
    assert answer.status == CLARIFY and "more than one column" in answer.text


# --- conversation, dataset switching, follow-up chains -----------------------------------------------------


def test_plan_answers_support_follow_ups(shop):
    c = Conversation("fp")
    first = c.ask(shop, "Filter to Pune, group quantity by category, and sort descending.")
    assert first.follow_up == PLAN
    again = c.ask(shop, "Sort it from lowest to highest").answer
    assert again.ok and again.table["sum of Quantity"].is_monotonic_increasing


def test_follow_up_chain(shop):
    c = Conversation("fp")
    c.ask(shop, "Show total quantity by category")
    turn = c.ask(shop, "Sort it from lowest to highest and plot it as a pie chart")
    assert turn.follow_up == PLAN_FOLLOW_UP
    assert turn.answer.ok and turn.answer.figure.data[0].type == "pie"
    assert [o.status for o in turn.answer.plan.outcomes] == [DONE, DONE]


def test_follow_up_chain_stops_at_an_impossible_step(shop):
    c = Conversation("fp")
    c.ask(shop, "Show total quantity by category")
    turn = c.ask(shop, "sort it descending and plot it as a line chart and explain it")
    assert turn.answer.status == UNSUPPORTED
    assert turn.answer.table is None and turn.answer.figure is None
    assert [o.status for o in turn.answer.plan.outcomes] == [DONE, FAILED, SKIPPED]


def test_follow_up_chain_without_context(shop):
    turn = Conversation("fp").ask(shop, "sort it descending and plot it as a pie chart")
    assert turn.answer.status == CLARIFY and "no earlier answer" in turn.answer.text


def test_dataset_switch_clears_plan_context(shop):
    c = Conversation("a")
    c.ask(shop, "Filter to Pune, group quantity by category, and sort descending.")
    assert c.reset_if_dataset_changed("b")
    turn = c.ask(shop, "Sort it from lowest to highest and plot it as a pie chart")
    assert turn.answer.status == CLARIFY and "no earlier answer" in turn.answer.text


def test_references_cannot_be_combined_with_other_steps(shop):
    c = Conversation("fp")
    c.ask(shop, "Which city has the most orders?")
    answer = c.ask(shop, "Show the top 5 products in that city and show a bar chart").answer
    assert answer.status == UNSUPPORTED and "on its own first" in answer.text


# --- safety and immutability ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["Filter to Pune; DROP TABLE sales, and show a bar chart",
     "total quantity by category and show a chart -- comment",
     "filter to __import__('os'), total quantity by category",
     "delete orders from Pune and show a bar chart"],
)  # fmt: skip
def test_unsafe_multi_step_requests_are_refused(shop, question):
    original = shop.copy(deep=True)
    turn = Conversation("fp").ask(shop, question)
    assert turn.answer.status == UNSUPPORTED
    pd.testing.assert_frame_equal(shop, original)


def test_plans_never_modify_the_dataset(shop):
    original = shop.copy(deep=True)
    for question in [
        "Find the top 2 cities by net revenue for Electronics and show a bar chart.",
        "Filter to Pune, group quantity by category, and sort descending.",
        "Show net revenue by city and sort descending",
        "Filter to UPI, show the order trend by month and show a line chart",
    ]:
        answer_plan(shop, question)
    pd.testing.assert_frame_equal(shop, original)


def test_planning_never_calls_eval_or_exec(shop, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eval/exec must never be called")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    assert answer_plan(shop, "Find the top 2 cities by net revenue for Electronics and show a bar chart.").ok


def test_planner_source_has_no_dynamic_code_or_database_access():
    tree = ast.parse((ROOT / "askyourdata" / "planner.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__", "open", "getattr"}
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name and ("sql" in name or "database" in name or "importlib" in name) for name in imported)


def test_answer_plan_never_raises(shop, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(planner, "split_clauses", broken)
    answer = answer_plan(shop, "Show total quantity by category and show a bar chart")
    assert answer.status == ERROR and "secret" not in answer.text
