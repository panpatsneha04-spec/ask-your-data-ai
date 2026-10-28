"""Tests for conversational follow-up questions (deterministic, no LLM)."""

import ast
import builtins
from pathlib import Path

import pandas as pd
import pytest

from askyourdata import conversation as conv
from askyourdata.conversation import (
    CHART,
    MAX_TURNS,
    REFERENCE,
    REGROUP,
    SORT,
    SWAP_VALUE,
    TOP_N,
    Conversation,
    suggest_followups,
)
from askyourdata.question_router import ANSWER, CLARIFY, ERROR, UNSUPPORTED, answer_question

ROOT = Path(__file__).parent.parent


@pytest.fixture
def shop():
    return pd.DataFrame(
        {
            "Order_Date": [
                "2026-01-05", "2026-01-20", "2026-02-03", "2026-02-14",
                "2026-03-01", "2026-03-15", "2026-03-20", "2026-03-28", "2026-03-30",
            ],
            "City": ["Pune", "Pune", "Pune", "Delhi", "Delhi", "Mumbai", "Pune", "Delhi", None],
            "Product": ["Pen", "Ink", "Pad", "Pen", "Pad", "Ink", "Pen", "Mug", "Pen"],
            "Category": ["Office", "Office", "Paper", "Office", "Paper", "Office", "Office", "Home", "Office"],
            "Quantity": [3, 1, 4, 2, 5, 6, 2, 1, 7],
            "Unit_Price": [2.0, 9.0, 4.0, 2.0, 4.0, 9.0, 2.0, 6.0, 2.0],
            "Payment_Method": ["UPI", "Card", "UPI", "Cash", "Card", "UPI", "Card", "UPI", "Cash"],
        }
    )  # fmt: skip


def chat(df, *questions):
    """Ask questions in one conversation; return the turns."""
    c = Conversation("fp")
    return c, [c.ask(df, q) for q in questions]


# --- follow-up recognition ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "follow_up", "kind"),
    [
        ("Which city has the most orders?", "Show the top 5 products in that city.", REFERENCE),
        ("Show total quantity by category", "Sort it from highest to lowest.", SORT),
        ("Show total quantity by category", "lowest first", SORT),
        ("Show total quantity by category", "reverse the order", SORT),
        ("Show total quantity by category", "Plot it as a pie chart.", CHART),
        ("Show total quantity by category", "pie chart instead", CHART),
        ("Show total quantity by category", "show it as a table", CHART),
        ("Find the top 3 products by quantity", "Show only the top 2.", TOP_N),
        ("Find the top 3 products by quantity", "just the bottom 1", TOP_N),
        ("Show orders from Pune", "Now group them by category.", REGROUP),
        ("Show orders from Pune", "break it down by product", REGROUP),
        ("Show orders from Pune", "What about Delhi?", SWAP_VALUE),
        ("Show orders from Pune", "same for Mumbai", SWAP_VALUE),
    ],
)
def test_follow_ups_are_recognised(shop, first, follow_up, kind):
    _, turns = chat(shop, first, follow_up)
    assert turns[0].answer.ok
    assert turns[1].follow_up == kind
    assert turns[1].answer.status == ANSWER, turns[1].answer.text


@pytest.mark.parametrize(
    "question",
    [
        "Order trend by month",  # starts with "order" but is not a sort
        "count by category",  # a fresh group-by, not a regroup
        "Show total quantity by category as a bar chart",  # a full question
        "What about missing values?",  # no filter value: a fresh question
        "Which city has the most orders?",
        "How many rows are in this dataset?",  # "this dataset" is not a reference
    ],
)
def test_ordinary_questions_are_not_treated_as_follow_ups(shop, question):
    _, turns = chat(shop, "Show orders from Pune", question)
    assert turns[1].follow_up is None
    assert turns[1].answer.ok, turns[1].answer.text
    assert "Filters" not in turns[1].answer.text  # nothing carried over


@pytest.mark.parametrize("question", ["Sort it", "Show only the top 3", "Plot it as a bar chart",
                                      "Now group them by city", "What about Delhi?",
                                      "Show the top 5 products in that city"])  # fmt: skip
def test_follow_up_without_previous_answer_asks_for_a_full_question(shop, question):
    _, turns = chat(shop, question)
    assert turns[0].answer.status == CLARIFY
    assert "no earlier answer" in turns[0].answer.text


# --- references: "that city", "those products" ---------------------------------------------


def test_that_city_uses_the_ranking_winner(shop):
    _, turns = chat(shop, "Which city has the most orders?", "Show the top 5 products in that city.")
    answer = turns[1].answer
    assert "`City` is Pune" in answer.text  # Pune has 4 orders
    expected = shop[shop["City"] == "Pune"]["Product"].value_counts()
    assert answer.table.set_index("Product")["count"].to_dict() == expected.to_dict()
    assert "“in that city” was read as `City` is Pune" in answer.details[0]


def test_that_city_after_a_tie_asks_which(shop):
    df = shop.assign(City=["Pune", "Delhi", "Pune", "Delhi", "Pune", "Delhi", "Pune", "Delhi", "Mumbai"])
    _, turns = chat(df, "Which city has the most orders?", "Show the top 5 products in that city")
    assert "tie" in turns[0].answer.text  # Pune 4, Delhi 4
    answer = turns[1].answer
    assert answer.status == CLARIFY and "Which city do you mean?" in answer.text
    assert set(answer.options) == {"Pune", "Delhi"}  # only the tied cities, never Mumbai
    assert all("in " in r for r in answer.rephrasings)


def test_that_product_with_several_candidates_offers_rephrasings(shop):
    c, turns = chat(shop, "Find the top 3 products by quantity", "What is the total quantity for that product?")
    answer = turns[1].answer
    assert answer.status == CLARIFY
    listed = turns[0].answer.table["Product"].tolist()
    assert list(answer.options) == listed
    picked = c.ask(shop, answer.rephrasings[0]).answer
    assert picked.ok
    total = shop.loc[shop["Product"] == listed[0], "Quantity"].sum()
    assert f"**{total}**" in picked.text


def test_those_products_uses_every_listed_value(shop):
    _, turns = chat(shop, "Find the top 2 products by quantity", "Show the total quantity for those products")
    top2 = shop.groupby("Product")["Quantity"].sum().nlargest(2)
    answer = turns[1].answer
    assert answer.ok
    assert f"**{top2.sum()}**" in answer.text
    assert answer.context.filters[-1].conditions[0].value == list(top2.index)


def test_that_city_from_a_previous_filter(shop):
    _, turns = chat(shop, "Show orders from Delhi", "What is the average quantity in that city?")
    expected = shop.loc[shop["City"] == "Delhi", "Quantity"].mean()
    assert f"**{expected:.2f}**" in turns[1].answer.text


def test_reference_to_something_not_in_the_previous_result(shop):
    _, turns = chat(shop, "Show total quantity by category", "Show the trend for that city")
    answer = turns[1].answer
    assert answer.status == CLARIFY
    assert "doesn't show which city" in answer.text


@pytest.mark.parametrize("question", ["Show the top 5 products in that warehouse",
                                      "total quantity for those stores"])  # fmt: skip
def test_unresolvable_reference_is_never_silently_dropped(shop, question):
    _, turns = chat(shop, "Which city has the most orders?", question)
    answer = turns[1].answer
    assert answer.status == CLARIFY
    assert "I can't tell what" in answer.text


@pytest.mark.parametrize("question", ["How many rows are in this dataset?", "What is the trend this month"])
def test_non_reference_words_after_this_are_left_alone(shop, question):
    _, turns = chat(shop, "Which city has the most orders?", question)
    assert turns[1].follow_up is None


def test_missing_value_is_never_used_as_a_reference(shop):
    df = shop.assign(City=[None] * 6 + ["Pune"] * 3)
    _, turns = chat(df, "Which city has the most orders?", "Show the top 5 products in that city")
    assert "(missing)" in turns[0].answer.text
    assert turns[1].answer.status == CLARIFY


# --- sorting and top N -------------------------------------------------------------------------


def test_sort_highest_and_lowest_match_pandas(shop):
    c, turns = chat(shop, "Show total quantity by category", "Sort it from lowest to highest")
    totals = shop.groupby("Category")["Quantity"].sum()
    assert turns[1].answer.table["sum of Quantity"].tolist() == sorted(totals.tolist())
    again = c.ask(shop, "sort it from highest to lowest").answer
    assert again.table["sum of Quantity"].tolist() == sorted(totals.tolist(), reverse=True)
    reversed_ = c.ask(shop, "reverse the order").answer
    assert reversed_.table["sum of Quantity"].tolist() == sorted(totals.tolist())


def test_sort_by_named_column_and_alphabetically(shop):
    c, _ = chat(shop, "Show total quantity by category")
    by_name = c.ask(shop, "sort it alphabetically").answer
    assert by_name.table["Category"].tolist() == ["Home", "Office", "Paper"]
    by_col = c.ask(shop, "sort by category descending").answer
    assert by_col.table["Category"].tolist() == ["Paper", "Office", "Home"]


def test_sort_unknown_column_and_no_table(shop):
    c, _ = chat(shop, "Show total quantity by category")
    assert c.ask(shop, "sort by colour").answer.status == CLARIFY
    _, turns = chat(shop, "What is the average quantity?", "sort it")
    assert turns[1].answer.status == CLARIFY and "no table" in turns[1].answer.text


def test_top_n_slices_the_previous_result(shop):
    _, turns = chat(shop, "Find the top 3 products by quantity", "Show only the top 2")
    expected = shop.groupby("Product")["Quantity"].sum().nlargest(2)
    assert turns[1].answer.table["Product"].tolist() == list(expected.index)
    assert turns[1].answer.context.n == 2


def test_larger_top_n_answers_the_question_again(shop):
    _, turns = chat(shop, "Find the top 2 products by quantity", "top 4 instead")
    answer = turns[1].answer
    assert len(answer.table) == 4
    assert "Answered again as" in answer.details[0]


def test_bottom_n_after_group(shop):
    _, turns = chat(shop, "Show total quantity by category", "show only the bottom 1")
    assert turns[1].answer.table["Category"].tolist() == ["Home"]


@pytest.mark.parametrize("bad", ["top 0", "top -2", "top 2.5", "top 500"])
def test_invalid_top_n_follow_up(shop, bad):
    _, turns = chat(shop, "Find the top 3 products by quantity", bad)
    assert turns[1].answer.status == ERROR
    assert "whole number between 1 and 100" in turns[1].answer.text


def test_top_n_beyond_ranking_rows_is_explained(shop):
    _, turns = chat(shop, "Which product has the most orders?", "show only the top 20")
    assert "only had" in turns[1].answer.text


# --- chart type ------------------------------------------------------------------------------------


def test_bar_and_pie_charts_of_the_previous_result(shop):
    c, turns = chat(shop, "Show total quantity by category", "Plot it as a pie chart")
    pie = turns[1].answer.figure
    assert pie.data[0].type == "pie"
    assert sum(pie.data[0].values) == shop["Quantity"].sum()
    bar = c.ask(shop, "make it a bar chart").answer.figure
    assert bar.data[0].type == "bar"


def test_line_chart_needs_dates(shop):
    c, turns = chat(shop, "Show total quantity by category", "plot it as a line chart")
    assert turns[1].answer.status == UNSUPPORTED and "needs dates" in turns[1].answer.text
    c.ask(shop, "What is the order trend by month?")
    line = c.ask(shop, "sort it from highest to lowest").answer
    assert line.ok and line.figure.data[0].type == "scatter"  # Plotly line traces are scatter
    assert list(line.figure.data[0].x) == sorted(line.figure.data[0].x)  # still in date order


def test_pie_chart_refused_for_averages_and_negatives(shop):
    _, turns = chat(shop, "Show average quantity by category", "plot it as a pie chart")
    assert turns[1].answer.status == UNSUPPORTED and "parts of a whole" in turns[1].answer.text
    df = shop.assign(Quantity=[3, 1, 4, 2, 5, -10, 2, 1, 7])  # Mumbai's total is -10
    _, turns = chat(df, "Show total quantity by city", "plot it as a pie chart")
    assert turns[1].answer.status == UNSUPPORTED
    assert "negative" in turns[1].answer.text


def test_chart_of_a_single_number_is_explained(shop):
    _, turns = chat(shop, "What is the average quantity?", "plot it as a bar chart")
    assert turns[1].answer.status == CLARIFY  # no table to chart


def test_table_only(shop):
    _, turns = chat(shop, "Show total quantity by category", "show it as a table")
    assert turns[1].answer.figure is None and len(turns[1].answer.table) == 3


# --- regrouping and swapping filters ------------------------------------------------------------


def test_group_filtered_rows_by_another_column(shop):
    _, turns = chat(shop, "Show orders from Pune", "Now group them by category.")
    answer = turns[1].answer
    expected = shop[shop["City"] == "Pune"]["Category"].value_counts()
    assert answer.table.set_index("Category")["rows"].to_dict() == expected.to_dict()
    assert "`City` is Pune" in answer.text


def test_regroup_keeps_the_measure(shop):
    _, turns = chat(shop, "Show total quantity for Office products", "break it down by city")
    office = shop[shop["Category"] == "Office"]
    expected = office.groupby("City")["Quantity"].sum()
    table = turns[1].answer.table.set_index("City")["sum of Quantity"]
    assert table["Pune"] == expected["Pune"]


def test_what_about_replaces_the_filter_on_the_same_column(shop):
    _, turns = chat(shop, "Show orders from Pune", "Now group them by category", "What about Delhi?")
    answer = turns[2].answer
    assert "`City` is Delhi" in answer.text and "Pune" not in answer.text
    expected = shop[shop["City"] == "Delhi"]["Category"].value_counts()
    assert answer.table.set_index("Category")["rows"].to_dict() == expected.to_dict()


def test_what_about_an_unknown_value(shop):
    _, turns = chat(shop, "Show orders from Pune", "What about Kolkata?")
    assert turns[1].answer.status == CLARIFY
    assert "kolkata" in turns[1].answer.text.lower()


def test_follow_up_on_an_empty_result(shop):
    _, turns = chat(shop, "orders from Mumbai paid by cash", "Now group them by category")
    assert "**0 of 9 rows**" in turns[0].answer.text
    assert "No rows match" in turns[1].answer.text


# --- conversation state -------------------------------------------------------------------------------


def test_history_is_bounded(shop):
    c = Conversation("fp", max_turns=5)
    for i in range(12):
        c.ask(shop, f"top {i + 1} products by quantity")
    assert len(c.turns) == 5
    assert [t.question for t in c.turns] == [f"top {i} products by quantity" for i in range(8, 13)]  # newest kept
    assert Conversation().turns.maxlen == MAX_TURNS


def test_dataset_switch_clears_the_context(shop):
    c, _ = chat(shop, "Which city has the most orders?")
    assert c.last is not None
    assert c.reset_if_dataset_changed("other") is True
    assert c.last is None and len(c.turns) == 0
    assert c.reset_if_dataset_changed("other") is False
    answer = c.ask(shop, "Show the top 5 products in that city").answer
    assert answer.status == CLARIFY and "no earlier answer" in answer.text


def test_stale_context_from_another_dataset_never_crashes(shop):
    c, _ = chat(shop, "Show orders from Pune")
    other = pd.DataFrame({"team": ["a", "b"], "points": [1, 2]})
    answer = c.ask(other, "Now group them by team").answer
    assert answer.status in (ERROR, CLARIFY)  # the old filter's column doesn't exist here


def test_failed_follow_up_keeps_the_last_good_context(shop):
    c, _ = chat(shop, "Show total quantity by category")
    c.ask(shop, "plot it as a line chart")  # refused
    assert c.ask(shop, "sort it from lowest to highest").answer.ok


def test_answer_context_is_structured(shop):
    answer = answer_question(shop, "Show total quantity by category for UPI")
    ctx = answer.context
    assert (ctx.intent, ctx.groups, ctx.value, ctx.agg) == ("group comparison", ("Category",), "Quantity", "sum")
    assert ctx.filters[0].description == "`Payment_Method` is UPI"
    assert ctx.core == "total quantity by category"


# --- suggestions ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first",
    ["Which city has the most orders?", "Show total quantity by category", "Show orders from Pune",
     "Find the top 3 products by quantity", "What is the order trend by month?"],
)  # fmt: skip
def test_suggested_follow_ups_all_work(shop, first):
    c, _ = chat(shop, first)
    suggestions = suggest_followups(c.last, shop)
    assert suggestions
    for question in suggestions:
        trial = Conversation("fp")
        trial.ask(shop, first)
        answer = trial.ask(shop, question).answer
        assert answer.ok, f"{first!r} -> {question!r}: {answer.text}"


def test_no_suggestions_without_context(shop):
    assert suggest_followups(None, shop) == []


# --- safety ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["sort it; DROP TABLE sales", "show the products in that city; delete everything",
     "what about __import__('os')", "plot it as a bar chart -- comment", "update it to 0"],
)  # fmt: skip
def test_unsafe_follow_ups_are_refused(shop, question):
    original = shop.copy(deep=True)
    _, turns = chat(shop, "Which city has the most orders?", question)
    assert turns[1].answer.status == UNSUPPORTED
    pd.testing.assert_frame_equal(shop, original)


def test_follow_ups_never_modify_the_dataset(shop):
    original = shop.copy(deep=True)
    chat(shop, "Show orders from Pune", "Now group them by category", "sort it", "Plot it as a pie chart",
         "What about Delhi?", "Which product has the most orders?", "Show the trend for that product")  # fmt: skip
    pd.testing.assert_frame_equal(shop, original)


def test_follow_ups_never_call_eval_or_exec(shop, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eval/exec must never be called")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    _, turns = chat(shop, "Which city has the most orders?", "Show the top 5 products in that city",
                    "sort it", "plot it as a pie chart", "what about Delhi?")  # fmt: skip
    assert all(t.answer.status in (ANSWER, CLARIFY) for t in turns)


def test_conversation_source_has_no_dynamic_code_or_database_access():
    tree = ast.parse((ROOT / "askyourdata" / "conversation.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__", "open"}
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name and ("sql" in name or "database" in name) for name in imported)


def test_respond_never_raises(shop, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    c, _ = chat(shop, "Show total quantity by category")
    monkeypatch.setattr(conv.de, "sort_rows", broken)
    answer = c.ask(shop, "sort it").answer
    assert answer.status == ERROR and "secret" not in answer.text
