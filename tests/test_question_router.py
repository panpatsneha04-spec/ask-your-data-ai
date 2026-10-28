"""Tests for the natural-language question router (no LLM, no network)."""

import ast
import builtins
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from askyourdata import question_router as qr
from askyourdata.question_router import (
    ANSWER,
    CLARIFY,
    ERROR,
    NO_DATA,
    UNSUPPORTED,
    ColumnMatcher,
    answer_question,
    suggest_questions,
    words,
)

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_sales.csv"
ROUTER_SOURCE = Path(qr.__file__)


@pytest.fixture
def shop():
    return pd.DataFrame(
        {
            "Order Date": [
                "2026-01-05", "2026-01-20", "2026-02-03", "2026-02-14",
                "2026-03-01", "2026-03-15", "2026-03-20", None,
            ],
            "City": ["Paris", "Rome", "Paris", "Oslo", "Rome", "Paris", "Oslo", None],
            "Product": ["Pen", "Ink", "Pad", "Pen", "Pad", "Pen", "Ink", "Pad"],
            "Category": ["Office", "Office", "Paper", "Office", "Paper", "Office", "Office", "Paper"],
            "Quantity": [3, 1, 4, 2, 5, 1, None, 2],
            "Unit Price": [2.5, 9.0, 4.0, 2.5, 4.0, 2.5, 9.0, 4.0],
            "Revenue": [7.5, 9.0, 16.0, 5.0, 20.0, 2.5, None, 8.0],
            "payment_method": ["Card", "Cash", "Card", "Card", None, "Cash", "Card", "Card"],
        }
    )  # fmt: skip


@pytest.fixture
def sample():
    return pd.read_csv(SAMPLE_CSV)


def ask(df, question):
    return answer_question(df, question)


# --- column matching -------------------------------------------------------------


def test_words_normalise_case_separators_and_plurals():
    assert words("Unit_Price") == ("unit", "price")
    assert words("unit prices") == ("unit", "price")
    assert words("paymentMethod") == ("payment", "method")
    assert words("Categories") == ("category",)


@pytest.mark.parametrize(
    ("phrase", "column", "score"),
    [
        ("unit price", "Unit Price", 100),
        ("UNIT_PRICE", "Unit Price", 100),
        ("unitprice", "Unit Price", 90),
        ("payment methods", "payment_method", 100),
        ("location", "City", 80),
        ("item", "Product", 80),
        ("payment", "payment_method", 80),
        ("date", "Order Date", 60),
        ("price", "Unit Price", 60),  # only one *price column, so no ambiguity
    ],
)
def test_column_matching(shop, phrase, column, score):
    found = ColumnMatcher(shop).resolve(words(phrase), "column")
    assert (found.column, found.score) == (column, score)


def test_quantity_synonym_matches_units_not_unit_price(sample):
    assert ColumnMatcher(sample).resolve(words("quantity"), "value").column == "units"


def test_order_never_matches_order_id_or_order_date(sample):
    assert ColumnMatcher(sample).candidates(words("orders")) == []


def test_ambiguous_columns_ask_for_clarification():
    df = pd.DataFrame({"unit_price": [1.0, 2.0], "list_price": [3.0, 4.0]})
    answer = ask(df, "What is the average price?")
    assert answer.status == CLARIFY
    assert "could mean more than one column" in answer.text
    assert set(answer.options) == {"unit_price", "list_price"}


def test_missing_column_is_explained_not_guessed(sample):
    answer = ask(sample, "Which city has the most orders?")
    assert answer.status == CLARIFY
    assert "couldn't find a column matching 'city'" in answer.text
    assert "region" in answer.text  # lists what does exist, but does not use it


def test_missing_revenue_column_gets_a_hint(sample):
    answer = ask(sample, "Show total revenue by region")
    assert answer.status == CLARIFY
    assert "not a column here" in answer.text


def test_wrong_column_type_is_explained(shop):
    answer = ask(shop, "What is the average category?")
    assert answer.status == CLARIFY
    assert "categorical column" in answer.text and "numeric" in answer.text


def test_synonym_match_is_reported_in_details(sample):
    answer = ask(sample, "Show total quantity by category")
    assert answer.ok
    assert any("'quantity' was matched to column `units`" in d for d in answer.details)


# --- intent detection --------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("How many rows are in this dataset?", "overview"),
        ("how many records do we have", "overview"),
        ("What columns are there?", "overview"),
        ("Give me an overview of the data", "overview"),
        ("Please describe the dataset", "overview"),
        ("Which city has the most orders?", "ranking"),
        ("What is the most common payment method?", "ranking"),
        ("Which city has the lowest average revenue?", "ranking"),
        ("Show total quantity by category.", "group comparison"),
        ("average unit price per city and category", "group comparison"),
        ("number of orders by city", "group comparison"),
        ("count by category", "group comparison"),
        ("What is the average unit price?", "aggregate"),
        ("maximum quantity", "aggregate"),
        ("What's the highest unit price?", "aggregate"),
        ("Find the top 5 products by quantity.", "top N"),
        ("bottom 2 products by revenue", "top N"),
        ("top five cities", "top N"),
        ("top 3 unit price", "top N"),
        ("Show total revenue by city.", "group comparison"),
        ("What is the order trend by month?", "trend"),
        ("monthly revenue trend", "trend"),
        ("total quantity per week", "trend"),
        ("Show the distribution of payment methods.", "distribution"),
        ("histogram of unit price", "distribution"),
        ("summary of quantity", "distribution"),
        ("how many different cities are there", "count"),
        ("Are there any missing values?", "missing values"),
        ("missing values in payment method", "missing values"),
        ("Show the correlation between numeric columns.", "correlation"),
        ("is quantity correlated with revenue", "correlation"),
    ],
)
def test_intent_detection(shop, question, intent):
    answer = ask(shop, question)
    assert answer.status == ANSWER, answer.text
    assert answer.intent == intent


# --- answers match pandas ------------------------------------------------------------


def test_row_count(shop):
    assert "**8 rows**" in ask(shop, "How many rows are in this dataset?").text


def test_which_city_has_most_orders(shop):
    answer = ask(shop, "Which city has the most orders?")
    assert answer.text.startswith("**Paris** has the most rows: **3**")
    assert answer.table.iloc[0]["count"] == shop["City"].value_counts().max()


def test_group_total_matches_pandas(shop):
    answer = ask(shop, "Show total quantity by category.")
    table = answer.table.set_index("Category")["sum of Quantity"]
    expected = shop.groupby("Category")["Quantity"].sum()
    assert table.to_dict() == expected.to_dict()
    assert "Highest: **Paper** with **11**" in answer.text
    assert isinstance(answer.figure, go.Figure)


def test_average_ignores_missing_values(shop):
    answer = ask(shop, "average quantity")
    assert f"**{shop['Quantity'].mean():.2f}**" in answer.text
    assert "1 missing values ignored" in answer.text


def test_average_unit_price(shop):
    answer = ask(shop, "What is the average unit price?")
    assert "**4.69**" in answer.text  # 37.5 / 8
    assert answer.figure is None
    assert any(d.startswith("Calculation:") for d in answer.details)


def test_top_n_matches_pandas(shop):
    answer = ask(shop, "Find the top 2 products by quantity")
    expected = shop.groupby("Product")["Quantity"].sum().nlargest(2)
    assert answer.table["Product"].tolist() == expected.index.tolist()
    assert answer.table["sum of Quantity"].tolist() == expected.tolist()


def test_top_n_reports_fewer_values_than_requested(shop):
    answer = ask(shop, "top 10 products by quantity")
    assert "Only 3 different values exist" in answer.text


def test_top_without_number_uses_default(shop):
    answer = ask(shop, "top products by revenue")
    assert "no number was given" in answer.text


def test_bottom_n_sorts_smallest_first(shop):
    answer = ask(shop, "bottom 1 products by quantity")
    assert answer.table["Product"].tolist() == ["Ink"]


def test_ranking_with_average_and_lowest(shop):
    answer = ask(shop, "Which city has the lowest average unit price?")
    means = shop.groupby("City")["Unit Price"].mean()
    assert answer.table.iloc[0]["City"] == means.idxmin()


def test_ranking_tie_is_reported():
    df = pd.DataFrame({"team": ["a", "b", "c"], "points": [5, 5, 1]})
    answer = ask(df, "Which team has the highest points?")
    assert "tie" in answer.text and "**a**" in answer.text and "**b**" in answer.text


def test_trend_counts_rows_per_month(shop):
    answer = ask(shop, "What is the order trend by month?")
    assert answer.table["rows"].tolist() == [2, 2, 3]
    assert "1 row(s) with a missing or unreadable date were left out" in answer.text


def test_trend_with_value_and_frequency(shop):
    answer = ask(shop, "average revenue per month")
    assert answer.intent == "trend"
    assert answer.table.columns[-1] == "mean of Revenue"


def test_trend_without_date_column_is_explained():
    df = pd.DataFrame({"city": ["a", "b"], "sales": [1, 2]})
    answer = ask(df, "What is the trend by month?")
    assert answer.status == UNSUPPORTED
    assert "no date column" in answer.text


def test_trend_with_several_date_columns_asks_which():
    df = pd.DataFrame(
        {"order_date": ["2026-01-01", "2026-02-01"], "ship_date": ["2026-01-03", "2026-02-04"]}
    )
    answer = ask(df, "order trend by month")
    assert answer.status == CLARIFY
    assert set(answer.options) == {"order_date", "ship_date"}
    named = ask(df, "trend of rows by month using ship date")
    assert named.ok and "`ship_date` (date)" in named.details[0]


def test_distribution_of_categories(shop):
    answer = ask(shop, "Show the distribution of payment methods.")
    assert "**3 different values**" in answer.text  # Card, Cash, (missing)
    assert "**Card**" in answer.text


def test_distribution_of_numbers(shop):
    answer = ask(shop, "distribution of unit price")
    assert "ranges from **2.50** to **9**" in answer.text


def test_count_distinct(shop):
    answer = ask(shop, "how many different cities are there")
    assert "**3 different values**" in answer.text
    assert "1 missing values not counted" in answer.text


def test_missing_values_total(shop):
    answer = ask(shop, "Are there any missing values?")
    assert f"**{int(shop.isna().sum().sum())} missing cells**" in answer.text


def test_missing_values_for_one_column(shop):
    answer = ask(shop, "how many missing values in quantity")
    assert "`Quantity` has **1 missing values**" in answer.text


def test_no_missing_values():
    answer = ask(pd.DataFrame({"a": [1, 2]}), "any missing values?")
    assert answer.ok and "No missing values" in answer.text


def test_correlation_pair_matches_pandas(shop):
    answer = ask(shop, "correlation between quantity and revenue")
    expected = shop["Quantity"].corr(shop["Revenue"])
    assert f"**{expected:.2f}**" in answer.text


def test_correlation_overview(shop):
    answer = ask(shop, "Show the correlation between numeric columns.")
    assert answer.ok and isinstance(answer.figure, go.Figure)


# --- invalid input and incompatible types ---------------------------------------------


@pytest.mark.parametrize("question", ["top 0 products", "top -3 products", "top 2.5 products",
                                      "top 500 products by quantity"])  # fmt: skip
def test_invalid_top_n(shop, question):
    answer = ask(shop, question)
    assert answer.status == ERROR
    assert "whole number between 1 and 100" in answer.text


@pytest.mark.parametrize(
    "question",
    [
        "total city by category",  # text column cannot be summed
        "correlation between city and revenue",
        "top 3 cities by product",
        "average orders by city",  # rows can only be counted
    ],
)
def test_incompatible_types_ask_for_clarification(shop, question):
    assert ask(shop, question).status == CLARIFY


def test_grouping_by_continuous_numbers_is_refused():
    df = pd.DataFrame({"amount": [float(i) for i in range(30)], "n": [1] * 30})
    answer = ask(df, "total n by amount")
    assert answer.status == ERROR
    assert "too many to group by" in answer.text


# --- empty data, no data ---------------------------------------------------------------


def test_no_dataset_selected():
    assert ask(None, "How many rows?").status == NO_DATA


def test_dataset_without_columns():
    assert ask(pd.DataFrame(), "How many rows?").status == NO_DATA


def test_empty_dataset(shop):
    empty = shop.iloc[0:0]
    assert "**0 rows**" in ask(empty, "how many rows are there").text
    answer = ask(empty, "What is the average unit price?")
    assert answer.status == ERROR and "no rows" in answer.text


def test_all_missing_column():
    df = pd.DataFrame({"x": [None, None]}, dtype=float)
    answer = ask(df, "average x")
    assert answer.status == ERROR and "all missing" in answer.text


# --- unsupported and unsafe requests -----------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["What is the meaning of life?", "tell me a joke", "", "   ", 42, None, "x" * 301,
     "predict next month's revenue", "why did revenue go down"],
)  # fmt: skip
def test_unsupported_questions(shop, question):
    answer = ask(shop, question)
    assert answer.status == UNSUPPORTED
    assert answer.table is None and answer.figure is None


def test_unsupported_answers_offer_working_examples(shop):
    answer = ask(shop, "What is the meaning of life?")
    assert answer.suggestions and "How many rows are in this dataset?" in answer.suggestions


@pytest.mark.parametrize(
    "question",
    [
        "DROP TABLE sales",
        "SELECT * FROM staff_salaries",
        "select revenue from sales where 1=1",
        "__import__('os').system('dir')",
        "import os",
        "eval('1+1')",
        "exec('x = 1')",
        "delete all rows where city is Paris",
        "update revenue to 0",
        "replace missing values with 0",
        "average revenue; DROP TABLE x",
        "total revenue -- comment",
        "<script>alert(1)</script>",
        "lambda x: x",
    ],
)
def test_code_sql_and_write_requests_are_refused(shop, question):
    original = shop.copy(deep=True)
    answer = ask(shop, question)
    assert answer.status == UNSUPPORTED
    pd.testing.assert_frame_equal(shop, original)


def test_router_never_calls_eval_or_exec(shop, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eval/exec must never be called")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    for question in suggest_questions(shop) + ["eval('1')", "exec('1')", "DROP TABLE x"]:
        assert ask(shop, question).status in (ANSWER, UNSUPPORTED)


def test_router_source_has_no_dynamic_code_or_database_access():
    tree = ast.parse(ROUTER_SOURCE.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls & {"eval", "exec", "compile", "__import__", "getattr", "open"}
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name and ("sql" in name or "database" in name) for name in imported)


def test_unexpected_internal_errors_are_hidden(shop, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(qr.de, "profile_dataset", broken)
    answer = ask(shop, "How many rows are in this dataset?")
    assert answer.status == ERROR
    assert "secret internal detail" not in answer.text


# --- general guarantees ------------------------------------------------------------------


QUESTIONS = [
    "How many rows are in this dataset?", "Which city has the most orders?",
    "Show total quantity by category.", "What is the average unit price?",
    "Find the top 5 products by quantity.", "Show total revenue by city.",
    "What is the order trend by month?", "Show the distribution of payment methods.",
    "Are there any missing values?", "Show the correlation between numeric columns.",
]  # fmt: skip


@pytest.mark.parametrize("question", QUESTIONS)
def test_brief_examples_answer_and_explain(shop, question):
    original = shop.copy(deep=True)
    answer = ask(shop, question)
    assert answer.ok, answer.text
    assert any(d.startswith("Calculation:") for d in answer.details)
    pd.testing.assert_frame_equal(shop, original)  # the dataset is never modified


@pytest.mark.parametrize("dataset", ["shop", "sample"])
def test_every_suggested_question_works(dataset, request):
    df = request.getfixturevalue(dataset)
    suggestions = suggest_questions(df)
    assert len(suggestions) >= 5
    for question in suggestions:
        answer = ask(df, question)
        assert answer.ok, f"{question!r} -> {answer.text}"


def test_suggestions_use_real_columns(sample):
    text = " ".join(suggest_questions(sample))
    assert "city" not in text and "revenue" not in text
    assert "average units" in text
    assert "order id" not in text  # ID columns are not suggested for maths
