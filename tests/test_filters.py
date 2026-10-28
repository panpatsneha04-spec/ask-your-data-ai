"""Tests for natural-language filters and filtering combined with analysis."""

import pandas as pd
import pytest

from askyourdata.nl_filters import extract_dates
from askyourdata.question_router import ANSWER, CLARIFY, ERROR, UNSUPPORTED, answer_question
from askyourdata.text_matching import ClarificationNeeded, UnsupportedRequest


@pytest.fixture
def shop():
    return pd.DataFrame(
        {
            "Order_ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Order_Date": [
                "2026-01-05", "2026-01-28", "2026-02-03", "2026-02-20",
                "2026-03-02", "2026-03-15", "2026-03-31", None,
            ],
            "City": ["Pune", "Mumbai", "Pune", "Delhi", "Pune", "Mumbai", None, "Delhi"],
            "Product": ["Phone", "Shirt", "Lamp", "Phone", "Shirt", "Lamp", "Phone", "Mug"],
            "Category": ["Electronics", "Clothing", "Home", "Electronics", "Clothing", "Home",
                         "Electronics", "Home"],
            "Quantity": [1, 4, 2, 5, 3, None, 2, 6],
            "Unit_Price": [500.0, 20.0, 40.0, 500.0, 20.0, 40.0, 500.0, 5.0],
            "Discount_Pct": [10, 0, 5, 20, 0, 5, 10, 0],
            "Payment_Method": ["UPI", "Card", "UPI", "Cash on Delivery", "Card", "UPI", "Card", "UPI"],
        }
    )  # fmt: skip


def ask(df, question):
    return answer_question(df, question)


# --- text value filters --------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_rows", "description"),
    [
        ("Show orders from Pune.", 3, "`City` is Pune"),
        ("Show products in the Electronics category.", 3, "`Category` is Electronics"),
        ("Show orders paid by UPI.", 4, "`Payment_Method` is UPI"),
        ("orders paid by cash on delivery", 1, "`Payment_Method` is Cash on Delivery"),
        ("show rows where city is mumbai", 2, "`City` is Mumbai"),
        ("orders from Pune or Mumbai", 5, "`City` is Pune or Mumbai"),
        ("electronics orders from pune", 1, "`Category` is Electronics"),
    ],
)
def test_text_value_filters(shop, question, expected_rows, description):
    answer = ask(shop, question)
    assert answer.status == ANSWER, answer.text
    assert answer.intent == "filter"
    assert len(answer.table) == expected_rows
    assert description in answer.text
    assert f"**{expected_rows} of 8 rows**" in answer.text


def test_filter_matching_ignores_case_and_returns_real_rows(shop):
    answer = ask(shop, "show orders from PUNE")
    assert answer.table["City"].tolist() == ["Pune"] * 3
    assert answer.table["row"].tolist() == [0, 2, 4]  # original row numbers are kept


def test_value_in_two_columns_asks_which():
    df = pd.DataFrame({"home_city": ["Pune", "Delhi"], "work_city": ["Delhi", "Pune"], "n": [1, 2]})
    answer = ask(df, "show rows from Pune")
    assert answer.status == CLARIFY
    assert set(answer.options) == {"home_city", "work_city"}
    named = ask(df, "show rows where work city is Pune")
    assert named.ok and len(named.table) == 1


def test_unknown_value_is_not_silently_ignored(shop):
    answer = ask(shop, "Show orders from Kolkata")
    assert answer.status == CLARIFY
    assert "couldn't find 'kolkata'" in answer.text


def test_typo_gets_a_suggestion(shop):
    answer = ask(shop, "orders from Pnue")
    assert answer.status == CLARIFY and "Did you mean Pune?" in answer.text


# --- number filters ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Show orders with quantity greater than 3.", 3),
        ("orders with quantity > 3", 3),
        ("orders where quantity is at least 3", 4),
        ("orders with quantity less than 2", 1),
        ("orders with quantity between 2 and 4", 4),
        ("orders above quantity 2", 4),
        ("orders with unit price over 100", 3),
        ("orders with unit price under 1,000", 8),
        ("orders with quantity equal to 2", 2),
    ],
)
def test_number_filters(shop, question, expected):
    answer = ask(shop, question)
    assert answer.ok, answer.text
    assert len(answer.table) == expected


def test_missing_numbers_never_match_a_number_filter(shop):
    answer = ask(shop, "orders with quantity less than 100")
    assert len(answer.table) == 7  # the row with a missing quantity is excluded


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ("orders with quantity greater than lots", "give a number after 'greater than'"),
        ("orders above 2", "Which column should be greater than 2"),
        ("orders with city greater than 3", "not a numeric column"),
    ],
)
def test_invalid_number_filters_ask_for_clarification(shop, question, message):
    answer = ask(shop, question)
    assert answer.status == CLARIFY
    assert message in answer.text


# --- date filters ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Show orders between 2026-02-01 and 2026-02-28.", 2),
        ("orders from 2026-01-01 to 2026-01-31", 2),
        ("orders after 2026-03-02", 2),
        ("orders before 2026-02-01", 2),
        ("orders since 2026-03-02", 3),
        ("orders until 2026-01-28", 2),
        ("orders in February 2026", 2),
        ("orders in march 2026", 3),
        ("orders on 31 March 2026", 1),
        ("orders on March 31, 2026", 1),
        ("orders on 31/03/2026", 1),
        ("orders in 2026", 7),
    ],
)
def test_date_filters(shop, question, expected):
    answer = ask(shop, question)
    assert answer.ok, answer.text
    assert len(answer.table) == expected


def test_date_parsing():
    _, spans = extract_dates("between 2026-02-01 and march 2026")
    assert spans[0].start == pd.Timestamp("2026-02-01")
    assert spans[1].start == pd.Timestamp("2026-03-01")
    assert spans[1].end.date() == pd.Timestamp("2026-03-31").date()


@pytest.mark.parametrize(
    ("question", "status", "message"),
    [
        ("orders on 03/04/2026", CLARIFY, "could mean 3 April or March 4"),
        ("orders on 2026-13-45", CLARIFY, "not a valid date"),
        ("Show orders between two dates.", CLARIFY, "Which two dates?"),
        ("orders between 2026-03-01 and 2026-01-01", CLARIFY, "ends before it starts"),
        ("total quantity last month", UNSUPPORTED, "Relative dates"),
    ],
)
def test_unclear_dates(shop, question, status, message):
    answer = ask(shop, question)
    assert answer.status == status
    assert message in answer.text


def test_date_filter_without_date_column():
    df = pd.DataFrame({"city": ["Pune"], "n": [1]})
    answer = ask(df, "rows in March 2026")
    assert answer.status == UNSUPPORTED and "no date column" in answer.text


def test_date_filter_with_two_date_columns_asks_which():
    df = pd.DataFrame({"order_date": ["2026-01-01"], "ship_date": ["2026-01-03"]})
    answer = ask(df, "rows in January 2026")
    assert answer.status == CLARIFY and set(answer.options) == {"order_date", "ship_date"}
    assert ask(df, "rows with ship date in January 2026").ok


def test_extract_dates_errors_directly():
    with pytest.raises(ClarificationNeeded):
        extract_dates("on 05/06/2026")
    with pytest.raises(UnsupportedRequest):
        extract_dates("sales this year")


# --- filtering combined with analysis ---------------------------------------------------------


def test_total_after_filter(shop):
    answer = ask(shop, "Show total quantity for Electronics products.")
    expected = shop.loc[shop["Category"] == "Electronics", "Quantity"].sum()
    assert answer.intent == "aggregate"
    assert f"**{expected:.0f}**" in answer.text
    assert answer.text.startswith("**Filters:** `Category` is Electronics")
    assert answer.details[0].startswith("Filters applied before the calculation")


def test_top_n_after_filter(shop):
    answer = ask(shop, "Find the top 5 products in Pune.")
    assert answer.intent == "top N"
    pune = shop[shop["City"] == "Pune"]
    assert answer.table["count"].sum() == len(pune)


def test_group_after_number_filter(shop):
    answer = ask(shop, "Compare payment methods for orders above quantity 2.")
    expected = shop[shop["Quantity"] > 2]["Payment_Method"].value_counts()
    table = answer.table.set_index("Payment_Method")["rows"]
    assert table.to_dict() == expected.to_dict()


def test_metric_after_filter(shop):
    answer = ask(shop, "net revenue by city for Electronics")
    table = answer.table.set_index("City")["sum of Net revenue (calculated)"]
    assert table["Pune"] == pytest.approx(450.0)  # 1 x 500 x 0.9
    assert table["Delhi"] == pytest.approx(2000.0)  # 5 x 500 x 0.8


def test_trend_after_filter(shop):
    answer = ask(shop, "order trend by month for UPI")
    assert answer.intent == "trend"
    assert answer.table["rows"].sum() == 3  # the UPI row with no date is left out


def test_several_filters_together(shop):
    answer = ask(shop, "total quantity for orders from Pune paid by UPI in 2026")
    rows = shop[(shop["City"] == "Pune") & (shop["Payment_Method"] == "UPI")]
    assert f"**{rows['Quantity'].sum():.0f}**" in answer.text
    assert answer.text.count(";") == 2  # three filters listed


def test_empty_filter_result(shop):
    answer = ask(shop, "total quantity for orders from Pune paid by cash on delivery")
    assert answer.status == ANSWER
    assert "**0 of 8 rows**" in answer.text and "No rows match" in answer.text
    assert answer.table is not None and answer.table.empty


def test_filtered_overview_wording(shop):
    answer = ask(shop, "how many orders are from Pune")
    assert "The filtered data has **3 rows**" in answer.text


def test_unsupported_rest_of_question_is_reported(shop):
    answer = ask(shop, "orders from Pune sorted by colour")
    assert answer.status in (CLARIFY, UNSUPPORTED)


# --- safety and regressions ------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "orders from Pune; DROP TABLE sales",
        "orders where city = 'Pune' OR 1=1 --",
        "orders from __import__('os')",
        "delete orders from Pune",
        "select * from orders where city is Pune",
    ],
)
def test_filter_questions_cannot_run_code_or_sql(shop, question):
    original = shop.copy(deep=True)
    assert ask(shop, question).status == UNSUPPORTED
    pd.testing.assert_frame_equal(shop, original)


def test_filters_never_modify_the_dataset(shop):
    original = shop.copy(deep=True)
    for question in ["Show orders from Pune.", "net revenue by city for Electronics",
                     "orders with quantity > 3", "orders in March 2026"]:  # fmt: skip
        ask(shop, question)
    pd.testing.assert_frame_equal(shop, original)


@pytest.mark.parametrize(
    "question",
    [
        "Which city has the most orders?",
        "Show total quantity by category",
        "What is the order trend by month?",
        "is quantity correlated with unit price",
        "Show the distribution of payment methods",
        "missing values in city",
    ],
)
def test_plain_questions_get_no_filters(shop, question):
    answer = ask(shop, question)
    assert answer.ok, answer.text
    assert "Filters" not in answer.text
