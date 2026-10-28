"""Tests for calculated metrics (gross amount, net revenue) and their use in questions."""

import pandas as pd
import pytest

from askyourdata import metrics
from askyourdata.metrics import (
    DISCOUNT_FRACTION,
    DISCOUNT_PERCENT,
    GENERIC_REVENUE,
    GROSS_AMOUNT,
    NET_REVENUE,
    DiscountScaleUnclear,
    MetricUnavailable,
    compute_metric,
    match_metric,
)
from askyourdata.question_router import ANSWER, CLARIFY, ERROR, answer_question
from askyourdata.text_matching import ClarificationNeeded, words


def orders(discount=(10, 0, 25, None), **extra):
    data = {
        "City": ["Pune", "Mumbai", "Pune", "Delhi"],
        "Quantity": [2, 1, 4, 3],
        "Unit_Price": [100.0, 50.0, 10.0, 20.0],
        "Discount_Pct": list(discount),
    }
    data.update(extra)
    return pd.DataFrame(data)


# --- phrase matching --------------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "key"),
    [
        ("net revenue", NET_REVENUE),
        ("net sales", NET_REVENUE),
        ("revenue after discount", NET_REVENUE),
        ("gross amount", GROSS_AMOUNT),
        ("gross revenue", GROSS_AMOUNT),
        ("revenue", GENERIC_REVENUE),
        ("sales", GENERIC_REVENUE),
        ("unit price", None),
    ],
)
def test_match_metric(phrase, key):
    assert match_metric(words(phrase)) == key


# --- formulas -----------------------------------------------------------------------


def test_gross_amount_formula():
    result = compute_metric(orders(), GROSS_AMOUNT)
    assert result.values.tolist() == [200.0, 50.0, 40.0, 60.0]
    assert result.formula == "Gross amount (calculated) = `Quantity` × `Unit_Price`"


def test_net_revenue_with_whole_number_percentages():
    result = compute_metric(orders(), NET_REVENUE)
    assert result.values.tolist()[:3] == pytest.approx([180.0, 50.0, 30.0])  # 10%, 0%, 25%
    assert pd.isna(result.values.iloc[3])  # missing discount -> unknown, not 0% and not 0
    assert "/ 100" in result.formula
    assert "whole-number percentages (10 = 10%)" in result.notes[0]


def test_net_revenue_with_percent_signs():
    result = compute_metric(orders(discount=("10%", "0%", "25%", "5%")), NET_REVENUE)
    assert result.values.tolist() == pytest.approx([180.0, 50.0, 30.0, 57.0])
    assert "written with '%'" in result.notes[0]


def test_fraction_discounts_are_ambiguous_without_a_hint():
    df = orders(discount=(0.1, 0.0, 0.25, 0.05))
    with pytest.raises(DiscountScaleUnclear, match="fraction .* or a percentage"):
        compute_metric(df, NET_REVENUE)


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(DISCOUNT_FRACTION, [180.0, 50.0, 30.0, 57.0]), (DISCOUNT_PERCENT, [199.8, 50.0, 39.9, 59.97])],
)
def test_explicit_discount_scale(scale, expected):
    df = orders(discount=(0.1, 0.0, 0.25, 0.05))
    result = compute_metric(df, NET_REVENUE, discount_scale=scale)
    assert result.values.tolist() == pytest.approx(expected)


def test_fraction_hint_rejected_for_values_above_one():
    with pytest.raises(MetricUnavailable, match="can't be fractions"):
        compute_metric(orders(), NET_REVENUE, discount_scale=DISCOUNT_FRACTION)


def test_all_zero_discounts():
    result = compute_metric(orders(discount=(0, 0, 0, 0)), NET_REVENUE)
    assert result.values.tolist() == [200.0, 50.0, 40.0, 60.0]
    assert "no discount was applied" in result.notes[0]


@pytest.mark.parametrize("bad", [(10, -5, 0, 0), (10, 150, 0, 0)])
def test_invalid_discount_range(bad):
    with pytest.raises(MetricUnavailable, match="between 0 and 100"):
        compute_metric(orders(discount=bad), NET_REVENUE)


def test_invalid_numbers_become_missing_and_are_reported():
    df = orders()
    df["Quantity"] = ["2", "1", "four", "3"]
    result = compute_metric(df, GROSS_AMOUNT)
    assert pd.isna(result.values.iloc[2])
    assert result.values.iloc[0] == 200.0
    assert any("1 value(s) in `Quantity` were not numbers" in n for n in result.notes)


def test_mostly_text_column_is_refused():
    df = orders()
    df["Quantity"] = ["a", "b", "c", "4"]
    with pytest.raises(MetricUnavailable, match="does not contain numbers"):
        compute_metric(df, GROSS_AMOUNT)


def test_negative_values_are_reported():
    df = orders()
    df.loc[1, "Quantity"] = -1  # a return
    result = compute_metric(df, GROSS_AMOUNT)
    assert any("negative `Quantity`" in n for n in result.notes)


def test_synonym_columns_are_found():
    df = pd.DataFrame({"qty": [2, 3], "price": [5.0, 1.0], "discount_rate": [10, 20]})
    result = compute_metric(df, NET_REVENUE)
    assert result.columns == {"quantity": "qty", "unit price": "price", "discount": "discount_rate"}
    assert result.values.tolist() == pytest.approx([9.0, 2.4])


def test_dataset_is_not_modified():
    df = orders()
    original = df.copy(deep=True)
    compute_metric(df, NET_REVENUE)
    pd.testing.assert_frame_equal(df, original)


# --- missing and ambiguous inputs ----------------------------------------------------


def test_missing_discount_column_is_explained():
    df = orders().drop(columns="Discount_Pct")
    with pytest.raises(MetricUnavailable, match="no discount column was found"):
        compute_metric(df, NET_REVENUE)
    assert metrics.missing_inputs(df, NET_REVENUE) == ["discount"]
    assert metrics.missing_inputs(df, GROSS_AMOUNT) == []


def test_missing_quantity_and_price_are_explained():
    df = pd.DataFrame({"City": ["a"], "Discount_Pct": [5]})
    with pytest.raises(MetricUnavailable, match="no quantity or unit price column"):
        compute_metric(df, NET_REVENUE)


def test_two_discount_columns_ask_which():
    df = orders(Discount_Amount=[1, 2, 3, 4])
    with pytest.raises(ClarificationNeeded, match="several discount columns"):
        compute_metric(df, NET_REVENUE)


def test_two_price_columns_ask_which():
    df = pd.DataFrame({"qty": [1], "list_price": [5.0], "sale_price": [4.0]})
    with pytest.raises(ClarificationNeeded, match="more than one column"):
        compute_metric(df, GROSS_AMOUNT)


# --- metrics inside questions ----------------------------------------------------------


def test_calculate_net_revenue_question():
    answer = answer_question(orders(), "Calculate net revenue.")
    assert answer.status == ANSWER
    assert "**260**" in answer.text  # 180 + 50 + 30; the unknown row is left out
    assert any(d.startswith("Formula: Net revenue") for d in answer.details)
    assert any("not counted as 0" in d for d in answer.details)
    assert list(answer.table.columns) == ["Quantity", "Unit_Price", "Discount_Pct", "Net revenue (calculated)"]


def test_net_revenue_by_city_matches_pandas():
    df = orders()
    answer = answer_question(df, "Show net revenue by city.")
    table = answer.table.set_index("City")["sum of Net revenue (calculated)"]
    assert table["Pune"] == pytest.approx(210.0)
    assert table["Mumbai"] == pytest.approx(50.0)


def test_which_product_has_highest_net_revenue():
    df = orders(Product=["Pen", "Ink", "Pen", "Pad"])
    answer = answer_question(df, "Which product has the highest net revenue?")
    assert answer.text.startswith("**Pen** has the highest total")


def test_generic_revenue_asks_gross_or_net_with_rephrasings():
    answer = answer_question(orders(Category=["A", "B", "A", "B"]), "Compare revenue across categories.")
    assert answer.status == CLARIFY
    assert "not a column here" in answer.text
    assert answer.rephrasings == (
        "Compare net revenue across categories.",
        "Compare gross amount across categories.",
    )
    assert answer_question(orders(Category=["A", "B", "A", "B"]), answer.rephrasings[0]).ok


def test_generic_revenue_without_discount_only_offers_gross():
    df = orders().drop(columns="Discount_Pct")
    answer = answer_question(df, "total revenue by city")
    assert answer.status == CLARIFY and answer.options == ("gross amount",)


def test_existing_revenue_column_is_used_and_never_overwritten():
    df = orders(Revenue=[1.0, 2.0, 3.0, 4.0])
    original = df.copy(deep=True)
    answer = answer_question(df, "total revenue")
    assert "**10**" in answer.text  # the existing column, not a calculation
    net = answer_question(df, "total net revenue")
    assert "**260**" in net.text
    assert any("`Revenue` were not used or changed" in d for d in net.details)
    pd.testing.assert_frame_equal(df, original)


def test_existing_net_revenue_column_wins():
    df = orders(Net_Revenue=[5.0, 5.0, 5.0, 5.0])
    answer = answer_question(df, "total net revenue")
    assert "**20**" in answer.text
    assert any("nothing was recalculated" in d for d in answer.details)


def test_unclear_discount_scale_offers_both_readings():
    df = orders(discount=(0.1, 0.0, 0.25, 0.05))
    answer = answer_question(df, "Calculate net revenue?")
    assert answer.status == CLARIFY
    assert answer.rephrasings == (
        "Calculate net revenue with discount as fraction",
        "Calculate net revenue with discount as percent",
    )
    fraction = answer_question(df, answer.rephrasings[0])
    assert fraction.ok and "**317**" in fraction.text  # 180 + 50 + 30 + 57


def test_net_revenue_unavailable_is_an_error_not_a_guess():
    df = orders().drop(columns="Discount_Pct")
    answer = answer_question(df, "Show net revenue by city")
    assert answer.status == ERROR
    assert "no discount column was found" in answer.text


def test_a_lone_units_column_is_never_used_as_the_price():
    # Regression: "unit price" used to match a `units` column (units x units).
    df = pd.DataFrame({"team": ["a", "b"], "units": [3, 4]})
    assert metrics.missing_inputs(df, GROSS_AMOUNT) == ["unit price"]
    with pytest.raises(MetricUnavailable, match="no unit price column"):
        compute_metric(df, GROSS_AMOUNT)


def test_quantity_and_price_are_never_the_same_column():
    df = pd.DataFrame({"unit_price": [2.0, 3.0]})
    assert metrics.missing_inputs(df, GROSS_AMOUNT) == ["quantity"]
