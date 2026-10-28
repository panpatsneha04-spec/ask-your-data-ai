"""Tests for the executive summary and dataset-switching state helpers."""

from pathlib import Path

import pandas as pd
import pytest

from askyourdata.summary import build_summary
from askyourdata.ui_state import (
    ACTIVE_DATASET_KEY,
    dataset_fingerprint,
    reset_if_dataset_changed,
)

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def ecommerce():
    return pd.read_csv(DATA / "sample_ecommerce.csv")


@pytest.fixture(scope="module")
def sales():
    return pd.read_csv(DATA / "sample_sales.csv")


def kpis(summary):
    return {k.label: k for k in summary.kpis}


# --- KPIs ---------------------------------------------------------------------------


def test_basic_kpis_match_pandas(sales):
    summary = build_summary(sales)
    k = kpis(summary)
    cells = sales.size
    assert k["Rows"].value == "30" and k["Columns"].value == "9"
    assert k["Duplicate rows"].value == str(int(sales.duplicated().sum()))
    assert k["Complete cells"].value == f"{(cells - sales.isna().sum().sum()) / cells * 100:.1f}%"


def test_averages_skip_id_columns(sales):
    k = kpis(build_summary(sales))
    assert "Average order_id" not in k
    assert k["Average units"].value == f"{sales['units'].mean():.2f}"
    assert "1 missing ignored" in k["Average units"].help


def test_net_revenue_kpi_and_formula(ecommerce):
    summary = build_summary(ecommerce)
    net = ecommerce["Quantity"] * ecommerce["Unit_Price"] * (1 - ecommerce["Discount_Pct"] / 100)
    k = kpis(summary)["Total Net revenue (calculated)"]
    assert k.value == f"{net.sum():,.2f}"
    assert summary.metric.formula == (
        "Net revenue (calculated) = `Quantity` × `Unit_Price` × (1 − `Discount_Pct` / 100)"
    )
    assert "whole-number percentages" in summary.metric.notes[0]
    assert "2 row(s) with missing inputs are left out" in k.help


def test_gross_amount_when_there_is_no_discount(sales):
    summary = build_summary(sales)
    assert summary.metric.label == "Gross amount (calculated)"


def test_existing_revenue_column_is_used_not_derived():
    df = pd.DataFrame({"quantity": [1, 2], "unit_price": [5.0, 5.0], "Revenue": [7.0, 3.0]})
    summary = build_summary(df)
    assert summary.metric is None
    assert kpis(summary)["Total Revenue"].value == "10"


def test_ambiguous_discount_is_not_derived_and_is_explained():
    df = pd.DataFrame({"qty": [1, 2, 3], "price": [10.0, 10.0, 10.0], "discount": [0.1, 0.2, 0.0]})
    summary = build_summary(df)
    assert summary.metric.label == "Gross amount (calculated)"  # the unambiguous one
    assert any("Net revenue was not calculated" in s and "fraction" in s for s in summary.skipped)


def test_no_metric_for_non_sales_data():
    df = pd.DataFrame({"station": ["A", "B", "A"], "temp_c": [1.0, 2.0, 3.0]})
    summary = build_summary(df)
    assert summary.metric is None
    assert not any("revenue" in k.label.lower() or "amount" in k.label.lower() for k in summary.kpis)
    assert kpis(summary)["Average temp_c"].value == "2"


# --- trend ----------------------------------------------------------------------------


def test_trend_uses_the_single_date_column(ecommerce):
    trend = build_summary(ecommerce).trend
    assert trend.kind == "line" and "`Order_Date`" in trend.text
    assert trend.chart.data[trend.chart.y].sum() == pytest.approx(
        (ecommerce["Quantity"] * ecommerce["Unit_Price"] * (1 - ecommerce["Discount_Pct"] / 100)).sum()
    )
    assert "partly covered" in trend.notes[0]


def test_row_trend_without_metric():
    df = pd.DataFrame({"day": ["2026-01-01", "2026-01-01", "2026-03-05"], "n": [1.0, 2.0, 3.0]})
    trend = build_summary(df).trend
    assert trend.chart.y == "rows"
    assert trend.chart.data["rows"].sum() == 3


def test_constant_trend_says_so():
    df = pd.DataFrame({"day": pd.date_range("2026-01-01", periods=5).astype(str), "t": range(5)})
    assert "Every day has the same value: 1" in build_summary(df).trend.text


@pytest.mark.parametrize(
    ("df", "reason"),
    [
        (pd.DataFrame({"a": [1, 2]}), "no date column"),
        (pd.DataFrame({"start": ["2026-01-01", "2026-02-01"], "end": ["2026-01-05", "2026-02-09"]}),
         "several date columns"),
        (pd.DataFrame({"day": ["2026-01-01", "2026-01-01"]}), "fewer than two different dates"),
    ],
)  # fmt: skip
def test_trend_is_skipped_with_a_reason(df, reason):
    summary = build_summary(df)
    assert summary.trend is None
    assert any(reason in s for s in summary.skipped)


# --- comparison ---------------------------------------------------------------------------


def test_comparison_counts_rows_per_category():
    df = pd.DataFrame({"team": ["a", "b", "b", "c"], "score": [1.0, 2.0, 3.0, 4.0]})
    comparison = build_summary(df).comparison
    assert comparison.chart.data.set_index("team")["count"].to_dict() == {"b": 2, "a": 1, "c": 1}
    assert "**b** has the most rows: 2 (50.0% of the total)" in comparison.text


def test_comparison_reports_ties():
    df = pd.DataFrame({"team": ["a", "b"], "score": [1.0, 2.0]})
    assert "share the most rows" in build_summary(df).comparison.text


def test_comparison_with_metric_matches_pandas(ecommerce):
    comparison = build_summary(ecommerce).comparison
    table = comparison.chart.data.set_index("City")
    net = ecommerce["Quantity"] * ecommerce["Unit_Price"] * (1 - ecommerce["Discount_Pct"] / 100)
    expected = net.groupby(ecommerce["City"]).sum()
    assert table.loc["Pune", comparison.chart.y] == pytest.approx(expected["Pune"])


def test_comparison_skips_id_like_and_high_cardinality_text():
    df = pd.DataFrame({"name": [f"person {i}" for i in range(30)], "x": range(30)})
    summary = build_summary(df)
    assert summary.comparison is None
    assert any("no text column with 2-20 different values" in s for s in summary.skipped)


# --- data quality -----------------------------------------------------------------------------


def test_quality_notes():
    df = pd.DataFrame(
        {
            "city": ["Pune", "Pune", None, "Pune", "Pune"],
            "code": ["1", "2", "3", "4", "5"],
            "amount": ["10", "12", "n/a", "14", "15"],
            "empty": [None, None, None, None, None],
            "when": ["2026-01-01", "not a date", "2026-01-03", "2026-01-04", "2026-01-05"],
        }
    )
    notes = " | ".join(build_summary(df).quality)
    assert "`empty` is completely empty" in notes
    assert "`city` has 1 missing values (20.0%)" in notes
    assert "`city` has the same value in every row (Pune)" in notes
    assert "`code` contains only numbers but is stored as text" in notes
    assert "`amount` is stored as text: most values are numbers, but 1 are not" in notes
    assert "1 value(s) in `when` could not be read as dates" in notes


def test_id_columns_are_information_not_issues(ecommerce):
    summary = build_summary(ecommerce)
    assert summary.info == ("`Order_ID` looks like an ID, so it is not averaged.",)
    assert not any("looks like an ID" in q for q in summary.quality)


def test_clean_dataset_quality_message():
    df = pd.DataFrame({"team": ["a", "b"], "score": [1.0, 2.0]})
    assert build_summary(df).quality == (
        "No missing values, duplicate rows or other data-quality issues were found.",
    )


# --- empty and unsupported datasets ---------------------------------------------------------------


def test_dataset_without_rows(sales):
    summary = build_summary(sales.iloc[0:0])
    assert kpis(summary)["Rows"].value == "0"
    assert summary.trend is None and summary.comparison is None and summary.metric is None
    assert "no rows" in summary.headline
    assert summary.quality == ("The dataset has no rows, so there is nothing to check.",)


@pytest.mark.parametrize("df", [pd.DataFrame(), None])
def test_dataset_without_columns(df):
    summary = build_summary(df)
    assert summary.kpis == ()
    assert "no columns" in summary.headline


def test_summary_never_modifies_the_dataset(ecommerce):
    original = ecommerce.copy(deep=True)
    summary = build_summary(ecommerce)
    pd.testing.assert_frame_equal(ecommerce, original)
    assert "Net revenue (calculated)" in summary.data.columns  # only in the summary's copy


# --- dataset fingerprints and stale state ------------------------------------------------------------


def test_fingerprint_depends_on_values_not_just_shape():
    a = pd.DataFrame({"city": ["Pune", "Delhi"], "n": [1, 2]})
    b = pd.DataFrame({"city": ["Pune", "Delhi"], "n": [100, 200]})  # same name, shape and columns
    assert dataset_fingerprint(a) == dataset_fingerprint(a.copy())
    assert dataset_fingerprint(a) != dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(a.rename(columns={"n": "m"}))
    assert dataset_fingerprint(None) is None


def test_fingerprint_handles_unhashable_cells():
    df = pd.DataFrame({"tags": [["a", "b"], ["c"]]})
    assert dataset_fingerprint(df) == dataset_fingerprint(df.copy())


def test_reset_clears_only_dataset_specific_keys():
    state = {
        ACTIVE_DATASET_KEY: "old",
        "ask_history": ["old answer"],
        "grp_by": ["region"],
        "flt_col_0": "units",
        "db_connection": "keep me",
        "data_source": "Sample dataset",
    }
    assert reset_if_dataset_changed(state, "new") is True
    assert state == {ACTIVE_DATASET_KEY: "new", "db_connection": "keep me", "data_source": "Sample dataset"}


def test_reset_keeps_state_for_the_same_dataset():
    state = {ACTIVE_DATASET_KEY: "same", "ask_history": ["answer"]}
    assert reset_if_dataset_changed(state, "same") is False
    assert state["ask_history"] == ["answer"]


def test_reset_on_first_dataset_and_when_dataset_is_removed():
    state = {"ask_history": ["from nowhere"]}
    assert reset_if_dataset_changed(state, "first") is True and "ask_history" not in state
    state["ask_history"] = ["answer"]
    assert reset_if_dataset_changed(state, None) is True and "ask_history" not in state


def test_long_kpis_get_a_wider_card(ecommerce):
    # Regression: "Total Net revenue (calculated)" / "159,020.80" was cut off in a 1/4-width card.
    from summary_view import kpi_width

    from summary_view import KPIS_PER_ROW, kpi_rows

    kpis = build_summary(ecommerce).kpis
    widths = {k.label: kpi_width(k) for k in kpis}
    assert widths["Total Net revenue (calculated)"] == 2
    assert widths["Rows"] == 1 and widths["Average Quantity"] == 1
    rows = kpi_rows(kpis)
    assert [k for row in rows for k in row] == list(kpis)  # nothing lost, order kept
    assert all(sum(kpi_width(k) for k in row) <= KPIS_PER_ROW for row in rows)  # never squeezed
