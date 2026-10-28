import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from askyourdata import charts
from askyourdata import data_engine as de
from askyourdata.data_engine import AnalysisError, Condition


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "product": ["A", "B", "A", "C", None, "B"],
            "region": ["North", "South", "North", "East", "West", "South"],
            "units": [5, 3, 2, None, 4, 7],
            "price": [10.0, 20.0, 10.0, 5.0, 8.0, 20.0],
            "date": ["2026-01-05", "2026-01-20", "2026-02-03", "2026-03-15", "not a date", None],
            "in_stock": [True, False, True, True, False, True],
        }
    )


@pytest.fixture
def empty(sales):
    return sales.iloc[0:0]


# --- column types ----------------------------------------------------------


def test_column_kinds(sales):
    assert de.column_kinds(sales) == {
        "product": de.CATEGORICAL,
        "region": de.CATEGORICAL,
        "units": de.NUMERIC,
        "price": de.NUMERIC,
        "date": de.DATETIME,
        "in_stock": de.BOOLEAN,
    }


def test_numbers_stored_as_text_are_not_dates():
    assert de.column_kind(pd.Series(["1001", "1002", "1003"])) == de.CATEGORICAL


def test_column_helpers(sales):
    assert de.numeric_columns(sales) == ["units", "price"]
    assert de.date_columns(sales) == ["date"]
    assert de.categorical_columns(sales) == ["product", "region", "in_stock"]


# --- profiling ---------------------------------------------------------------


def test_profile_dataset(sales):
    profile = de.profile_dataset(sales)
    assert profile.overview.rows == 6
    assert profile.overview.missing_cells == 3
    kinds = profile.columns.set_index("column")["kind"]
    assert kinds["units"] == de.NUMERIC
    assert kinds["date"] == de.DATETIME


def test_describe_numeric_ignores_missing_values(sales):
    stats = de.describe_numeric(sales, ["units"])
    assert stats.loc["units", "missing"] == 1
    assert stats.loc["units", "count"] == 5
    assert stats.loc["units", "mean"] == pytest.approx(4.2)


def test_describe_numeric_errors(sales, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.describe_numeric(sales, ["product"])
    with pytest.raises(AnalysisError, match="Unknown column"):
        de.describe_numeric(sales, ["nope"])
    with pytest.raises(AnalysisError, match="no rows"):
        de.describe_numeric(empty)
    with pytest.raises(AnalysisError, match="no numeric columns"):
        de.describe_numeric(sales[["product"]])


def test_aggregate_column_ignores_missing_values(sales):
    result = de.aggregate_column(sales, "units", "mean")
    assert result.value == pytest.approx(4.2)
    assert (result.used_values, result.missing) == (5, 1)
    assert de.aggregate_column(sales, "units", "sum").value == 21
    assert de.aggregate_column(sales, "units", "max").value == 7


def test_aggregate_column_count_and_nunique_work_on_text(sales):
    assert de.aggregate_column(sales, "product", "count").value == 5
    assert de.aggregate_column(sales, "product", "nunique").value == 3


def test_aggregate_column_errors(sales, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.aggregate_column(sales, "product", "mean")
    with pytest.raises(AnalysisError, match="Unknown aggregation"):
        de.aggregate_column(sales, "units", "average")
    with pytest.raises(AnalysisError, match="Unknown column"):
        de.aggregate_column(sales, "nope", "sum")
    with pytest.raises(AnalysisError, match="no rows"):
        de.aggregate_column(empty, "units", "sum")
    with pytest.raises(AnalysisError, match="all missing"):
        de.aggregate_column(pd.DataFrame({"x": [np.nan, np.nan]}), "x", "mean")


# --- filtering ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "expected_rows"),
    [
        (Condition("units", de.GREATER, "3"), 3),  # missing units never match
        (Condition("units", de.NOT_EQUALS, 5), 4),  # ...not even "does not equal"
        (Condition("units", de.LESS_EQUAL, "1,000"), 5),  # thousands separator accepted
        (Condition("product", de.EQUALS, " a "), 2),  # case and spaces ignored
        (Condition("product", de.CONTAINS, "b"), 2),
        (Condition("product", de.NOT_CONTAINS, "a"), 3),
        (Condition("product", de.CONTAINS, ".*"), 0),  # text is literal, not a regex
        (Condition("region", de.IS_ONE_OF, "north, EAST"), 3),
        (Condition("region", de.IS_ONE_OF, ["South"]), 2),
        (Condition("product", de.IS_MISSING), 1),
        (Condition("units", de.IS_NOT_MISSING), 5),
        (Condition("date", de.GREATER, "2026-02-01"), 2),
        (Condition("in_stock", de.EQUALS, "yes"), 4),
    ],
)
def test_filter_rows(sales, condition, expected_rows):
    assert len(de.filter_rows(sales, [condition])) == expected_rows


def test_filter_match_all_and_any(sales):
    conditions = [Condition("units", de.GREATER, 6), Condition("product", de.EQUALS, "C")]
    assert len(de.filter_rows(sales, conditions, match="all")) == 0
    assert len(de.filter_rows(sales, conditions, match="any")) == 2


def test_filter_without_conditions_returns_a_copy(sales):
    result = de.filter_rows(sales, [])
    assert result.equals(sales)
    assert result is not sales


def test_filter_on_empty_data_returns_empty(empty):
    assert de.filter_rows(empty, [Condition("units", de.GREATER, 1)]).empty


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (Condition("nope", de.EQUALS, 1), "Unknown column"),
        (Condition("units", "drop table", 1), "Unknown filter condition"),
        (Condition("product", de.GREATER, "A"), "cannot be used"),
        (Condition("units", de.CONTAINS, "5"), "cannot be used"),
        (Condition("units", de.GREATER, "abc"), "must be a number"),
        (Condition("date", de.GREATER, "someday"), "must be a date"),
        (Condition("in_stock", de.EQUALS, "maybe"), "true or false"),
        (Condition("units", de.EQUALS, "  "), "enter a value"),
        (Condition("region", de.IS_ONE_OF, " , "), "at least one value"),
    ],
)
def test_filter_rejects_invalid_conditions(sales, condition, message):
    with pytest.raises(AnalysisError, match=message):
        de.filter_rows(sales, [condition])


def test_filter_rejects_bad_match_mode(sales):
    with pytest.raises(AnalysisError, match="match mode"):
        de.filter_rows(sales, [], match="some")


# --- sorting -----------------------------------------------------------------


def test_sort_descending_puts_missing_last(sales):
    result = de.sort_rows(sales, "units", ascending=False)
    assert result["units"].tolist()[:5] == [7, 5, 4, 3, 2]
    assert pd.isna(result["units"].iloc[-1])


def test_sort_by_multiple_columns(sales):
    result = de.sort_rows(sales, ["region", "units"], ascending=[True, False])
    assert result["region"].tolist() == ["East", "North", "North", "South", "South", "West"]
    assert result["units"].tolist()[1:3] == [5, 2]


def test_sort_text_dates_by_date_not_alphabet():
    df = pd.DataFrame({"when": ["Jan 15 2026", "Feb 2 2026", "Dec 31 2025"]})
    assert de.sort_rows(df, "when")["when"].tolist() == ["Dec 31 2025", "Jan 15 2026", "Feb 2 2026"]


def test_sort_mixed_numbers_and_text():
    df = pd.DataFrame({"mixed": [3, "b", 1]})
    assert len(de.sort_rows(df, "mixed")) == 3


def test_sort_errors(sales):
    with pytest.raises(AnalysisError, match="Unknown column"):
        de.sort_rows(sales, "nope")
    with pytest.raises(AnalysisError, match="one sort direction"):
        de.sort_rows(sales, ["units", "price"], ascending=[True])
    with pytest.raises(AnalysisError, match="at least one column"):
        de.sort_rows(sales, [])


# --- grouping ----------------------------------------------------------------


def test_group_summary(sales):
    result = de.group_summary(sales, "region", "units").set_index("region")
    assert result.loc["South", "rows"] == 2
    assert result.loc["South", "sum of units"] == 10
    assert result.loc["South", "mean of units"] == 5
    assert result.loc["North", "min of units"] == 2
    # East's only unit value is missing: counted as a row, ignored by the maths
    assert result.loc["East", "rows"] == 1
    assert result.loc["East", "count of units"] == 0


def test_group_summary_labels_missing_groups(sales):
    result = de.group_summary(sales, "product")
    assert de.MISSING_LABEL in result["product"].tolist()
    assert list(result.columns) == ["product", "rows"]


def test_group_by_two_columns(sales):
    result = de.group_summary(sales, ["region", "product"], "units", ["sum"])
    assert list(result.columns) == ["region", "product", "rows", "sum of units"]
    assert result.iloc[0]["rows"] == 2  # biggest group first


def test_group_summary_errors(sales, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.group_summary(sales, "region", "product")
    with pytest.raises(AnalysisError, match="Unknown aggregation"):
        de.group_summary(sales, "region", "units", ["average"])
    with pytest.raises(AnalysisError, match="Unknown column"):
        de.group_summary(sales, "nope")
    with pytest.raises(AnalysisError, match="cannot also be"):
        de.group_summary(sales, "region", "region")
    with pytest.raises(AnalysisError, match="no rows"):
        de.group_summary(empty, "region")
    many_numbers = pd.DataFrame({"amount": np.arange(30.0)})
    with pytest.raises(AnalysisError, match="too many"):
        de.group_summary(many_numbers, "amount")


# --- top N -------------------------------------------------------------------


def test_top_n_most_common(sales):
    result = de.top_n(sales, "region", n=1)
    assert result.iloc[0]["count"] == 2
    assert result.iloc[0]["percent"] == pytest.approx(33.33)


def test_top_n_ranked_by_value(sales):
    result = de.top_n(sales, "product", n=2, value_column="units", agg="sum")
    assert result["product"].tolist() == ["B", "A"]
    assert result["sum of units"].tolist() == [10, 7]
    smallest = de.top_n(sales, "product", n=1, value_column="units", largest=False)
    assert smallest["product"].tolist() == ["C"]


@pytest.mark.parametrize("bad_n", [0, -1, "5", 2.5, True])
def test_top_n_rejects_bad_n(sales, bad_n):
    with pytest.raises(AnalysisError, match="N"):
        de.top_n(sales, "product", n=bad_n)


def test_top_n_errors(sales, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.top_n(sales, "product", value_column="region")
    with pytest.raises(AnalysisError, match="Unknown column"):
        de.top_n(sales, "nope")
    with pytest.raises(AnalysisError, match="no rows"):
        de.top_n(empty, "product")


def test_top_rows(sales):
    assert de.top_rows(sales, "units", n=2)["units"].tolist() == [7, 5]
    assert de.top_rows(sales, "units", n=1, largest=False)["units"].tolist() == [2]
    with pytest.raises(AnalysisError, match="numeric"):
        de.top_rows(sales, "product")


# --- date trends -------------------------------------------------------------


def test_monthly_row_counts_skip_bad_dates(sales):
    trend = de.date_trend(sales, "date", freq="Month")
    assert trend.table["rows"].tolist() == [2, 1, 1]
    assert trend.table["period"].iloc[0] == pd.Timestamp("2026-01-01")
    assert trend.skipped_rows == 2  # "not a date" and the missing date


def test_monthly_sum_of_values(sales):
    trend = de.date_trend(sales, "date", "units", freq="Month", agg="sum")
    assert trend.value_label == "sum of units"
    assert trend.table["sum of units"].tolist() == [8, 2, 0]


def test_trend_fills_gaps():
    df = pd.DataFrame(
        {"day": pd.to_datetime(["2026-01-10", "2026-03-10"]), "amount": [1.0, 2.0]}
    )
    counts = de.date_trend(df, "day", freq="Month").table
    assert counts["rows"].tolist() == [1, 0, 1]
    means = de.date_trend(df, "day", "amount", freq="Month", agg="mean").table
    assert pd.isna(means["mean of amount"].iloc[1])  # no data is not the same as 0


def test_trend_errors(sales, empty):
    with pytest.raises(AnalysisError, match="does not contain dates"):
        de.date_trend(sales, "region")
    with pytest.raises(AnalysisError, match="Unknown time period"):
        de.date_trend(sales, "date", freq="Fortnight")
    with pytest.raises(AnalysisError, match="numeric"):
        de.date_trend(sales, "date", "product")
    with pytest.raises(AnalysisError, match="no rows"):
        de.date_trend(empty, "date")


# --- correlations ------------------------------------------------------------


@pytest.fixture
def numbers():
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "double": [2, 4, 6, 8, 10, None],
            "reverse": [6, 5, 4, 3, 2, 1],
            "constant": [1, 1, 1, 1, 1, 1],
            "label": list("abcdef"),
        }
    )


def test_correlation_matrix(numbers):
    corr = de.correlation_matrix(numbers)
    assert list(corr.columns) == ["x", "double", "reverse", "constant"]
    assert corr.loc["x", "double"] == pytest.approx(1)  # missing value skipped
    assert corr.loc["x", "reverse"] == pytest.approx(-1)


def test_strongest_correlations(numbers):
    pairs = de.strongest_correlations(de.correlation_matrix(numbers))
    assert abs(pairs.iloc[0]["correlation"]) == pytest.approx(1)
    assert "constant" not in set(pairs["column_a"]) | set(pairs["column_b"])
    assert set(pairs["strength"]) == {"strong positive", "strong negative"}


@pytest.mark.parametrize(
    ("r", "label"), [(0.9, "strong positive"), (-0.5, "moderate negative"), (0.05, "very weak / none")]
)
def test_correlation_strength(r, label):
    assert de.correlation_strength(r) == label


def test_correlation_errors(numbers, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.correlation_matrix(numbers, ["x", "label"])
    with pytest.raises(AnalysisError, match="at least two"):
        de.correlation_matrix(numbers, ["x"])
    with pytest.raises(AnalysisError, match="correlation method"):
        de.correlation_matrix(numbers, method="kendall")
    with pytest.raises(AnalysisError, match="no rows"):
        de.correlation_matrix(empty)


# --- chart data --------------------------------------------------------------


def test_bar_chart_data(sales):
    chart = de.bar_chart_data(sales, "product", "units")
    assert (chart.x, chart.y) == ("product", "sum of units")
    assert set(chart.data.columns) == {"product", "sum of units"}
    counts = de.bar_chart_data(sales, "region")
    assert counts.y == "count"


def test_line_chart_data(sales):
    chart = de.line_chart_data(sales, "date", freq="Month")
    assert (chart.x, chart.y) == ("period", "rows")
    assert "2 row(s)" in chart.note


def test_histogram_data(sales):
    chart = de.histogram_data(sales, "units", bins=3)
    assert len(chart.data) == 3
    assert chart.data["count"].sum() == 5
    assert "1 missing" in chart.note


def test_histogram_single_value():
    chart = de.histogram_data(pd.DataFrame({"v": [5, 5, 5]}), "v", bins=4)
    assert chart.data["count"].sum() == 3


def test_histogram_errors(sales, empty):
    with pytest.raises(AnalysisError, match="numeric"):
        de.histogram_data(sales, "product")
    with pytest.raises(AnalysisError, match="at most 200"):
        de.histogram_data(sales, "units", bins=500)
    with pytest.raises(AnalysisError, match="at least 1"):
        de.histogram_data(sales, "units", bins=0)
    with pytest.raises(AnalysisError, match="no numeric values"):
        de.histogram_data(pd.DataFrame({"v": [np.nan, np.nan]}), "v")
    with pytest.raises(AnalysisError, match="no rows"):
        de.histogram_data(empty, "units")


def test_chart_builders_return_figures(sales, numbers):
    figures = [
        charts.bar_chart(de.bar_chart_data(sales, "product")),
        charts.line_chart(de.line_chart_data(sales, "date")),
        charts.histogram_chart(de.histogram_data(sales, "units")),
        charts.correlation_heatmap(de.correlation_matrix(numbers)),
    ]
    assert all(isinstance(fig, go.Figure) for fig in figures)


# --- the original data is never modified ---------------------------------------


@pytest.mark.parametrize(
    "analysis",
    [
        lambda df: de.profile_dataset(df),
        lambda df: de.describe_numeric(df),
        lambda df: de.filter_rows(df, [Condition("date", de.GREATER, "2026-01-01")]),
        lambda df: de.sort_rows(df, ["date", "units"]),
        lambda df: de.group_summary(df, "product", "units"),
        lambda df: de.top_n(df, "product", value_column="units"),
        lambda df: de.top_rows(df, "units"),
        lambda df: de.date_trend(df, "date", "units"),
        lambda df: de.correlation_matrix(df),
        lambda df: de.histogram_data(df, "units"),
    ],
)
def test_analyses_do_not_modify_input(sales, analysis):
    original = sales.copy(deep=True)
    analysis(sales)
    pd.testing.assert_frame_equal(sales, original)
