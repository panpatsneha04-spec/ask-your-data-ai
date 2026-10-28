"""Tests for automatic, evidence-based insights (askyourdata/insights.py)."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from askyourdata import insights as ins
from askyourdata.insights import PERFORMANCE, QUALITY, TREND, find_insights

ROOT = Path(__file__).parent.parent


@pytest.fixture
def orders():
    """Hand-checkable data.

    region totals:  North 10+10+10+10 = 40, South 5*4 = 20, East 2+3+2+3 = 10  (total 70)
    month totals:   Jan 10+5+2+10 = 27, Feb 5+3+10+5 = 23, Mar 2+10+5+3 = 20
    rows per month: 4, 4, 4. Dates cover Jan 1 - Mar 31, so every month is complete.
    """
    return pd.DataFrame(
        {
            "order_id": range(1, 13),
            "order_date": ["2026-01-01", "2026-01-10", "2026-01-20", "2026-01-31", "2026-02-01", "2026-02-10",
                           "2026-02-20", "2026-02-28", "2026-03-01", "2026-03-10", "2026-03-20", "2026-03-31"],
            "region": ["North", "South", "East", "North", "South", "East",
                       "North", "South", "East", "North", "South", "East"],
            "quantity": [10, 5, 2, 10, 5, 3, 10, 5, 2, 10, 5, 3],
        }
    )  # fmt: skip


def by_kind(report, kind):
    return [i for i in report.insights if i.kind == kind]


def values(insight):
    return dict(insight.values)


# --- performance --------------------------------------------------------------------------------


def test_ranking_matches_hand_calculation(orders):
    report = find_insights(orders)
    [ranking] = by_kind(report, "ranking")
    assert ranking.title == "North has the highest total quantity by region"
    v = values(ranking)
    assert v["Highest"] == "North: 40" and v["Lowest"] == "East: 10"
    assert v["Highest ÷ lowest"] == "4.0×"
    assert v["Share of total"] == "57.1%"  # 40 / 70
    assert "not a statistical test" in ranking.explanation
    assert "summed" in ranking.evidence


def test_identifier_columns_are_never_ranked_or_summed(orders):
    report = find_insights(orders)
    text = " ".join(i.title + i.explanation + i.evidence + str(i.values) for i in report.insights)
    assert "order_id" not in text


def test_text_identifiers_are_not_used_as_groups(orders):
    df = orders.assign(customer_code=[f"C{i:03d}" for i in range(12)] * 1)
    df = pd.concat([df] * 3, ignore_index=True)
    df["customer_code"] = [f"C{i:03d}" for i in range(len(df))]  # unique per row
    report = find_insights(df)
    assert not any("customer_code" in i.key for i in report.insights)


def test_prices_are_averaged_not_summed():
    df = pd.DataFrame({"category": ["a", "a", "b", "b"], "unit_price": [10.0, 20.0, 4.0, 6.0]})
    [ranking] = by_kind(find_insights(df), "ranking")
    assert values(ranking)["Highest"] == "a: 15" and values(ranking)["Lowest"] == "b: 5"
    assert "averaged, not summed" in ranking.evidence
    assert "Share of total" not in values(ranking)  # shares of averages are meaningless


def test_rows_are_counted_when_nothing_numeric_can_be_measured():
    df = pd.DataFrame({"team": ["x", "x", "x", "y"], "id": [1, 2, 3, 4]})
    [ranking] = by_kind(find_insights(df), "ranking")
    assert values(ranking)["Highest"] == "x: 3" and values(ranking)["Share of total"] == "75.0%"


def test_metric_insight_only_when_unambiguous():
    df = pd.DataFrame({"city": ["P", "P", "M"], "qty": [2, 1, 4], "price": [10.0, 10.0, 5.0], "discount_pct": [10, 0, 20]})
    report = find_insights(df)
    [metric] = by_kind(report, "metric")
    assert metric.title == "Total Net revenue (calculated): 44"  # 18 + 10 + 16
    assert "Formula" in values(metric)
    ambiguous = df.assign(discount_pct=[0.1, 0.0, 0.2])  # 10% or 0.1%? -> not net revenue
    [metric] = by_kind(find_insights(ambiguous), "metric")
    assert metric.title.startswith("Total Gross amount (calculated)")


def test_existing_revenue_column_is_used_not_recalculated():
    # By revenue P leads (4 + 4 = 8 vs 1); by qty M would lead (4 vs 3).
    df = pd.DataFrame({"city": ["P", "P", "M"], "qty": [2, 1, 4], "price": [10.0, 10.0, 5.0], "revenue": [4.0, 4.0, 1.0]})
    report = find_insights(df)
    assert not by_kind(report, "metric")
    [ranking] = by_kind(report, "ranking")
    assert values(ranking)["Highest"] == "P: 8"  # the existing revenue column, summed


def test_top_items_for_many_values():
    # 25 products, two orders each: product pNN sold NN + 1 units in total.
    products = [f"p{i:02d}" for i in range(25)] * 2
    units = [i for i in range(25)] + [1] * 25
    df = pd.DataFrame({"product": products, "units": units})
    [top] = by_kind(find_insights(df), "top items")
    assert [v for _, v in top.values] == ["p24: 25", "p23: 24", "p22: 23", "p21: 22", "p20: 21"]
    assert not by_kind(find_insights(df), "ranking")  # 25 values: a list, not highest/lowest


def test_identifier_like_numbers_are_not_measures():
    df = pd.DataFrame({"product": ["a", "b", "c"] * 4, "row_no": range(12)})  # 0..11: an ID
    report = find_insights(df)
    assert "rows" in by_kind(report, "ranking")[0].title and "row_no" not in by_kind(report, "ranking")[0].title


def test_ties_zeros_and_negatives():
    tie = pd.DataFrame({"team": ["a", "b", "c"], "units": [5, 5, 1]})
    assert by_kind(find_insights(tie), "ranking")[0].title == "a and b share the highest total units by team"
    zero = pd.DataFrame({"team": ["a", "b"], "units": [4, 0]})
    v = values(by_kind(find_insights(zero), "ranking")[0])
    assert "Highest ÷ lowest" not in v  # no ratio against 0
    negative = pd.DataFrame({"team": ["a", "b", "c"], "units": [4, 2, -3]})
    ranking = by_kind(find_insights(negative), "ranking")[0]
    assert "negative values" in ranking.explanation and "Share of total" not in values(ranking)


def test_equal_groups_and_small_groups():
    equal = pd.DataFrame({"team": ["a", "a", "b", "b"], "units": [1, 2, 2, 1]})  # 3 and 3
    assert by_kind(find_insights(equal), "ranking")[0].title == "Every team has the same total units"
    small = pd.DataFrame({"team": ["a", "b", "b", "b"], "units": [9, 1, 1, 1]})
    assert "fewer than 3 rows for a" in by_kind(find_insights(small), "ranking")[0].explanation


def test_missing_group_values_are_left_out_and_reported(orders):
    df = orders.copy()
    df.loc[0, "region"] = None
    [ranking] = by_kind(find_insights(df), "ranking")
    assert "1 row(s) with a missing region were left out" in ranking.evidence
    assert values(ranking)["Highest"] == "North: 30"


# --- trends ---------------------------------------------------------------------------------------


def test_trend_matches_hand_calculation(orders):
    report = find_insights(orders)
    [main, rows] = by_kind(report, "change")
    assert main.title == "Total quantity fell 25.9% from Jan 2026 to Mar 2026"  # 27 -> 20
    assert values(main)["Change"] == "−7 (−25.9%)"
    assert "0 rise(s) and 2 fall(s)" in main.explanation
    assert rows.title == "Rows stayed roughly flat from Jan 2026 to Mar 2026"  # 4, 4, 4


def test_partly_covered_periods_are_left_out(orders):
    df = orders.copy()
    df.loc[0, "order_date"] = "2026-01-25"  # January is now only partly covered
    df.loc[1:3, "order_date"] = ["2026-01-26", "2026-01-27", "2026-01-31"]
    [main, _] = by_kind(find_insights(df), "change")
    assert main.title.endswith("from Feb 2026 to Mar 2026")
    assert "1 partly covered period(s)" in main.explanation


def test_trend_from_zero_has_no_percentage():
    # Three rows per month (the minimum for a monthly trend); January's units are all 0.
    df = pd.DataFrame({"day": ["2026-01-01", "2026-01-15", "2026-01-31", "2026-02-01", "2026-02-14",
                               "2026-02-28", "2026-03-01", "2026-03-15", "2026-03-31"],
                       "units": [0, 0, 0, 1, 1, 1, 2, 2, 2]})  # fmt: skip
    main = by_kind(find_insights(df), "change")[0]
    assert "no percentage" in values(main)["Change"]


@pytest.mark.parametrize(
    ("df", "note"),
    [
        (pd.DataFrame({"team": ["a", "b"], "units": [1, 2]}), "No date column"),
        (pd.DataFrame({"start": ["2026-01-01", "2026-02-01"], "end": ["2026-01-05", "2026-02-05"], "units": [1, 2]}),
         "several date columns"),
        (pd.DataFrame({"day": ["2026-01-01", "2026-01-01"], "units": [1, 2]}), "fewer than two different valid dates"),
        (pd.DataFrame({"day": ["2026-01-05", "2026-02-10"], "units": [1, 2]}), "fully covered"),
    ],
)  # fmt: skip
def test_no_trend_without_reliable_dates(df, note):
    report = find_insights(df)
    assert not report.by_category(TREND)
    assert note in report.notes[TREND]


def test_invalid_dates_are_reported_and_left_out(orders):
    df = orders.copy()
    df.loc[5, "order_date"] = "not a date"
    report = find_insights(df)
    [unreadable] = by_kind(report, "unreadable dates")
    assert values(unreadable)["Unreadable"] == "1"
    assert "1 row(s) with a missing or unreadable date were left out" in by_kind(report, "change")[0].explanation


# --- data quality -------------------------------------------------------------------------------


@pytest.fixture
def messy():
    return pd.DataFrame(
        {
            "team": ["a", "b"] * 6,
            "units": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 1000],  # 1000 is far outside the rest
            "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, -4],  # one negative
            "discount_pct": [5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 150],  # 150% is impossible
            "notes": [None] * 12,  # empty
            "country": ["IN"] * 12,  # constant
            "code": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "x"],  # mostly numbers
        }
    )


def test_quality_checks(messy):
    df = pd.concat([messy, messy.iloc[[0]]], ignore_index=True)  # one duplicate row
    quality = {i.kind: i for i in find_insights(df).by_category(QUALITY)}
    assert values(quality["potential outliers"])["Most extreme"] == "1,000"
    assert "heuristic" in quality["potential outliers"].explanation
    assert values(quality["negative values"]) == {"Negative values": "1", "Lowest": "-4"}
    assert "potential issue" in quality["negative values"].explanation
    assert values(quality["empty columns"])["Columns"] == "notes"
    assert values(quality["duplicate rows"])["Duplicate rows"] == "1"
    assert len(find_insights(df).by_category(QUALITY)) <= ins.LIMITS[QUALITY]


def test_remaining_quality_checks(messy):
    quality = {i.key: i for i in find_insights(messy).insights}
    assert values(quality["quality:percent:discount_pct"])["Out of range"] == "1"
    assert values(quality["quality:constant:country"])["Value"] == "IN"
    assert values(quality["quality:text-numbers:code"])["Non-numeric values"] == "1"


def test_missing_values_are_summarised(orders):
    df = orders.copy()
    df.loc[[1, 2], "quantity"] = np.nan
    [missing] = by_kind(find_insights(df), "missing values")
    assert values(missing) == {"quantity": "2 (16.7%)"}


def test_clean_data_gets_an_all_clear(orders):
    [ok] = find_insights(orders).by_category(QUALITY)
    assert ok.kind == "no issues"


# --- empty and unsupported data ----------------------------------------------------------------------


@pytest.mark.parametrize("df", [None, pd.DataFrame(), pd.DataFrame({"a": pd.Series([], dtype=float)})])
def test_empty_or_columnless_data(df):
    report = find_insights(df)
    assert report.insights == ()
    assert set(report.notes) == {PERFORMANCE, TREND, QUALITY}
    assert all("no insight can be generated" in n for n in report.notes.values())


def test_unsupported_schema_explains_why():
    df = pd.DataFrame({"comment": [f"text {i}" for i in range(30)]})
    report = find_insights(df)
    assert not report.by_category(PERFORMANCE) and not report.by_category(TREND)
    assert "No text column with 2–500 different values" in report.notes[PERFORMANCE]


def test_all_null_numeric_column_is_not_a_measure():
    df = pd.DataFrame({"team": ["a", "b"], "units": [np.nan, np.nan], "score": [1.0, 3.0]})
    report = find_insights(df)
    assert "average score" in by_kind(report, "ranking")[0].title
    assert by_kind(report, "empty columns")


# --- guarantees ------------------------------------------------------------------------------------------


def test_results_are_reproducible_and_order_independent(orders):
    first, second = find_insights(orders), find_insights(orders)
    shuffled = find_insights(orders.sample(frac=1, random_state=7))
    snapshot = lambda r: [(i.key, i.title, i.values) for i in r.insights]  # noqa: E731
    assert snapshot(first) == snapshot(second) == snapshot(shuffled)


def test_dataset_is_never_modified(orders, messy):
    for df in (orders, messy):
        original = df.copy(deep=True)
        find_insights(df)
        pd.testing.assert_frame_equal(df, original)


def test_no_causal_or_significance_claims(orders, messy):
    text = " ".join(i.title + " " + i.explanation for df in (orders, messy) for i in find_insights(df).insights)
    for phrase in ("because", "caused", "due to", "leads to", "statistically significant", "proves"):
        assert phrase not in text.lower()


def test_limits_and_unique_keys(orders):
    report = find_insights(orders)
    keys = [i.key for i in report.insights]
    assert len(keys) == len(set(keys))
    for category, limit in ins.LIMITS.items():
        assert len(report.by_category(category)) <= limit


def test_every_insight_carries_values_and_evidence(orders, messy):
    for df in (orders, messy):
        for insight in find_insights(df).insights:
            assert insight.title and insight.explanation and insight.values and insight.evidence
            assert insight.category in ins.CATEGORIES


def test_insight_source_has_no_dynamic_code_or_database_access():
    tree = ast.parse((ROOT / "askyourdata" / "insights.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__", "open"}
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name and ("sql" in name or "database" in name) for name in imported)
