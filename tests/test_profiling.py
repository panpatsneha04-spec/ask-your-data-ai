import pandas as pd

from askyourdata import profiling


def make_df():
    return pd.DataFrame(
        {
            "city": ["Paris", "Rome", None, "Rome"],
            "sales": [10.0, 20.0, 30.0, 20.0],
        }
    )


def test_dataset_overview():
    overview = profiling.dataset_overview(make_df())
    assert overview.rows == 4
    assert overview.columns == 2
    assert overview.missing_cells == 1
    assert overview.duplicate_rows == 1


def test_missing_values_sorted_worst_first():
    table = profiling.missing_values(make_df())
    assert table.iloc[0]["column"] == "city"
    assert table.iloc[0]["missing"] == 1
    assert table.iloc[0]["missing_pct"] == 25.0


def test_column_info():
    info = profiling.column_info(make_df()).set_index("column")
    assert info.loc["sales", "dtype"] == "float64"
    assert info.loc["city", "non_null"] == 3
    assert info.loc["city", "unique_values"] == 2


def test_summaries_handle_missing_column_types():
    text_only = pd.DataFrame({"name": ["a", "b"]})
    numbers_only = pd.DataFrame({"n": [1, 2]})
    assert profiling.numeric_summary(text_only).empty
    assert profiling.categorical_summary(numbers_only).empty
    assert "mean" in profiling.numeric_summary(numbers_only).columns
