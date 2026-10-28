"""Tests for CSV / chart exports of Ask Your Data answers."""

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from askyourdata import exports
from askyourdata.question_router import answer_question, suggest_questions

ECOMMERCE_CSV = Path(__file__).parent.parent / "data" / "sample_ecommerce.csv"


@pytest.fixture(scope="module")
def ecommerce():
    return pd.read_csv(ECOMMERCE_CSV)


def read_back(csv_bytes):
    return pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")


# --- display_table -----------------------------------------------------------------


def test_plain_tables_are_shown_as_is():
    table = pd.DataFrame({"a": [1, 2]})
    assert exports.display_table(table) is table
    assert exports.display_table(None) is None


def test_original_row_numbers_become_a_column():
    table = pd.DataFrame({"a": [1, 2, 3]}).iloc[[0, 2]]
    shown = exports.display_table(table)
    assert list(shown.columns) == ["row", "a"]
    assert shown["row"].tolist() == [0, 2]


def test_text_labels_become_a_column():
    stats = pd.DataFrame({"mean": [1.5]}, index=["price"])
    assert exports.display_table(stats).to_dict("list") == {"column": ["price"], "mean": [1.5]}


def test_label_column_name_never_overwrites_data():
    table = pd.DataFrame({"row": [9, 9]}, index=[5, 6])
    assert list(exports.display_table(table).columns) == ["row_", "row"]


# --- CSV exports match the displayed answers ------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Show net revenue by city.",
        "Show orders from Pune.",
        "Find the top 5 products in Pune.",
        "Show the distribution of unit price",
        "Calculate net revenue.",
        "What is the order trend by month?",
        "Show the correlation between numeric columns.",
        "Are there any missing values?",
    ],
)
def test_csv_matches_displayed_table(ecommerce, question):
    answer = answer_question(ecommerce, question)
    assert answer.ok, answer.text
    shown = exports.display_table(answer.table)
    exported = read_back(exports.table_to_csv(answer.table))
    assert list(exported.columns) == [str(c) for c in shown.columns]
    assert len(exported) == len(shown)
    expected = shown.copy()
    expected.columns = [str(c) for c in expected.columns]
    for column in expected.columns:  # compare values the way a spreadsheet would see them
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            expected[column] = expected[column].dt.strftime("%Y-%m-%d")
    pd.testing.assert_frame_equal(
        exported, read_back(expected.to_csv(index=False).encode("utf-8-sig")), check_dtype=False
    )


def test_csv_numbers_are_exact(ecommerce):
    answer = answer_question(ecommerce, "Show net revenue by city.")
    exported = read_back(exports.table_to_csv(answer.table)).set_index("City")
    expected = (
        ecommerce.assign(net=ecommerce["Quantity"] * ecommerce["Unit_Price"] * (1 - ecommerce["Discount_Pct"] / 100))
        .groupby("City")["net"].sum()
    )  # fmt: skip
    for city, value in expected.items():
        assert exported.loc[city, "sum of Net revenue (calculated)"] == pytest.approx(value)


def test_each_answer_exports_its_own_rows(ecommerce):
    pune = read_back(exports.table_to_csv(answer_question(ecommerce, "orders from Pune").table))
    upi = read_back(exports.table_to_csv(answer_question(ecommerce, "orders paid by UPI").table))
    assert set(pune["City"]) == {"Pune"}
    assert set(upi["Payment_Method"]) == {"UPI"}


def test_csv_has_excel_friendly_encoding():
    data = exports.table_to_csv(pd.DataFrame({"city": ["Pune"], "price": [1.5]}))
    assert data.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8 correctly
    assert data.decode("utf-8-sig").splitlines() == ["city,price", "Pune,1.5"]


# --- chart exports and file names ---------------------------------------------------------------


def test_chart_html_export(ecommerce):
    figure = answer_question(ecommerce, "Show net revenue by city.").figure
    html = exports.figure_to_html(figure)
    assert b"<html>" in html[:100] and b"plotly" in html
    assert len(html) < 200_000  # small: Plotly's script is loaded from its CDN


def test_html_export_accepts_any_figure():
    assert b"<html>" in exports.figure_to_html(go.Figure())


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Show net revenue by city.", "askyourdata-show-net-revenue-by-city.csv"),
        ("../../etc/passwd", "askyourdata-etc-passwd.csv"),
        ("???", "askyourdata-answer.csv"),
    ],
)
def test_export_filenames_are_safe(question, expected):
    assert exports.export_filename(question, "csv") == expected


# --- sample dataset regression --------------------------------------------------------------------


def test_every_suggested_question_works_on_the_ecommerce_sample(ecommerce):
    suggestions = suggest_questions(ecommerce)
    assert any("net revenue" in s for s in suggestions)
    assert any(" for " in s for s in suggestions)  # a filter example
    for question in suggestions:
        answer = answer_question(ecommerce, question)
        assert answer.ok, f"{question!r} -> {answer.text}"
