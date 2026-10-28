import io

import pandas as pd
import pytest

from askyourdata.data_loader import DataLoadError, load_file


def test_loads_valid_csv():
    df = load_file("data.csv", b"a,b\n1,x\n2,y\n")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_loads_valid_xlsx():
    buffer = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buffer, index=False)
    df = load_file("data.xlsx", buffer.getvalue())
    assert df.shape == (2, 2)


def test_handles_non_utf8_csv():
    df = load_file("data.csv", "name\nJosé\n".encode("latin-1"))
    assert df.iloc[0, 0] == "José"


@pytest.mark.parametrize(
    ("file_name", "content", "message"),
    [
        ("data.txt", b"a,b\n1,2\n", "Unsupported file type"),
        ("data.csv", b"", "empty"),
        ("data.csv", b"\n\n", "no data"),
        ("data.csv", b"a,b\n", "no data rows"),
        ("data.xlsx", b"this is not an excel file", "could not be read"),
    ],
)
def test_rejects_bad_files(file_name, content, message):
    with pytest.raises(DataLoadError, match=message):
        load_file(file_name, content)


def test_excel_upload_gives_the_same_results_as_csv():
    """The same data as .csv and as .xlsx must give identical summaries, insights and answers."""
    from pathlib import Path

    from askyourdata.insights import find_insights
    from askyourdata.question_router import answer_question
    from askyourdata.summary import build_summary

    csv_bytes = (Path(__file__).parent.parent / "data" / "sample_ecommerce.csv").read_bytes()
    from_csv = load_file("sample_ecommerce.csv", csv_bytes)
    buffer = io.BytesIO()
    from_csv.to_excel(buffer, index=False)
    from_xlsx = load_file("sample_ecommerce.xlsx", buffer.getvalue())

    assert from_xlsx.shape == from_csv.shape == (48, 9)
    assert build_summary(from_xlsx).headline == build_summary(from_csv).headline
    assert [i.title for i in find_insights(from_xlsx).insights] == [i.title for i in find_insights(from_csv).insights]
    question = "Show net revenue by city for Electronics"
    assert answer_question(from_xlsx, question).text == answer_question(from_csv, question).text
