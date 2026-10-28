"""Turn uploaded CSV / Excel files into pandas DataFrames.

The loader works on raw bytes instead of Streamlit objects, which keeps it
independent of the web interface and easy to test.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")

# CSV files exported from Excel often start with a hidden "BOM" marker, which
# "utf-8-sig" strips. "latin-1" is a fallback that can decode any byte.
CSV_ENCODINGS = ("utf-8-sig", "latin-1")


class DataLoadError(Exception):
    """Raised when a file cannot be turned into a usable DataFrame.

    The message is written for end users, so the app can show it directly.
    """


def load_file(file_name: str, content: bytes) -> pd.DataFrame:
    """Read a CSV or Excel file and return its contents as a DataFrame.

    Args:
        file_name: Original file name; its extension decides how to parse it.
        content: The raw bytes of the file.

    Raises:
        DataLoadError: If the file type is unsupported, the file is empty,
            or it cannot be parsed.
    """
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        raise DataLoadError(
            f"Unsupported file type '{extension or 'none'}'. Please upload one of: {allowed}."
        )

    if not content:
        raise DataLoadError("The file is empty (0 bytes).")

    if extension == ".csv":
        df = _read_csv(content)
    else:
        df = _read_excel(content)

    if df.shape[1] == 0:
        raise DataLoadError("No columns were found in the file.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has column headers but no data rows.")

    return df


def _read_csv(content: bytes) -> pd.DataFrame:
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue  # try the next encoding
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("The CSV file contains no data.") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(
                f"The CSV file could not be parsed. Check that every row has the same "
                f"number of columns. Details: {exc}"
            ) from exc

    raise DataLoadError("The CSV file's text encoding could not be recognised.")


def _read_excel(content: bytes) -> pd.DataFrame:
    try:
        # Only the first sheet is read for now.
        return pd.read_excel(io.BytesIO(content), sheet_name=0, engine="openpyxl")
    except Exception as exc:  # openpyxl raises many different error types
        raise DataLoadError(
            "The Excel file could not be read. Make sure it is a valid .xlsx file "
            f"and not password-protected. Details: {exc}"
        ) from exc
