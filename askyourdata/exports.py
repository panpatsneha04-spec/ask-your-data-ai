"""Turn answer tables and charts into downloadable files.

``display_table`` is used both for showing a table in the app and for the
CSV export, so the download always contains exactly what is on screen.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go


def display_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Make row labels a visible column, so nothing is hidden on screen or in CSV.

    Results such as statistics (one row per column name) or a filtered list of
    rows (original row numbers) keep their labels as the first column.
    """
    if table is None:
        return None
    if table.index.name is None and table.index.equals(pd.RangeIndex(len(table))):
        return table  # plain 0, 1, 2, ... row numbers carry no information
    name = table.index.name
    if name is None:
        name = "row" if pd.api.types.is_numeric_dtype(table.index) else "column"
    while name in table.columns:
        name = f"{name}_"
    return table.rename_axis(name).reset_index()


def table_to_csv(table: pd.DataFrame) -> bytes:
    """CSV bytes of the displayed table (UTF-8 with BOM so Excel shows accents)."""
    return display_table(table).to_csv(index=False).encode("utf-8-sig")


def figure_to_html(figure: go.Figure) -> bytes:
    """A standalone interactive chart. Plotly's script loads from its CDN when opened."""
    return figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def export_filename(question: str, extension: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(question).lower()).strip("-")[:50].strip("-")
    return f"askyourdata-{slug or 'answer'}.{extension}"
