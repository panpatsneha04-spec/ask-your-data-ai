"""Plotly figures built from the chart-ready data in ``data_engine``."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from askyourdata.data_engine import ChartData


def bar_chart(chart: ChartData) -> go.Figure:
    fig = px.bar(chart.data, x=chart.x, y=chart.y, title=chart.title)
    # Treat the categories as labels, even when they look like numbers or dates.
    fig.update_xaxes(type="category")
    return fig


def line_chart(chart: ChartData) -> go.Figure:
    return px.line(chart.data, x=chart.x, y=chart.y, title=chart.title, markers=True)


def histogram_chart(chart: ChartData) -> go.Figure:
    data = chart.data
    fig = px.bar(
        data,
        x=chart.x,
        y=chart.y,
        title=chart.title,
        hover_data={"bin_range": True, chart.x: False},
    )
    width = float(data["bin_end"].iloc[0] - data["bin_start"].iloc[0])
    if width > 0:
        fig.update_traces(width=width)
    fig.update_layout(bargap=0.05, xaxis_title="value")
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Correlation between numeric columns",
    )
    return fig


def pie_chart(chart: ChartData) -> go.Figure:
    """Shares of a whole; only meaningful for non-negative totals or counts."""
    fig = px.pie(chart.data, names=chart.x, values=chart.y, title=chart.title)
    fig.update_traces(textinfo="percent+label")
    return fig
