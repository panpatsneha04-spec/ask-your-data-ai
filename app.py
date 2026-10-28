"""AskYourData AI - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from askyourdata import charts, data_engine, profiling, ui_state
from askyourdata.data_engine import AnalysisError, ChartData, Condition
from askyourdata.data_loader import SUPPORTED_EXTENSIONS, DataLoadError, load_file
from askyourdata.db_config import load_env_file
from askyourdata.question_router import suggest_questions
from askyourdata.insights import InsightReport, find_insights
from askyourdata.summary import ExecutiveSummary, build_summary
from ask_your_data import show_ask_your_data
from database_explorer import (
    DATA_SOURCE_KEY,
    DATABASE_SOURCE,
    SESSION_DATASET,
    show_database_explorer,
)
from insights_view import show_insights
from summary_view import show_executive_summary

PROJECT_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = PROJECT_DIR / "data" / "sample_sales.csv"
ECOMMERCE_DATA_PATH = PROJECT_DIR / "data" / "sample_ecommerce.csv"
UPLOAD_SOURCE = "Upload a file"
SAMPLE_SOURCE = "Sample dataset"
ECOMMERCE_SOURCE = "Sample e-commerce dataset"
DATA_SOURCES = (UPLOAD_SOURCE, SAMPLE_SOURCE, ECOMMERCE_SOURCE, DATABASE_SOURCE)
SOURCE_CAPTIONS = (
    "Your own CSV or Excel (.xlsx) file",
    "Small synthetic sales data",
    "Synthetic orders with discounts",
    "Rows sent from the Database Explorer",
)
SAMPLES = {
    SAMPLE_SOURCE: (SAMPLE_DATA_PATH, "A small synthetic sales dataset bundled with the app."),
    ECOMMERCE_SOURCE: (
        ECOMMERCE_DATA_PATH,
        "Synthetic Indian e-commerce orders (Discount_Pct is a whole-number percentage, "
        "10 = 10%). Try: 'Show net revenue by city'.",
    ),
}
PREVIEW_ROW_OPTIONS = (10, 50, 100, 500)
NO_VALUE_COLUMN = "(none - just count rows)"
ANALYSIS_TOOLS = (
    "Filter & sort",
    "Group & summarise",
    "Top N",
    "Trends over time",
    "Correlations",
    "Distribution",
)
TAB_LABELS = ("📋 Summary", "💡 Insights", "🧭 Explore", "🔎 Analysis tools", "💬 Ask Your Data", "🗄️ Database Explorer")
# Cached data and results are kept in memory only, for a limited number of datasets
# and at most an hour, so memory use stays bounded and uploads are not kept forever.
CACHE_ENTRIES = 20
CACHE_SECONDS = 3600
NO_DATASET_MESSAGE = (
    "No dataset is selected yet. Choose a data source in the sidebar, or click one of the "
    "sample buttons above."
)


@st.cache_data(show_spinner="Reading file...", max_entries=CACHE_ENTRIES, ttl=CACHE_SECONDS)
def cached_load(file_name: str, content: bytes) -> pd.DataFrame:
    """Load a file once and reuse the result while the user clicks around.

    Streamlit re-runs this whole script on every interaction; caching stops
    the file from being parsed again each time.
    """
    return load_file(file_name, content)


@st.cache_data(show_spinner="Summarising the dataset...", max_entries=CACHE_ENTRIES, ttl=CACHE_SECONDS)
def cached_summary(fingerprint: str, _df: pd.DataFrame) -> ExecutiveSummary:
    """The summary is calculated once per dataset (identified by its fingerprint)."""
    return build_summary(_df)


INSIGHTS_NONCE_KEY = "insights_nonce"  # dataset-scoped: cleared when the dataset changes


@st.cache_data(show_spinner="Looking for insights...", max_entries=CACHE_ENTRIES, ttl=CACHE_SECONDS)
def cached_insights(fingerprint: str, nonce: int, _df: pd.DataFrame) -> InsightReport:
    """Insights for one dataset (its fingerprint). "Regenerate" bumps ``nonce``."""
    return find_insights(_df)


def _regenerate_insights() -> None:
    st.session_state[INSIGHTS_NONCE_KEY] = st.session_state.get(INSIGHTS_NONCE_KEY, 0) + 1


def load_dataset(source: str) -> tuple[tuple[str, pd.DataFrame] | None, str | None]:
    """The dataset for the chosen source, as ((name, DataFrame) or None, error message)."""
    if source == DATABASE_SOURCE:
        return st.session_state.get(SESSION_DATASET), None

    if source in SAMPLES:
        path, caption = SAMPLES[source]
        st.sidebar.caption(caption)
        name, content = path.name, path.read_bytes()
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload a CSV or Excel file",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            help="The file stays on this computer; it is only read into memory.",
        )
        if uploaded is None:
            return None, None
        name, content = uploaded.name, uploaded.getvalue()

    try:
        return (name, cached_load(name, content)), None
    except DataLoadError as exc:
        return None, f"Could not load **{name}**: {exc}"
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        return None, f"An unexpected error occurred while reading **{name}** ({type(exc).__name__})."


def show_overview(df: pd.DataFrame) -> None:
    overview = profiling.dataset_overview(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{overview.rows:,}")
    col2.metric("Columns", f"{overview.columns:,}")
    col3.metric("Missing cells", f"{overview.missing_cells:,}")
    col4.metric("Duplicate rows", f"{overview.duplicate_rows:,}")
    memory = (
        f"{overview.memory_mb:.2f} MB"
        if overview.memory_mb >= 1
        else f"{overview.memory_mb * 1024:.1f} KB"
    )
    col5.metric("Memory", memory)


def show_preview(df: pd.DataFrame) -> None:
    n_rows = st.selectbox("Rows to show", PREVIEW_ROW_OPTIONS, index=0, key="preview_rows")
    st.dataframe(df.head(n_rows))


def show_columns(df: pd.DataFrame) -> None:
    st.caption(
        "**kind** is how the analysis tools treat each column: numeric, datetime, "
        "categorical (text) or boolean (True/False)."
    )
    st.dataframe(data_engine.profile_dataset(df).columns, hide_index=True)


def show_missing_values(df: pd.DataFrame) -> None:
    missing = profiling.missing_values(df)
    if missing["missing"].sum() == 0:
        st.success("No missing values found.")
        return

    st.dataframe(missing, hide_index=True)
    with_missing = missing[missing["missing"] > 0]
    fig = px.bar(
        with_missing,
        x="column",
        y="missing",
        hover_data=["missing_pct"],
        title="Missing values per column",
    )
    st.plotly_chart(fig)


def show_duplicates(df: pd.DataFrame) -> None:
    count = profiling.duplicate_row_count(df)
    if count == 0:
        st.success("No duplicate rows found.")
        return

    st.warning(f"{count:,} row(s) are exact duplicates of an earlier row.")
    # keep=False marks every copy, so the user sees the originals too.
    st.dataframe(df[df.duplicated(keep=False)])


def show_statistics(df: pd.DataFrame) -> None:
    st.subheader("Numeric columns")
    numeric = profiling.numeric_summary(df)
    if numeric.empty:
        st.info("This dataset has no numeric columns.")
    else:
        st.dataframe(numeric.round(2))

    st.subheader("Text / categorical columns")
    categorical = profiling.categorical_summary(df)
    if categorical.empty:
        st.info("This dataset has no text or categorical columns.")
    else:
        st.dataframe(categorical.astype(str))


# ---------------------------------------------------------------------------
# Data analysis tab
# ---------------------------------------------------------------------------


def show_chart_note(chart: ChartData) -> None:
    if chart.note:
        st.caption(f"ℹ️ {chart.note}")


def analysis_filter_sort(df: pd.DataFrame) -> None:
    st.markdown(
        "Keep only the rows that match your conditions, then sort them. "
        "Text matching ignores upper/lower case. Your original data is never changed."
    )
    n_conditions = st.number_input("Number of conditions", 0, 5, 0, key="flt_n")
    match = st.radio(
        "Rows must match",
        ["all", "any"],
        format_func=lambda m: "ALL conditions" if m == "all" else "ANY condition",
        horizontal=True,
        key="flt_match",
    )

    conditions = []
    for i in range(int(n_conditions)):
        col_c, col_op, col_v = st.columns([2, 2, 3])
        column = col_c.selectbox("Column", list(df.columns), key=f"flt_col_{i}")
        op = col_op.selectbox(
            "Condition", data_engine.operators_for(df[column]), key=f"flt_op_{i}"
        )
        value = None
        if op not in data_engine.VALUELESS_OPERATORS:
            hint = "Separate several values with commas" if op == data_engine.IS_ONE_OF else None
            value = col_v.text_input("Value", key=f"flt_val_{i}", help=hint)
        conditions.append(Condition(column, op, value))

    sort_columns = st.multiselect(
        "Sort by (optional, first column has priority)", list(df.columns), key="srt_cols"
    )
    descending = st.toggle("Sort descending (largest / latest first)", key="srt_desc")

    result = data_engine.filter_rows(df, conditions, match=match)
    if sort_columns:
        result = data_engine.sort_rows(result, sort_columns, ascending=not descending)

    st.write(f"**{len(result):,}** of {len(df):,} rows match.")
    st.dataframe(result)
    st.download_button(
        "⬇️ Download these rows as CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )


def analysis_group(df: pd.DataFrame) -> None:
    st.markdown(
        "Split the rows into groups (e.g. one per region) and summarise a number "
        "for each group. Missing group values become their own **(missing)** group."
    )
    group_options = data_engine.categorical_columns(df)
    if not group_options:
        st.info("This dataset has no text or category columns to group by.")
        return

    group_by = st.multiselect(
        "Group by", group_options, default=group_options[:1], key="grp_by"
    )
    value_column = st.selectbox(
        "Number to summarise",
        [NO_VALUE_COLUMN, *data_engine.numeric_columns(df)],
        key="grp_val",
    )
    aggregations = st.multiselect(
        "Calculations",
        data_engine.AGGREGATIONS,
        default=data_engine.DEFAULT_GROUP_AGGREGATIONS,
        key="grp_aggs",
        disabled=value_column == NO_VALUE_COLUMN,
    )

    result = data_engine.group_summary(
        df,
        group_by,
        value_column=None if value_column == NO_VALUE_COLUMN else value_column,
        aggregations=aggregations,
    )
    st.dataframe(result.round(2), hide_index=True)

    if len(group_by) == 1:
        measures = [c for c in result.columns if c != group_by[0]]
        measure = st.selectbox("Chart this column", measures, key="grp_chart")
        chart = ChartData(
            data=result, x=group_by[0], y=measure, title=f"{measure} by {group_by[0]}"
        )
        st.plotly_chart(charts.bar_chart(chart))
    else:
        st.caption("Choose a single 'Group by' column to see a chart.")


def analysis_top_n(df: pd.DataFrame) -> None:
    modes = (
        "Most common values in a column",
        "Top categories ranked by a number",
        "Rows with the largest values",
    )
    mode = st.radio("What do you want to find?", modes, key="top_mode")
    col_n, col_order = st.columns(2)
    n = int(col_n.number_input("N (how many)", 1, 100, 5, key="top_n"))
    largest = col_order.radio(
        "Order", ["Largest first", "Smallest first"], horizontal=True, key="top_order"
    ) == "Largest first"

    if mode == modes[2]:
        options = data_engine.numeric_columns(df)
        if not options:
            st.info("This dataset has no numeric columns.")
            return
        column = st.selectbox("Numeric column", options, key="top_rows_col")
        st.dataframe(data_engine.top_rows(df, column, n=n, largest=largest))
        return

    options = data_engine.categorical_columns(df) or list(df.columns)
    column = st.selectbox("Category column (e.g. product)", options, key="top_col")
    value_column, agg = None, "sum"
    if mode == modes[1]:
        numbers = data_engine.numeric_columns(df)
        if not numbers:
            st.info("This dataset has no numeric columns to rank by.")
            return
        col_val, col_agg = st.columns(2)
        value_column = col_val.selectbox("Rank by", numbers, key="top_val")
        agg = col_agg.selectbox("Using", data_engine.AGGREGATIONS, index=1, key="top_agg")

    table = data_engine.top_n(
        df, column, n=n, value_column=value_column, agg=agg, largest=largest
    )
    st.dataframe(table.round(2), hide_index=True)
    y = "count" if value_column is None else table.columns[-1]
    st.plotly_chart(
        charts.bar_chart(ChartData(table, x=column, y=y, title=f"Top {n}: {y} by {column}"))
    )


def analysis_trends(df: pd.DataFrame) -> None:
    dates = data_engine.date_columns(df)
    if not dates:
        st.info(
            "No date columns were detected. Dates should look like 2026-01-31 "
            "or 31/01/2026 to be recognised."
        )
        return

    col_date, col_freq = st.columns(2)
    date_column = col_date.selectbox("Date column", dates, key="trd_date")
    freq = col_freq.selectbox(
        "Group dates by", list(data_engine.TREND_FREQUENCIES), index=2, key="trd_freq"
    )
    col_val, col_agg = st.columns(2)
    value_column = col_val.selectbox(
        "Number to track", [NO_VALUE_COLUMN, *data_engine.numeric_columns(df)], key="trd_val"
    )
    agg = col_agg.selectbox(
        "Using",
        data_engine.AGGREGATIONS,
        index=1,
        key="trd_agg",
        disabled=value_column == NO_VALUE_COLUMN,
    )

    chart = data_engine.line_chart_data(
        df,
        date_column,
        value_column=None if value_column == NO_VALUE_COLUMN else value_column,
        freq=freq,
        agg=agg,
    )
    st.plotly_chart(charts.line_chart(chart))
    show_chart_note(chart)
    with st.expander("Show the numbers"):
        st.dataframe(chart.data, hide_index=True)


def analysis_correlations(df: pd.DataFrame) -> None:
    st.markdown(
        "Correlation measures how strongly two numbers move together, from **-1** "
        "(one goes up as the other goes down) through **0** (no relationship) to "
        "**+1** (they go up together). It does not prove that one causes the other."
    )
    options = data_engine.numeric_columns(df)
    if len(options) < 2:
        st.info("Correlations need at least two numeric columns.")
        return

    columns = st.multiselect("Numeric columns", options, default=options[:10], key="cor_cols")
    method = st.radio(
        "Method",
        data_engine.CORRELATION_METHODS,
        format_func=lambda m: {
            "pearson": "Pearson (straight-line relationships)",
            "spearman": "Spearman (any steadily rising/falling relationship)",
        }[m],
        horizontal=True,
        key="cor_method",
    )

    corr = data_engine.correlation_matrix(df, columns, method=method)
    st.plotly_chart(charts.correlation_heatmap(corr))
    pairs = data_engine.strongest_correlations(corr)
    if pairs.empty:
        st.info("Not enough overlapping values to calculate correlations.")
    else:
        st.markdown("**Strongest relationships**")
        st.dataframe(pairs.round(3), hide_index=True)


def analysis_distribution(df: pd.DataFrame) -> None:
    st.markdown("A histogram shows how often values fall into each range.")
    options = data_engine.numeric_columns(df)
    if not options:
        st.info("This dataset has no numeric columns.")
        return

    col_c, col_b = st.columns(2)
    column = col_c.selectbox("Numeric column", options, key="hist_col")
    bins = col_b.slider("Number of bins", 5, 100, 20, key="hist_bins")
    chart = data_engine.histogram_data(df, column, bins=bins)
    st.plotly_chart(charts.histogram_chart(chart))
    show_chart_note(chart)
    st.dataframe(data_engine.describe_numeric(df, [column]).round(2))


def show_analysis(df: pd.DataFrame) -> None:
    tool = st.radio("Choose an analysis", ANALYSIS_TOOLS, horizontal=True, key="analysis_tool")
    st.divider()
    handlers = {
        "Filter & sort": analysis_filter_sort,
        "Group & summarise": analysis_group,
        "Top N": analysis_top_n,
        "Trends over time": analysis_trends,
        "Correlations": analysis_correlations,
        "Distribution": analysis_distribution,
    }
    try:
        handlers[tool](df)
    except AnalysisError as exc:
        st.warning(str(exc))
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Something went wrong while running this analysis: {exc}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


def _use_source(source: str) -> None:
    # Button callback: runs before the page is redrawn, so it may change the sidebar choice.
    st.session_state[DATA_SOURCE_KEY] = source


def show_sidebar() -> tuple[tuple[str, pd.DataFrame] | None, str | None]:
    with st.sidebar:
        st.markdown("## 📊 AskYourData AI")
        st.caption("Profile, explore, chart and question your data, offline.")
        st.divider()
        st.markdown("**1 · Choose a data source**")
        source = st.radio(
            "Data source", DATA_SOURCES, key=DATA_SOURCE_KEY, captions=SOURCE_CAPTIONS,
            label_visibility="collapsed",
        )  # fmt: skip

    dataset, error = load_dataset(source)

    with st.sidebar:
        if source == DATABASE_SOURCE and dataset is None:
            st.info("Run a query in the 🗄️ **Database Explorer** tab, then click **Send to analysis**.")
        if dataset is not None:
            name, df = dataset
            with st.container(border=True):
                st.markdown(f"**Active dataset**  \n{name}")
                st.caption(f"{len(df):,} rows · {df.shape[1]} columns · {source}")
                if source == DATABASE_SOURCE:
                    st.caption("A snapshot sent from the Database Explorer.")
        st.divider()
        st.markdown("**2 · Explore it in the tabs**")
        st.caption(
            "📋 Summary → 💡 Insights → 🧭 Explore → 🔎 Analysis tools → 💬 Ask Your Data. "
            "Use 🗄️ Database Explorer to read from a SQL database."
        )
        with st.expander("About & privacy"):
            st.markdown(
                "- Analysis runs **locally** with pandas. No AI service, API key or internet "
                "connection is used.\n"
                "- Your data is never changed; every calculation works on a copy.\n"
                "- Database access is **read-only**: single SELECT queries on approved tables, "
                "with row limits and timeouts.\n"
                "- Questions are never run as code or SQL."
            )
    return dataset, error


def show_header(dataset: tuple[str, pd.DataFrame] | None) -> None:
    st.title("AskYourData AI")
    st.caption(
        "An offline data-analyst app: profile, explore and question CSV, Excel and SQL data. "
        "Every number is calculated with pandas; no AI service or API key is needed."
    )
    if dataset is not None:
        name, df = dataset
        st.markdown(f"**Active dataset:** {name} · {len(df):,} rows · {df.shape[1]} columns")


def show_welcome() -> None:
    """Onboarding shown while no dataset is selected."""
    with st.container(border=True):
        st.markdown("### 👋 Welcome")
        st.markdown(
            "AskYourData AI helps you understand a table of data in minutes: it checks data "
            "quality, summarises the key figures, draws charts and answers plain-English "
            "questions such as *\"Show net revenue by city for Electronics\"*."
        )
        steps = st.columns(3)
        steps[0].markdown(
            "**1. Choose data**  \nUpload a CSV/Excel file, pick a sample, or read from a "
            "database (read-only)."
        )
        steps[1].markdown(
            "**2. Explore**  \nSee the executive summary, column types, missing values, "
            "statistics and analysis tools."
        )
        steps[2].markdown(
            "**3. Ask**  \nAsk questions in plain English, with filters and calculated "
            "metrics. Download tables and charts."
        )
        st.markdown("**Start with a sample:**")
        left, right, _ = st.columns([1, 1, 1])
        left.button(
            "🛒 E-commerce orders", on_click=_use_source, args=(ECOMMERCE_SOURCE,),
            type="primary", width="stretch", key="welcome_ecommerce",
        )  # fmt: skip
        right.button(
            "📦 Sales sample", on_click=_use_source, args=(SAMPLE_SOURCE,),
            width="stretch", key="welcome_sales",
        )  # fmt: skip
        try:
            examples = suggest_questions(cached_load(ECOMMERCE_DATA_PATH.name, ECOMMERCE_DATA_PATH.read_bytes()))
        except Exception:
            examples = []
        if examples:
            with st.expander("Example questions for the e-commerce sample"):
                st.markdown("\n".join(f"- {q}" for q in examples))


def safe_section(render: Callable[..., None], *args) -> None:
    """Run one tab's content; show a friendly message instead of a traceback on failure."""
    try:
        render(*args)
    except AnalysisError as exc:
        st.warning(str(exc))
    except Exception as exc:  # last-resort guard
        st.error(f"This section could not be shown ({type(exc).__name__}). Try another dataset or tab.")


def show_explore(df: pd.DataFrame) -> None:
    show_overview(df)
    tabs = st.tabs(["Preview", "Columns & types", "Missing values", "Duplicates", "Statistics"])
    with tabs[0]:
        show_preview(df)
    with tabs[1]:
        show_columns(df)
    with tabs[2]:
        show_missing_values(df)
    with tabs[3]:
        show_duplicates(df)
    with tabs[4]:
        show_statistics(df)


def main() -> None:
    # Database settings come from a .env file next to app.py (if it exists).
    load_env_file(PROJECT_DIR / ".env")
    st.set_page_config(
        page_title="AskYourData AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "AskYourData AI: an offline data-analyst portfolio project built with "
            "Streamlit, pandas, Plotly and SQLAlchemy. No AI service or API key is used."
        },
    )

    dataset, error = show_sidebar()
    # Before any dataset-specific widget is drawn: forget settings and answers that
    # belong to a different dataset, so nothing stale is shown.
    fingerprint = ui_state.dataset_fingerprint(dataset[1]) if dataset else None
    ui_state.reset_if_dataset_changed(st.session_state, fingerprint)

    show_header(dataset)
    if error:
        st.error(error)
        st.caption("Check that the file is a valid, non-empty CSV or .xlsx file, then try again.")
    if dataset is None:
        show_welcome()

    summary_tab, insights_tab, explore_tab, analysis_tab, ask_tab, database_tab = st.tabs(TAB_LABELS)
    with summary_tab:
        if dataset is None:
            st.info(NO_DATASET_MESSAGE)
        else:
            safe_section(lambda: show_executive_summary(cached_summary(fingerprint, dataset[1])))
    with insights_tab:
        if dataset is None:
            st.info(NO_DATASET_MESSAGE)
        else:
            nonce = st.session_state.get(INSIGHTS_NONCE_KEY, 0)
            safe_section(lambda: show_insights(cached_insights(fingerprint, nonce, dataset[1]), _regenerate_insights))
    with explore_tab:
        if dataset is None:
            st.info(NO_DATASET_MESSAGE)
        else:
            safe_section(show_explore, dataset[1])
    with analysis_tab:
        if dataset is None:
            st.info(NO_DATASET_MESSAGE)
        else:
            st.caption("Point-and-click analysis tools. Results update as you change the options.")
            show_analysis(dataset[1])
    with ask_tab:
        safe_section(show_ask_your_data, dataset)
    with database_tab:
        safe_section(show_database_explorer, SAMPLE_DATA_PATH)


if __name__ == "__main__":
    main()
