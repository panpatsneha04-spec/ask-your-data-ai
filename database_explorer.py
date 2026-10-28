"""Streamlit UI for the Database Explorer tab.

All database work is done by ``askyourdata.database.SafeDatabase``; this file
only draws the controls and shows results.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from askyourdata.database import (
    BUILDER_OPERATORS,
    DEMO_ALLOWLIST,
    BuilderFilter,
    DatabaseError,
    QueryResult,
    SafeDatabase,
    create_demo_engine,
    create_mysql_engine,
)
from askyourdata.db_config import ConfigError, DatabaseSettings
from askyourdata.sql_safety import UnsafeQueryError

# Shared with app.py: the sidebar data-source selector and the dataset sent from here.
DATA_SOURCE_KEY = "data_source"
DATABASE_SOURCE = "Database query result"
SESSION_DATASET = "db_dataset"

SESSION_DB = "db_connection"
SESSION_VERSION = "db_server_version"
SESSION_SENT = "db_sent_from"
RESULT_KEYS = ("db_result_builder", "db_result_sql")

SECTIONS = ("Browse tables", "Query builder", "Custom SQL")
NO_SORT = "(no sorting)"
VALUELESS = ("is missing", "is not missing")
EXPECTED_ERRORS = (DatabaseError, UnsafeQueryError, ConfigError)

SECURITY_NOTE = """
**1. The database account is the real protection.** Connect with a MySQL user
that only has `SELECT` permission on the tables you want to explore (see the
README). Even if the app had a bug, the server would refuse any change.
The checks below are an *extra* layer and are **not a substitute** for this.

**2. Read-only checks in the app**
- Only single `SELECT` statements run. `INSERT`, `UPDATE`, `DELETE`, `DROP`,
  `ALTER`, `CREATE`, `TRUNCATE`, `INTO OUTFILE`, `FOR UPDATE`, `SLEEP()` and
  multiple statements are rejected.
- Only tables on the approved list (`DB_ALLOWED_TABLES`) can be read.
- Every result is capped by a row limit, and slow queries are stopped by a timeout.
- MySQL sessions are switched to read-only mode.
- Values typed into the query builder are sent as bound parameters, separately
  from the SQL.

**3. Secrets**: the password is never shown, logged or saved by the app.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def show_database_explorer(sample_csv: Path) -> None:
    st.markdown("Connect to a database, browse approved tables and run read-only queries.")
    with st.expander("🔒 How your database is protected"):
        st.markdown(SECURITY_NOTE)

    db: SafeDatabase | None = st.session_state.get(SESSION_DB)
    if db is None:
        show_connect_panel(sample_csv)
        return

    show_status_bar(db)
    section = st.radio("What do you want to do?", SECTIONS, horizontal=True, key="db_section")
    st.divider()
    try:
        if section == "Browse tables":
            show_browse_tables(db)
        elif section == "Query builder":
            show_query_builder(db)
        else:
            show_custom_sql(db)
    except EXPECTED_ERRORS as exc:
        st.warning(str(exc))


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------


def _connect(factory: Callable[[], SafeDatabase]) -> None:
    db = None
    try:
        with st.spinner("Connecting..."):
            db = factory()
            version = db.test_connection()
    except EXPECTED_ERRORS as exc:
        if db is not None:
            db.close()
        st.error(f"Could not connect. {exc}")
        return
    except Exception as exc:  # never show a raw traceback (it could contain settings)
        if db is not None:
            db.close()
        st.error(f"Could not connect ({type(exc).__name__}).")
        return

    st.session_state[SESSION_DB] = db
    st.session_state[SESSION_VERSION] = version
    st.rerun()


def show_connect_panel(sample_csv: Path) -> None:
    mode = st.radio(
        "Connect to",
        ["MySQL server", "Demo database (no setup needed)"],
        horizontal=True,
        key="db_mode",
    )

    if mode != "MySQL server":
        st.markdown(
            "A temporary SQLite database built from the sample sales data. It has an "
            "approved `sales` table and a `staff_salaries` table that is **not** "
            "approved, so you can see the table allowlist in action."
        )
        if st.button("Open demo database", type="primary"):
            _connect(
                lambda: SafeDatabase(
                    create_demo_engine(sample_csv), DEMO_ALLOWLIST, label="Demo SQLite database"
                )
            )
        return

    try:
        settings = DatabaseSettings.from_env()
    except ConfigError as exc:
        st.error(f"There is a problem in your .env settings: {exc}")
        settings = DatabaseSettings()

    if settings.allowed_tables.is_empty:
        st.warning(
            "No tables are approved yet. Add `DB_ALLOWED_TABLES=table1,table2` to your "
            "`.env` file and restart the app (see `.env.example`)."
        )
    else:
        st.caption(
            f"Approved tables (from DB_ALLOWED_TABLES): **{settings.allowed_tables.describe()}** · "
            f"row limit {settings.max_rows:,} · timeout {settings.timeout_seconds}s"
        )

    with st.form("db_connect_form"):
        col_host, col_port = st.columns([3, 1])
        host = col_host.text_input("Host", settings.host)
        port = col_port.number_input("Port", 1, 65535, settings.port)
        database = st.text_input("Database name", settings.database)
        user = st.text_input("User (use a read-only account)", settings.user)
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Using the password from .env" if settings.has_password else "",
            help="Leave blank to use DB_PASSWORD from .env. "
            "The app never shows, logs or saves the password.",
        )
        submitted = st.form_submit_button("Connect", type="primary")

    if submitted:
        final = settings.with_overrides(
            host=host.strip(),
            port=int(port),
            database=database.strip(),
            user=user.strip(),
            password=password or settings.password,
        )
        _connect(
            lambda: SafeDatabase(
                create_mysql_engine(final),
                final.allowed_tables,
                final.max_rows,
                label=final.describe(),
                allow_custom_sql=final.allow_custom_sql,
                secret=final.password,
            )
        )


def show_status_bar(db: SafeDatabase) -> None:
    col_status, col_button = st.columns([5, 1])
    col_status.success(
        f"✅ Connected to **{db.label}** ({st.session_state.get(SESSION_VERSION, '')}) · "
        f"row limit {db.max_rows:,}"
    )
    if col_button.button("Disconnect"):
        db.close()
        for key in (SESSION_DB, SESSION_VERSION, SESSION_SENT, *RESULT_KEYS):
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Results and "Send to analysis"
# ---------------------------------------------------------------------------


def _send_to_analysis(name: str, data: pd.DataFrame, origin: str) -> None:
    # Runs as a button callback, before the page is redrawn, so it may change
    # the sidebar's data-source selection.
    st.session_state[SESSION_DATASET] = (name, data)
    st.session_state[DATA_SOURCE_KEY] = DATABASE_SOURCE
    st.session_state[SESSION_SENT] = origin


def show_result(result: QueryResult, name: str, origin: str) -> None:
    summary = f"**{len(result.data):,} row(s)**"
    if result.maybe_truncated:
        summary += f" (the row limit of {result.row_limit:,} was reached, so there may be more)"
    st.markdown(summary)
    st.dataframe(result.data, hide_index=True)
    with st.expander("SQL sent to the database"):
        st.code(result.sql, language="sql")

    if result.data.empty:
        return
    st.button(
        "📊 Send to analysis",
        key=f"db_send_{origin}",
        on_click=_send_to_analysis,
        args=(name, result.data, origin),
        help="Use these rows in the Analyse data tab (profiling, charts, filters...).",
    )
    if st.session_state.get(SESSION_SENT) == origin:
        st.success("Sent! Open the **📊 Analyse data** tab to explore these rows.")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _choose_approved_table(db: SafeDatabase, key: str) -> str | None:
    approved = db.approved_tables()
    if not approved:
        st.warning(
            "None of the tables this user can see are approved. "
            "Add them to DB_ALLOWED_TABLES in your .env file."
        )
        return None
    return st.selectbox("Approved table", approved, key=key)


def show_browse_tables(db: SafeDatabase) -> None:
    st.markdown("**Tables this database user can see**")
    st.dataframe(db.list_tables(), hide_index=True)

    table = _choose_approved_table(db, "db_browse_table")
    if table is None:
        return
    st.markdown("**Columns and data types**")
    st.dataframe(db.describe_table(table), hide_index=True)

    rows = st.number_input(
        "Rows to preview", 1, db.max_rows, min(20, db.max_rows), key="db_preview_rows"
    )
    show_result(db.preview(table, int(rows)), f"{table} (preview)", origin="browse")


def show_query_builder(db: SafeDatabase) -> None:
    st.markdown(
        "Build a query by choosing options - no SQL needed. Values you type are sent "
        "to the database as **bound parameters** (placeholders such as `?` or "
        "`%(param_1)s` in the SQL), so they can never change the query itself."
    )
    table = _choose_approved_table(db, "db_qb_table")
    if table is None:
        return
    names = db.describe_table(table)["column"].tolist()

    columns = st.multiselect("Columns (leave empty for all)", names, key="db_qb_cols")
    n_filters = int(st.number_input("Number of filters", 0, 3, 0, key="db_qb_n"))
    filters = []
    for i in range(n_filters):
        col_c, col_op, col_v = st.columns([2, 2, 3])
        column = col_c.selectbox("Column", names, key=f"db_qb_col_{i}")
        op = col_op.selectbox("Condition", BUILDER_OPERATORS, key=f"db_qb_op_{i}")
        value = None if op in VALUELESS else col_v.text_input("Value", key=f"db_qb_val_{i}")
        filters.append(BuilderFilter(column, op, value))

    col_sort, col_desc, col_limit = st.columns(3)
    order_by = col_sort.selectbox("Sort by", [NO_SORT, *names], key="db_qb_sort")
    descending = col_desc.toggle("Descending", key="db_qb_desc")
    limit = col_limit.number_input("Max rows", 1, db.max_rows, db.max_rows, key="db_qb_limit")

    if st.button("Run query", type="primary", key="db_qb_run"):
        result = db.build_query(
            table,
            columns or None,
            filters,
            order_by=None if order_by == NO_SORT else order_by,
            descending=descending,
            limit=int(limit),
        )
        st.session_state["db_result_builder"] = (f"{table} (query)", result)

    stored = st.session_state.get("db_result_builder")
    if stored:
        show_result(stored[1], stored[0], origin="builder")


def show_custom_sql(db: SafeDatabase) -> None:
    if not db.allow_custom_sql:
        st.info("Custom SQL is turned off (DB_ALLOW_CUSTOM_SQL=false). Use the query builder.")
        return

    approved = db.approved_tables()
    st.markdown(
        f"Write **one read-only SELECT** query. Approved tables: "
        f"`{', '.join(approved) or 'none'}`. At most **{db.max_rows:,}** rows are "
        "returned - a LIMIT is added automatically."
    )
    example = f"SELECT *\nFROM {approved[0]}" if approved else "SELECT 1"
    sql = st.text_area("SQL query", value=example, height=160, key="db_sql_text")

    if st.button("Run query", type="primary", key="db_sql_run"):
        st.session_state["db_result_sql"] = ("Custom SQL query", db.run_select(sql))

    stored = st.session_state.get("db_result_sql")
    if stored:
        show_result(stored[1], stored[0], origin="sql")
