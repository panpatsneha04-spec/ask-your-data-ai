"""Database tests using in-memory SQLite, so no MySQL server is needed."""

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from askyourdata.database import (
    DEMO_ALLOWLIST,
    BuilderFilter,
    DatabaseError,
    SafeDatabase,
    _friendly_error,
    _session_guards,
    _tidy_types,
    create_demo_engine,
)
from askyourdata.sql_safety import TableAllowlist, UnsafeQueryError

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_sales.csv"
SECRET = "pw-12345"


@pytest.fixture
def engine():
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    pd.DataFrame(
        {
            "id": range(1, 61),
            "product": (["Desk", "Chair", "Lamp", None] * 15),
            "units": [i % 7 for i in range(1, 61)],
            "price": [10.5] * 60,
        }
    ).to_sql("sales", engine, index=False)
    pd.DataFrame({"secret": ["top secret"]}).to_sql("secrets", engine, index=False)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE VIEW sales_summary AS SELECT product, SUM(units) AS units "
            "FROM sales GROUP BY product"
        )
    return engine


@pytest.fixture
def db(engine):
    allowlist = TableAllowlist.of("sales", "sales_summary", "ghost")
    return SafeDatabase(engine, allowlist, max_rows=50, label="test db", secret=SECRET)


def count_rows(engine, table):
    with engine.connect() as conn:
        return conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()


# --- connecting and browsing ----------------------------------------------------


def test_connection(db):
    assert db.test_connection().startswith("sqlite")


def test_list_tables_marks_approved(db):
    tables = db.list_tables().set_index("table")
    assert tables.loc["sales", "approved"]
    assert not tables.loc["secrets", "approved"]
    assert tables.loc["sales_summary", "type"] == "view"


def test_approved_tables(db):
    assert db.approved_tables() == ["sales", "sales_summary"]  # "ghost" does not exist


def test_describe_table_ignores_case(db):
    columns = db.describe_table("SALES")
    assert columns["column"].tolist() == ["id", "product", "units", "price"]
    assert set(columns.columns) == {"column", "type", "nullable"}


@pytest.mark.parametrize(
    ("table", "error", "message"),
    [
        ("secrets", UnsafeQueryError, "not on the approved list"),
        ("ghost", DatabaseError, "not found"),
        ("sales; DROP TABLE secrets", UnsafeQueryError, "not valid"),
        ("", UnsafeQueryError, "not valid"),
    ],
)
def test_describe_table_rejects(db, table, error, message):
    with pytest.raises(error, match=message):
        db.describe_table(table)


def test_preview_respects_limits(db):
    assert len(db.preview("sales", 3).data) == 3
    result = db.preview("sales")
    assert len(result.data) == 50  # connection limit, not all 60 rows
    assert result.maybe_truncated
    with pytest.raises(UnsafeQueryError, match="row limit for this connection"):
        db.preview("sales", 51)
    with pytest.raises(UnsafeQueryError, match="row limit"):
        db.preview("sales", 0)


def test_preview_of_unapproved_table_is_refused(db):
    with pytest.raises(UnsafeQueryError):
        db.preview("secrets")


# --- custom SELECT queries ---------------------------------------------------------


def test_run_select(db):
    result = db.run_select(
        "SELECT product, SUM(units) AS total FROM sales WHERE product IS NOT NULL "
        "GROUP BY product ORDER BY product"
    )
    assert result.data["product"].tolist() == ["Chair", "Desk", "Lamp"]
    assert "LIMIT 50" in result.sql


def test_run_select_enforces_row_limit(db):
    assert len(db.run_select("SELECT * FROM sales").data) == 50
    assert len(db.run_select("SELECT * FROM sales LIMIT 1000").data) == 50
    assert len(db.run_select("SELECT * FROM sales", limit=7).data) == 7


def test_run_select_with_percent_and_colon_in_text(db):
    result = db.run_select("SELECT * FROM sales WHERE product LIKE 'De%' OR product = 'a:b'")
    assert set(result.data["product"]) == {"Desk"}


def test_run_select_on_view(db):
    assert len(db.run_select("SELECT * FROM sales_summary").data) == 4


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sales",
        "DROP TABLE sales",
        "SELECT 1; DELETE FROM sales",
        "UPDATE sales SET units = 0",
        "SELECT * FROM secrets",
        "SELECT * FROM sales WHERE id IN (SELECT 1 FROM secrets)",
    ],
)
def test_unsafe_queries_are_refused_and_change_nothing(db, engine, sql):
    with pytest.raises(UnsafeQueryError):
        db.run_select(sql)
    assert count_rows(engine, "sales") == 60
    assert count_rows(engine, "secrets") == 1


def test_database_errors_are_friendly(db):
    with pytest.raises(DatabaseError, match="Database error: no such column"):
        db.run_select("SELECT no_such_column FROM sales")


def test_error_messages_hide_the_password():
    error = _friendly_error(Exception(f"login failed for password {SECRET}"), SECRET)
    assert SECRET not in str(error)
    assert "***" in str(error)


def test_known_mysql_errors_are_explained():
    class FakeMySQLError(Exception):
        pass

    error = _friendly_error(FakeMySQLError(1045, "Access denied for user 'x'"))
    assert "check the user name and password" in str(error)


def test_custom_sql_can_be_turned_off(engine):
    db = SafeDatabase(engine, TableAllowlist.of("sales"), allow_custom_sql=False)
    with pytest.raises(UnsafeQueryError, match="turned off"):
        db.run_select("SELECT * FROM sales")


def test_duplicate_column_names_are_made_unique(db):
    result = db.run_select("SELECT id, id FROM sales")
    assert list(result.data.columns) == ["id", "id_2"]


# --- query builder (bound parameters) -------------------------------------------------


def test_builder_filters_sorts_and_limits(db):
    result = db.build_query(
        "sales",
        columns=["id", "units"],
        filters=[BuilderFilter("units", ">=", "5")],  # text is converted to a number
        order_by="id",
        descending=True,
        limit=3,
    )
    assert list(result.data.columns) == ["id", "units"]
    assert result.data["id"].tolist() == [55, 54, 48]
    assert (result.data["units"] >= 5).all()


def test_builder_values_are_bound_not_pasted(db, engine):
    injection = "x' OR '1'='1"
    result = db.build_query("sales", filters=[BuilderFilter("product", "=", injection)])
    assert result.data.empty  # treated as a plain value, matches nothing
    assert injection not in result.sql
    assert "?" in result.sql  # SQLite's placeholder
    assert count_rows(engine, "secrets") == 1


def test_builder_contains_matches_wildcards_literally(db):
    assert db.build_query("sales", filters=[BuilderFilter("product", "contains", "%")]).data.empty
    found = db.build_query("sales", filters=[BuilderFilter("product", "contains", "es")])
    assert set(found.data["product"]) == {"Desk"}


def test_builder_missing_value_filters(db):
    missing = db.build_query("sales", filters=[BuilderFilter("product", "is missing")])
    assert len(missing.data) == 15


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"table": "secrets"}, "approved list"),
        ({"table": "sales", "columns": ["nope"]}, "does not exist"),
        ({"table": "sales", "columns": ["id; DROP"]}, "not valid"),
        ({"table": "sales", "order_by": "nope"}, "does not exist"),
        ({"table": "sales", "filters": [BuilderFilter("units", "LIKE", "1")]}, "Unknown filter"),
        ({"table": "sales", "filters": [BuilderFilter("units", ">", " ")]}, "enter a value"),
        ({"table": "sales", "filters": [BuilderFilter("units", ">", "lots")]}, "must be a number"),
        ({"table": "sales", "limit": 1000}, "row limit"),
    ],
)
def test_builder_rejects_invalid_input(db, kwargs, message):
    with pytest.raises(UnsafeQueryError, match=message):
        db.build_query(**kwargs)


# --- demo database and MySQL session safety --------------------------------------------


def test_demo_database():
    demo = SafeDatabase(create_demo_engine(SAMPLE_CSV), DEMO_ALLOWLIST)
    assert demo.approved_tables() == ["sales"]
    assert len(demo.preview("sales").data) == 30
    with pytest.raises(UnsafeQueryError):
        demo.preview("staff_salaries")


def test_demo_database_is_read_only_even_without_the_app_checks():
    engine = create_demo_engine(SAMPLE_CSV)
    with pytest.raises(OperationalError, match="readonly"):
        with engine.connect() as conn:
            conn.exec_driver_sql("DELETE FROM sales")


def test_session_guards_fall_back_when_a_setting_is_unsupported():
    executed = []

    class FakeCursor:
        def execute(self, statement):
            if "MAX_EXECUTION_TIME" in statement:
                raise RuntimeError("unknown variable")  # e.g. MariaDB
            executed.append(statement)

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    _session_guards(timeout_seconds=20)(FakeConnection(), None)
    assert executed == [
        "SET SESSION TRANSACTION READ ONLY",
        "SET SESSION max_statement_time = 20",
    ]


def test_decimal_columns_become_numbers():
    df = _tidy_types(pd.DataFrame({"amount": [Decimal("1.50"), None], "name": ["a", "b"]}))
    assert pd.api.types.is_float_dtype(df["amount"])
    assert df["name"].tolist() == ["a", "b"]  # text columns are left alone
