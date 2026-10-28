"""Safe, read-only access to a SQL database (MySQL in the app, SQLite in tests).

All database access in the app goes through ``SafeDatabase``. It only offers
read operations, and every one of them checks table names against the
allowlist and applies a row limit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from askyourdata.db_config import DatabaseSettings
from askyourdata.sql_safety import (
    DEFAULT_ROW_LIMIT,
    TableAllowlist,
    UnsafeQueryError,
    allowed_subset,
    prepare_select,
    validate_identifier,
    validate_row_limit,
)

# Only events (connected, query returned N rows) are logged - never
# passwords, connection URLs or query results.
logger = logging.getLogger(__name__)

DEMO_ALLOWLIST = TableAllowlist.of("sales")


class DatabaseError(Exception):
    """A database problem explained for users. Never contains the password."""


# Well-known MySQL error numbers -> friendlier explanations.
_MYSQL_ERRORS = {
    1044: "This user has no access to that database.",
    1045: "Login failed: check the user name and password.",
    1049: "That database does not exist on the server.",
    1142: "The database refused this command for this user (permissions).",
    1146: "That table does not exist.",
    1792: "The database refused this because the session is read-only.",
    2003: "Could not reach the MySQL server. Is it running, and are the host and port right?",
    2005: "Unknown MySQL host name.",
    2013: "The connection was lost (the query may have taken too long).",
    3024: "The query took longer than the configured timeout and was stopped.",
}


def _friendly_error(exc: Exception, secret: str = "") -> DatabaseError:
    original = getattr(exc, "orig", None) or exc
    args = getattr(original, "args", ())
    code = args[0] if args and isinstance(args[0], int) else None
    detail = str(args[1] if code is not None and len(args) > 1 else original).splitlines()[0]
    if secret:
        detail = detail.replace(secret, "***")
    detail = detail[:300]
    if code in _MYSQL_ERRORS:
        return DatabaseError(f"{_MYSQL_ERRORS[code]} (MySQL error {code}: {detail})")
    return DatabaseError(f"Database error: {detail}")


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def _session_guards(timeout_seconds: int) -> Callable[[Any, Any], None]:
    """Settings applied to every new MySQL connection (defence in depth)."""
    statements = [
        # The server itself refuses writes in this session.
        ("SET SESSION TRANSACTION READ ONLY",),
        # Stop long SELECTs: MySQL uses milliseconds, MariaDB uses seconds.
        (
            f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_seconds) * 1000}",
            f"SET SESSION max_statement_time = {int(timeout_seconds)}",
        ),
    ]

    def on_connect(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            for alternatives in statements:
                for statement in alternatives:
                    try:
                        cursor.execute(statement)
                        break
                    except Exception:  # not supported by this server version
                        continue
                else:
                    logger.warning("Session safety setting not supported: %s", alternatives[0])
        finally:
            cursor.close()

    return on_connect


def create_mysql_engine(settings: DatabaseSettings) -> Engine:
    """Create (but do not yet open) a connection pool to MySQL."""
    connect_args: dict[str, Any] = {
        "connect_timeout": 10,
        # Backstop for servers without MAX_EXECUTION_TIME support.
        "read_timeout": settings.timeout_seconds + 5,
        "write_timeout": settings.timeout_seconds + 5,
    }
    if settings.ssl_ca:
        connect_args["ssl"] = {"ca": settings.ssl_ca}

    engine = sa.create_engine(
        settings.url(),
        connect_args=connect_args,
        pool_pre_ping=True,  # replace dropped connections automatically
        pool_recycle=1800,
        hide_parameters=True,  # keep bound values out of error messages and logs
    )
    sa.event.listen(engine, "connect", _session_guards(settings.timeout_seconds))
    return engine


def create_demo_engine(csv_path: str | Path) -> Engine:
    """An in-memory SQLite database with the sample data, for trying the app.

    It contains an approved ``sales`` table and a ``staff_salaries`` table that
    is deliberately NOT on the allowlist, to show the allowlist in action.
    """
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    pd.read_csv(csv_path).to_sql("sales", engine, index=False)
    pd.DataFrame(
        {"employee": ["Ana", "Ben", "Chen"], "salary": [52000, 61000, 58000]}
    ).to_sql("staff_salaries", engine, index=False)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")  # SQLite's read-only switch
    return engine


# ---------------------------------------------------------------------------
# SafeDatabase
# ---------------------------------------------------------------------------

BUILDER_OPERATORS = ("=", "!=", ">", ">=", "<", "<=", "contains", "is missing", "is not missing")


@dataclass(frozen=True)
class QueryResult:
    data: pd.DataFrame
    sql: str  # the SQL that was actually sent (values shown as placeholders)
    row_limit: int

    @property
    def maybe_truncated(self) -> bool:
        """True when the row limit was reached, so more rows may exist."""
        return len(self.data) >= self.row_limit


@dataclass(frozen=True)
class BuilderFilter:
    column: str
    operator: str
    value: Any = None


def _unique_names(names: list) -> list[str]:
    """Make column names unique, e.g. two "id" columns from a JOIN -> id, id_2."""
    seen: dict[str, int] = {}
    unique = []
    for name in map(str, names):
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return unique


def _tidy_types(df: pd.DataFrame) -> pd.DataFrame:
    """MySQL DECIMAL values arrive as Python Decimals; turn them into numbers."""
    for column in df.columns:
        if df[column].dtype == object:
            values = df[column].dropna()
            if not values.empty and all(isinstance(v, Decimal) for v in values):
                df[column] = pd.to_numeric(df[column])
    return df


class SafeDatabase:
    """Read-only, allowlisted access to one database."""

    def __init__(
        self,
        engine: Engine,
        allowlist: TableAllowlist,
        max_rows: int = DEFAULT_ROW_LIMIT,
        *,
        label: str = "database",
        allow_custom_sql: bool = True,
        secret: str = "",
    ) -> None:
        self.engine = engine
        self.allowlist = allowlist
        self.max_rows = validate_row_limit(max_rows)
        self.label = label
        self.allow_custom_sql = allow_custom_sql
        self._secret = secret  # only used to scrub error messages
        self.sql_dialect = {"mysql": "mysql", "mariadb": "mysql", "sqlite": "sqlite"}.get(
            engine.dialect.name
        )

    def __repr__(self) -> str:
        return f"SafeDatabase({self.label!r})"

    # -- helpers --------------------------------------------------------------

    def _run(self, work: Callable[[Connection], Any]) -> Any:
        try:
            with self.engine.connect() as conn:
                return work(conn)
        except UnsafeQueryError:
            raise
        except (SQLAlchemyError, DBAPIError) as exc:
            raise _friendly_error(exc, self._secret) from None

    def _row_limit(self, limit: int | None) -> int:
        limit = self.max_rows if limit is None else validate_row_limit(limit)
        if limit > self.max_rows:
            raise UnsafeQueryError(f"The row limit for this connection is {self.max_rows:,}.")
        return limit

    def _all_table_names(self) -> list[tuple[str, str]]:
        def work(conn: Connection) -> list[tuple[str, str]]:
            inspector = sa.inspect(conn)
            tables = [(name, "table") for name in inspector.get_table_names()]
            return tables + [(name, "view") for name in inspector.get_view_names()]

        return self._run(work)

    def _resolve_table(self, name: str) -> str:
        """Check a requested table and return its exact name in the database."""
        validate_identifier(name)
        if not self.allowlist.allows(name):
            raise UnsafeQueryError(f"Table '{name}' is not on the approved list.")
        for actual, _kind in self._all_table_names():
            if actual.casefold() == name.casefold():
                return actual
        raise DatabaseError(f"Table '{name}' was not found in the database.")

    def _fetch(self, statement: Any, limit: int, display_sql: str) -> QueryResult:
        def work(conn: Connection) -> pd.DataFrame:
            if isinstance(statement, str):
                # no_parameters: send the text as-is, so "%" or ":" inside
                # string literals are not mistaken for placeholders.
                result = conn.exec_driver_sql(
                    statement, execution_options={"no_parameters": True}
                )
            else:
                result = conn.execute(statement)
            rows = result.fetchmany(limit)  # never more than the limit
            columns = _unique_names(list(result.keys()))
            return pd.DataFrame([tuple(r) for r in rows], columns=columns)

        data = _tidy_types(self._run(work))
        logger.info("Query on %s returned %d row(s)", self.label, len(data))
        return QueryResult(data=data, sql=display_sql, row_limit=limit)

    # -- public, read-only operations -------------------------------------------

    def test_connection(self) -> str:
        """Open a connection and return a short server description."""
        self._run(lambda conn: conn.exec_driver_sql("SELECT 1").scalar())
        version = self.engine.dialect.server_version_info or ()
        return f"{self.engine.dialect.name} {'.'.join(map(str, version))}".strip()

    def list_tables(self) -> pd.DataFrame:
        """All tables/views this database user can see, and whether each is approved."""
        names = self._all_table_names()
        return pd.DataFrame(
            {
                "table": [n for n, _ in names],
                "type": [k for _, k in names],
                "approved": [self.allowlist.allows(n) for n, _ in names],
            },
        ).sort_values(["approved", "table"], ascending=[False, True], ignore_index=True)

    def approved_tables(self) -> list[str]:
        return allowed_subset((n for n, _ in self._all_table_names()), self.allowlist)

    def describe_table(self, name: str) -> pd.DataFrame:
        """Column names, SQL data types and whether NULL is allowed."""
        table = self._resolve_table(name)
        columns = self._run(lambda conn: sa.inspect(conn).get_columns(table))
        return pd.DataFrame(
            {
                "column": [c["name"] for c in columns],
                "type": [str(c["type"]) for c in columns],
                "nullable": [c.get("nullable", True) for c in columns],
            }
        )

    def preview(self, name: str, limit: int | None = None) -> QueryResult:
        """The first rows of an approved table."""
        table = self._resolve_table(name)
        limit = self._row_limit(limit)
        # sa.table() quotes the name correctly; the LIMIT value is a bound parameter.
        statement = sa.select(sa.text("*")).select_from(sa.table(table)).limit(limit)
        return self._fetch(statement, limit, str(statement.compile(self.engine)))

    def build_query(
        self,
        table: str,
        columns: Sequence[str] | None = None,
        filters: Sequence[BuilderFilter] = (),
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> QueryResult:
        """A SELECT built from choices instead of typed SQL.

        Table and column names are checked against the real table; every
        filter VALUE is sent as a bound parameter, never pasted into the SQL.
        """
        name = self._resolve_table(table)
        limit = self._row_limit(limit)
        reflected = self._run(lambda conn: sa.Table(name, sa.MetaData(), autoload_with=conn))

        def column(col_name: str) -> sa.Column:
            validate_identifier(col_name, "Column name")
            if col_name not in reflected.c:
                raise UnsafeQueryError(f"Column '{col_name}' does not exist in '{name}'.")
            return reflected.c[col_name]

        selected = [column(c) for c in columns] if columns else [reflected]
        statement = sa.select(*selected).limit(limit)
        for f in filters:
            statement = statement.where(self._filter_clause(column(f.column), f))
        if order_by:
            sort_col = column(order_by)
            statement = statement.order_by(sort_col.desc() if descending else sort_col.asc())

        return self._fetch(statement, limit, str(statement.compile(self.engine)))

    @staticmethod
    def _filter_clause(col: sa.Column, f: BuilderFilter) -> Any:
        if f.operator not in BUILDER_OPERATORS:
            raise UnsafeQueryError(f"Unknown filter operator '{f.operator}'.")
        if f.operator == "is missing":
            return col.is_(None)
        if f.operator == "is not missing":
            return col.is_not(None)
        if f.value is None or (isinstance(f.value, str) and not f.value.strip()):
            raise UnsafeQueryError(f"Please enter a value for '{col.name} {f.operator}'.")
        if f.operator == "contains":
            # autoescape: % and _ typed by the user are matched literally
            return col.contains(str(f.value), autoescape=True)

        value: Any = f.value.strip() if isinstance(f.value, str) else f.value
        try:
            python_type = col.type.python_type
        except NotImplementedError:
            python_type = None
        if python_type in (int, float, Decimal) and isinstance(value, str):
            try:
                value = python_type(value.replace(",", ""))
            except (ValueError, ArithmeticError):
                raise UnsafeQueryError(
                    f"Column '{col.name}' contains numbers, so the value must be a number."
                ) from None
        comparisons = {
            "=": col.__eq__, "!=": col.__ne__, ">": col.__gt__,
            ">=": col.__ge__, "<": col.__lt__, "<=": col.__le__,
        }  # fmt: skip
        return comparisons[f.operator](value)

    def run_select(self, sql: str, limit: int | None = None) -> QueryResult:
        """Validate and run a custom SELECT (see ``sql_safety.prepare_select``)."""
        if not self.allow_custom_sql:
            raise UnsafeQueryError("Custom SQL is turned off (DB_ALLOW_CUSTOM_SQL=false).")
        limit = self._row_limit(limit)
        safe_sql = prepare_select(sql, self.allowlist, limit, self.sql_dialect)
        return self._fetch(safe_sql, limit, safe_sql)

    def close(self) -> None:
        self.engine.dispose()
