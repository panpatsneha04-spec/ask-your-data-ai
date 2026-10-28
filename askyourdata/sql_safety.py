"""Checks that make sure only safe, read-only SELECT queries reach a database.

These checks are a second line of defence. The FIRST line must always be the
database itself: connect with an account that only has SELECT permission on
the approved tables. Application checks can have bugs; database permissions
are enforced by the server no matter what SQL arrives.

How a custom query is checked (``prepare_select``):
1. Basic checks: not empty, not too long, no MySQL "executable comments".
2. Every word (token) is scanned for forbidden keywords such as DROP or INTO
   and dangerous functions such as SLEEP.
3. The SQL is parsed into a tree with sqlglot. It must be exactly ONE
   statement, and that statement must be a SELECT (or SELECT ... UNION ...).
4. The tree is searched for any write / lock / export operation.
5. Every table it reads must be on the allowlist (no "other_db.table").
6. A LIMIT is added or lowered so at most ``max_rows`` rows come back.
7. The SQL is re-generated from the checked tree WITHOUT comments, and only
   that re-generated text is sent to the database.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ParseError, TokenError

DEFAULT_ROW_LIMIT = 500
MAX_ROW_LIMIT = 10_000
MAX_QUERY_LENGTH = 10_000

# Letters, digits, underscore and $, not starting with a digit, max 64 chars
# (MySQL's limit). Anything else (spaces, quotes, dots, ;) is rejected.
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,63}")

FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        "GRANT", "REVOKE", "RENAME", "MERGE", "CALL", "EXECUTE", "PREPARE",
        "DEALLOCATE", "HANDLER", "LOCK", "UNLOCK", "SET", "INTO", "OUTFILE",
        "DUMPFILE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "KILL", "SHUTDOWN",
    }
)  # fmt: skip

# Functions that can pause the server, read files or take locks.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "SLEEP", "BENCHMARK", "LOAD_FILE", "GET_LOCK", "RELEASE_LOCK",
        "RELEASE_ALL_LOCKS", "IS_FREE_LOCK", "IS_USED_LOCK", "MASTER_POS_WAIT",
        "SOURCE_POS_WAIT", "WAIT_FOR_EXECUTED_GTID_SET", "LOAD_EXTENSION",
        "READFILE", "WRITEFILE", "FTS3_TOKENIZER",
    }
)  # fmt: skip

# Parts of a parsed query that write, lock, export or change settings.
_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "TruncateTable",
        "Merge", "Command", "Into", "Lock", "Set", "Pragma", "Use", "Grant",
        "Revoke", "Transaction", "Commit", "Rollback", "LoadData", "Copy", "Kill",
    )
    if hasattr(exp, name)
)  # fmt: skip

# Tokens that are data, not SQL words: 'text', "text", `name`.
_LITERAL_TOKEN_TYPES = frozenset(
    {
        "STRING", "IDENTIFIER", "NATIONAL_STRING", "RAW_STRING", "BIT_STRING",
        "HEX_STRING", "BYTE_STRING", "HEREDOC_STRING", "UNICODE_STRING",
    }
)  # fmt: skip

_QUERY_TYPES = (exp.Select, exp.SetOperation)


class UnsafeQueryError(ValueError):
    """A request was refused for safety reasons. The message is user-facing."""


# ---------------------------------------------------------------------------
# Identifiers and the table allowlist
# ---------------------------------------------------------------------------


def validate_identifier(name: object, what: str = "Table name") -> str:
    """Return ``name`` if it is a plain SQL identifier, otherwise raise.

    Bound parameters protect VALUES (e.g. WHERE city = :city) but can never be
    used for table or column names, so names are checked against this strict
    pattern and against the allowlist instead.
    """
    if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name):
        raise UnsafeQueryError(
            f"{what} {name!r} is not valid. Use only letters, digits and underscores "
            "(max 64 characters, not starting with a digit)."
        )
    return name


@dataclass(frozen=True)
class TableAllowlist:
    """The tables users may read. Matching ignores upper/lower case."""

    names: frozenset[str] = frozenset()
    allow_all: bool = False

    @classmethod
    def of(cls, *names: str) -> TableAllowlist:
        return cls(frozenset(validate_identifier(n).casefold() for n in names))

    @classmethod
    def from_string(cls, raw: str | None) -> TableAllowlist:
        """Parse "orders, customers" (or "*" for every table) from settings."""
        parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
        if parts == ["*"]:
            return cls(allow_all=True)
        if "*" in parts:
            raise UnsafeQueryError("Use either '*' or a list of table names, not both.")
        return cls.of(*parts)

    def allows(self, name: str) -> bool:
        if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name):
            return False
        return self.allow_all or name.casefold() in self.names

    @property
    def is_empty(self) -> bool:
        return not self.allow_all and not self.names

    def describe(self) -> str:
        if self.allow_all:
            return "all tables in the database (*)"
        return ", ".join(sorted(self.names)) or "none"


# ---------------------------------------------------------------------------
# Custom SELECT validation
# ---------------------------------------------------------------------------


def validate_row_limit(max_rows: object) -> int:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise UnsafeQueryError(f"The row limit must be a whole number (got {max_rows!r}).")
    if not 1 <= max_rows <= MAX_ROW_LIMIT:
        raise UnsafeQueryError(f"The row limit must be between 1 and {MAX_ROW_LIMIT:,}.")
    return max_rows


def _check_tokens(sql: str, dialect: str | None) -> None:
    try:
        tokens = Dialect.get_or_raise(dialect).tokenize(sql)
    except (TokenError, ParseError, ValueError) as exc:
        raise UnsafeQueryError(f"The SQL could not be read: {exc}") from None

    for token in tokens:
        if token.token_type.name in _LITERAL_TOKEN_TYPES:
            continue
        word = token.text.upper()
        if word in FORBIDDEN_KEYWORDS:
            raise UnsafeQueryError(
                f"'{word}' is not allowed. Only read-only SELECT queries can be run."
            )
        if word in FORBIDDEN_FUNCTIONS:
            raise UnsafeQueryError(f"The function {word}() is not allowed.")


def _check_tables(query: exp.Expression, allowlist: TableAllowlist) -> None:
    cte_names = {cte.alias_or_name.casefold() for cte in query.find_all(exp.CTE)}
    for table in query.find_all(exp.Table):
        name = table.name
        if table.db or table.catalog:
            raise UnsafeQueryError(
                f"'{table.sql()}' refers to another database. Use plain table names only."
            )
        if not name:
            raise UnsafeQueryError("This kind of table expression is not supported.")
        if name.casefold() in cte_names:
            continue  # a named sub-query defined in the WITH clause
        validate_identifier(name)
        if not allowlist.allows(name):
            raise UnsafeQueryError(
                f"Table '{name}' is not on the approved list. "
                f"Approved tables: {allowlist.describe()}."
            )


def validate_select(
    sql: object, allowlist: TableAllowlist, dialect: str | None = "mysql"
) -> exp.Expression:
    """Check a custom query and return its parsed form. Raises ``UnsafeQueryError``."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeQueryError("Please enter a SQL query.")
    if len(sql) > MAX_QUERY_LENGTH:
        raise UnsafeQueryError(f"The query is too long (max {MAX_QUERY_LENGTH:,} characters).")
    if "/*!" in sql:
        # MySQL runs the text inside /*! ... */ even though it looks like a comment.
        raise UnsafeQueryError("MySQL executable comments (/*! ... */) are not allowed.")

    _check_tokens(sql, dialect)

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as exc:
        first_line = str(exc).splitlines()[0]
        raise UnsafeQueryError(f"The SQL could not be understood: {first_line}") from None

    if len(statements) != 1:
        raise UnsafeQueryError("Run exactly one statement at a time (no ';' between queries).")
    query = statements[0]
    if not isinstance(query, _QUERY_TYPES):
        raise UnsafeQueryError("Only SELECT queries are allowed.")

    forbidden = next(query.find_all(*_FORBIDDEN_NODES), None)
    if forbidden is not None:
        raise UnsafeQueryError(
            f"This query contains a non-read-only operation ({type(forbidden).__name__})."
        )
    for func in query.find_all(exp.Func):
        name = func.name if isinstance(func, exp.Anonymous) else func.sql_name()
        if name.upper() in FORBIDDEN_FUNCTIONS:
            raise UnsafeQueryError(f"The function {name.upper()}() is not allowed.")

    _check_tables(query, allowlist)
    return query


def _literal_limit(query: exp.Expression) -> int | None:
    limit = query.args.get("limit")
    value = limit.args.get("expression") if limit is not None else None
    if isinstance(value, exp.Literal) and value.is_int:
        return int(value.name)
    return None


def apply_row_limit(query: exp.Expression, max_rows: int) -> exp.Expression:
    """Return a copy of the query that returns at most ``max_rows`` rows.

    A smaller LIMIT written by the user is kept; a larger or missing one is
    replaced. UNION queries are wrapped: SELECT * FROM (...) LIMIT n.
    """
    max_rows = validate_row_limit(max_rows)
    if isinstance(query, exp.Select):
        current = _literal_limit(query)
        if current is not None and current <= max_rows:
            return query.copy()
        return query.limit(max_rows, copy=True)
    return exp.select("*").from_(query.subquery("limited_result", copy=True)).limit(max_rows)


def prepare_select(
    sql: object,
    allowlist: TableAllowlist,
    max_rows: int = DEFAULT_ROW_LIMIT,
    dialect: str | None = "mysql",
) -> str:
    """Validate a custom query and return the safe SQL text to execute."""
    max_rows = validate_row_limit(max_rows)
    query = apply_row_limit(validate_select(sql, allowlist, dialect), max_rows)
    return query.sql(dialect=dialect, comments=False)


def allowed_subset(names: Iterable[str], allowlist: TableAllowlist) -> list[str]:
    """The names (as spelled in the database) that the allowlist approves."""
    return sorted((n for n in names if allowlist.allows(n)), key=str.casefold)
