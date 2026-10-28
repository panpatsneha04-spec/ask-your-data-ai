"""Database connection settings, read from environment variables / a .env file.

The password is kept out of ``repr()`` and ``describe()`` so it cannot end up
in logs or error messages by accident.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.engine import URL

from askyourdata.sql_safety import (
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
    TableAllowlist,
    UnsafeQueryError,
    validate_identifier,
)

DEFAULT_TIMEOUT_SECONDS = 15
MAX_TIMEOUT_SECONDS = 300


class ConfigError(ValueError):
    """A setting is missing or invalid. The message never contains the password."""


def load_env_file(path: str | Path = ".env") -> bool:
    """Load KEY=value lines from a .env file into the environment.

    Variables that are already set (e.g. in PowerShell) win over the file.
    Returns False when the file does not exist.
    """
    from dotenv import load_dotenv

    return load_dotenv(path, override=False)


def _env_int(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a whole number (got '{raw}').") from None
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high:,} (got {value}).")
    return value


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be true or false (got '{raw}').")


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = "localhost"
    port: int = 3306
    user: str = ""
    password: str = field(default="", repr=False)
    database: str = ""
    allowed_tables: TableAllowlist = TableAllowlist()
    max_rows: int = DEFAULT_ROW_LIMIT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    allow_custom_sql: bool = True
    ssl_ca: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DatabaseSettings:
        env = os.environ if env is None else env
        try:
            allowlist = TableAllowlist.from_string(env.get("DB_ALLOWED_TABLES"))
        except UnsafeQueryError as exc:
            raise ConfigError(f"DB_ALLOWED_TABLES: {exc}") from None
        return cls(
            host=(env.get("DB_HOST") or "localhost").strip(),
            port=_env_int(env, "DB_PORT", 3306, 1, 65535),
            user=(env.get("DB_USER") or "").strip(),
            password=env.get("DB_PASSWORD") or "",
            database=(env.get("DB_NAME") or "").strip(),
            allowed_tables=allowlist,
            max_rows=_env_int(env, "DB_MAX_ROWS", DEFAULT_ROW_LIMIT, 1, MAX_ROW_LIMIT),
            timeout_seconds=_env_int(
                env, "DB_QUERY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 1, MAX_TIMEOUT_SECONDS
            ),
            allow_custom_sql=_env_bool(env, "DB_ALLOW_CUSTOM_SQL", True),
            ssl_ca=(env.get("DB_SSL_CA") or "").strip(),
        )

    def with_overrides(self, **changes) -> DatabaseSettings:
        """A copy with some fields changed, e.g. values typed into the app."""
        return dataclasses.replace(self, **changes)

    @property
    def has_password(self) -> bool:
        return bool(self.password)

    def validate(self) -> None:
        required = (("host", self.host), ("user", self.user), ("database name", self.database))
        missing = [label for label, value in required if not value]
        if missing:
            raise ConfigError(f"Missing connection details: {', '.join(missing)}.")
        if not 1 <= self.port <= 65535:
            raise ConfigError("The port must be between 1 and 65535.")
        try:
            validate_identifier(self.database, "Database name")
        except UnsafeQueryError as exc:
            raise ConfigError(str(exc)) from None
        if self.ssl_ca and not Path(self.ssl_ca).is_file():
            raise ConfigError(f"DB_SSL_CA file not found: {self.ssl_ca}")

    def url(self) -> URL:
        """SQLAlchemy connection URL. ``URL`` hides the password when printed."""
        self.validate()
        return URL.create(
            "mysql+pymysql",
            username=self.user,
            password=self.password or None,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"charset": "utf8mb4"},
        )

    def describe(self) -> str:
        """Safe to display or log: never includes the password."""
        return f"{self.user}@{self.host}:{self.port}/{self.database}"
