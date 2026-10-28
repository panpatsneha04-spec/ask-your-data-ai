import os

import pytest

from askyourdata.database import create_mysql_engine
from askyourdata.db_config import ConfigError, DatabaseSettings, load_env_file

PASSWORD = "s3cr3t!pass"

FULL_ENV = {
    "DB_HOST": "db.example.test",
    "DB_PORT": "3307",
    "DB_USER": "reader",
    "DB_PASSWORD": PASSWORD,
    "DB_NAME": "shop",
    "DB_ALLOWED_TABLES": "orders, customers",
    "DB_MAX_ROWS": "250",
    "DB_QUERY_TIMEOUT_SECONDS": "30",
    "DB_ALLOW_CUSTOM_SQL": "false",
}


def test_defaults_when_nothing_is_set():
    settings = DatabaseSettings.from_env({})
    assert (settings.host, settings.port) == ("localhost", 3306)
    assert settings.max_rows == 500
    assert settings.timeout_seconds == 15
    assert settings.allow_custom_sql is True
    assert settings.allowed_tables.is_empty  # nothing readable until configured


def test_reads_all_settings():
    settings = DatabaseSettings.from_env(FULL_ENV)
    assert settings.host == "db.example.test"
    assert settings.port == 3307
    assert settings.max_rows == 250
    assert settings.timeout_seconds == 30
    assert settings.allow_custom_sql is False
    assert settings.allowed_tables.allows("ORDERS")
    assert not settings.allowed_tables.allows("payments")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DB_PORT", "abc"),
        ("DB_PORT", "70000"),
        ("DB_MAX_ROWS", "0"),
        ("DB_MAX_ROWS", "20000"),
        ("DB_QUERY_TIMEOUT_SECONDS", "-5"),
        ("DB_ALLOW_CUSTOM_SQL", "maybe"),
        ("DB_ALLOWED_TABLES", "orders; DROP TABLE x"),
    ],
)
def test_invalid_settings_name_the_variable(key, value):
    with pytest.raises(ConfigError, match=key):
        DatabaseSettings.from_env({**FULL_ENV, key: value})


def test_password_never_appears_in_text_output():
    settings = DatabaseSettings.from_env(FULL_ENV)
    assert PASSWORD not in repr(settings)
    assert PASSWORD not in settings.describe()
    url = settings.url()
    assert PASSWORD not in str(url)  # SQLAlchemy prints *** instead
    assert url.password == PASSWORD


def test_validate_reports_missing_fields():
    with pytest.raises(ConfigError, match="Missing connection details: user, database name"):
        DatabaseSettings(host="localhost").validate()


def test_validate_rejects_bad_database_name():
    settings = DatabaseSettings.from_env({**FULL_ENV, "DB_NAME": "shop; DROP"})
    with pytest.raises(ConfigError, match="not valid"):
        settings.validate()


def test_validate_rejects_missing_ssl_file():
    settings = DatabaseSettings.from_env({**FULL_ENV, "DB_SSL_CA": "no/such/ca.pem"})
    with pytest.raises(ConfigError, match="DB_SSL_CA"):
        settings.validate()


def test_mysql_engine_is_created_without_connecting():
    engine = create_mysql_engine(DatabaseSettings.from_env(FULL_ENV))
    assert engine.dialect.name == "mysql"
    assert engine.dialect.driver == "pymysql"
    assert PASSWORD not in str(engine.url)
    assert PASSWORD not in repr(engine)


def test_load_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AYD_TEST_FROM_FILE=file\nAYD_TEST_EXISTING=file\n", encoding="utf-8")
    monkeypatch.setenv("AYD_TEST_EXISTING", "real")
    try:
        assert load_env_file(env_file) is True
        assert os.environ["AYD_TEST_FROM_FILE"] == "file"
        assert os.environ["AYD_TEST_EXISTING"] == "real"
    finally:
        os.environ.pop("AYD_TEST_FROM_FILE", None)


def test_load_env_file_missing_file(tmp_path):
    assert load_env_file(tmp_path / "missing.env") is False
