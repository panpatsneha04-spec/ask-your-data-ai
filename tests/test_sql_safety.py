import pytest

from askyourdata.sql_safety import (
    MAX_ROW_LIMIT,
    TableAllowlist,
    UnsafeQueryError,
    allowed_subset,
    prepare_select,
    validate_identifier,
    validate_select,
)

ALLOW = TableAllowlist.of("sales", "customers")


# --- identifiers and the allowlist ---------------------------------------------


@pytest.mark.parametrize("name", ["sales", "_tmp", "Orders2026", "t$1", "a" * 64])
def test_valid_identifiers(name):
    assert validate_identifier(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "1sales", "sales;", "my table", "sales--", "`sales`", "db.sales", "a" * 65, None, 42],
)
def test_invalid_identifiers(name):
    with pytest.raises(UnsafeQueryError, match="not valid"):
        validate_identifier(name)


def test_allowlist_from_string_ignores_case_and_spaces():
    allowlist = TableAllowlist.from_string(" sales , Customers ")
    assert allowlist.allows("SALES")
    assert allowlist.allows("customers")
    assert not allowlist.allows("orders")
    assert not allowlist.allows("sales; DROP TABLE x")


def test_empty_allowlist_allows_nothing():
    allowlist = TableAllowlist.from_string("")
    assert allowlist.is_empty
    assert not allowlist.allows("sales")
    assert allowlist.describe() == "none"


def test_star_allowlist_still_requires_valid_names():
    allowlist = TableAllowlist.from_string("*")
    assert allowlist.allow_all and allowlist.allows("anything")
    assert not allowlist.allows("bad name")


@pytest.mark.parametrize("raw", ["*, sales", "sales; drop", "db.sales", "my table"])
def test_invalid_allowlist_settings(raw):
    with pytest.raises(UnsafeQueryError):
        TableAllowlist.from_string(raw)


def test_allowed_subset_keeps_database_spelling():
    assert allowed_subset(["Sales", "orders", "customers"], ALLOW) == ["customers", "Sales"]


# --- accepted queries ------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sales",
        "select region, SUM(units) AS total from Sales group by region order by total desc",
        "SELECT s.region, c.name FROM sales s JOIN customers c ON s.id = c.id",
        "WITH big AS (SELECT * FROM sales WHERE units > 5) SELECT * FROM big",
        "SELECT * FROM sales WHERE product = 'DROP TABLE sales; DELETE'",  # just text
        "SELECT `update` FROM sales",  # a quoted column name, not the keyword
        "SELECT region FROM sales UNION SELECT name FROM customers",
        "SELECT * FROM sales WHERE id IN (SELECT id FROM customers)",
        "SELECT REPLACE(product, 'a', 'b') FROM sales",
        "SELECT * FROM sales -- a comment\n",
        "SELECT * FROM sales;",
        "SELECT 1",
    ],
)
def test_accepts_read_only_selects(sql):
    validate_select(sql, ALLOW)


# --- rejected queries --------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("", "enter a SQL query"),
        ("   ", "enter a SQL query"),
        (None, "enter a SQL query"),
        ("INSERT INTO sales VALUES (1)", "INSERT"),
        ("UPDATE sales SET units = 0", "UPDATE"),
        ("DELETE FROM sales", "DELETE"),
        ("DROP TABLE sales", "DROP"),
        ("ALTER TABLE sales ADD x INT", "ALTER"),
        ("CREATE TABLE x (a INT)", "CREATE"),
        ("TRUNCATE TABLE sales", "TRUNCATE"),
        ("REPLACE INTO sales VALUES (1)", "Only SELECT"),
        ("sElEcT 1; dRoP TABLE sales", "DROP"),
        ("SELECT * FROM sales; SELECT * FROM customers", "exactly one statement"),
        ("WITH c AS (SELECT 1) UPDATE sales SET units = 1", "UPDATE"),
        ("SELECT * FROM sales INTO OUTFILE '/tmp/x'", "INTO"),
        ("SELECT * INTO @v FROM sales", "INTO"),
        ("SELECT * FROM sales FOR UPDATE", "UPDATE"),
        ("SELECT * FROM sales LOCK IN SHARE MODE", "LOCK"),
        ("SELECT SLEEP(10)", "SLEEP"),
        ("SELECT BENCHMARK(1000000, MD5('x'))", "BENCHMARK"),
        ("SELECT LOAD_FILE('/etc/passwd')", "LOAD_FILE"),
        ("SELECT /*! 1; DROP TABLE sales */ 2", "executable comments"),
        ("SHOW TABLES", "Only SELECT"),
        ("EXPLAIN SELECT * FROM sales", "Only SELECT"),
        ("CALL do_something()", "CALL"),
        ("SET @a = 1", "SET"),
        ("GRANT ALL ON sales TO someone", "GRANT"),
        ("SELECT * FROM secrets", "not on the approved list"),
        ("SELECT * FROM sales JOIN secrets ON 1 = 1", "'secrets'"),
        ("SELECT * FROM sales WHERE id IN (SELECT id FROM secrets)", "'secrets'"),
        ("WITH x AS (SELECT * FROM secrets) SELECT * FROM x", "'secrets'"),
        ("SELECT * FROM other_db.sales", "another database"),
        ("SELECT * FROM information_schema.tables", "another database"),
        ("SELECT * FROM mysql.user", "another database"),
        ("SELECT " + "x, " * 4000 + "1 FROM sales", "too long"),
    ],
)
def test_rejects_unsafe_queries(sql, message):
    with pytest.raises(UnsafeQueryError, match=message):
        validate_select(sql, ALLOW)


def test_rejects_unreadable_sql():
    with pytest.raises(UnsafeQueryError):
        validate_select("SELEC * FROM sales WHERE (", ALLOW)


def test_empty_allowlist_blocks_every_table():
    with pytest.raises(UnsafeQueryError, match="approved list"):
        validate_select("SELECT * FROM sales", TableAllowlist())


# --- row limits and the final SQL ---------------------------------------------------


def test_adds_limit_when_missing():
    assert prepare_select("SELECT * FROM sales", ALLOW, 500).endswith("LIMIT 500")


def test_keeps_smaller_user_limit():
    assert prepare_select("SELECT * FROM sales LIMIT 10", ALLOW, 500).endswith("LIMIT 10")


def test_lowers_larger_user_limit():
    assert prepare_select("SELECT * FROM sales LIMIT 100000", ALLOW, 500).endswith("LIMIT 500")


def test_keeps_offset_when_lowering_limit():
    sql = prepare_select("SELECT * FROM sales LIMIT 5, 100000", ALLOW, 500)
    assert sql.endswith("LIMIT 500 OFFSET 5")


def test_union_is_wrapped_with_limit():
    sql = prepare_select("SELECT region FROM sales UNION SELECT name FROM customers", ALLOW, 50)
    assert sql.startswith("SELECT * FROM (")
    assert sql.endswith("LIMIT 50")


def test_comments_are_removed_from_executed_sql():
    sql = prepare_select("SELECT * FROM sales /* hidden note */ -- another", ALLOW, 5)
    assert "hidden" not in sql and "another" not in sql


@pytest.mark.parametrize("bad_limit", [0, -1, MAX_ROW_LIMIT + 1, "500", 2.5, True, None])
def test_rejects_invalid_row_limits(bad_limit):
    with pytest.raises(UnsafeQueryError, match="row limit"):
        prepare_select("SELECT * FROM sales", ALLOW, bad_limit)
