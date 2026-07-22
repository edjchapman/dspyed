"""SafeExecutor safety battery.

The write-rejection tests are the contract that makes serving model-generated
SQL defensible; the layer-independence test proves the defenses are redundant,
not sequential.
"""

from pathlib import Path

import pytest

import dspyed.engine.executor as executor_module
from dspyed.engine import SafeExecutor

DB = "mini_singer"


@pytest.fixture
def executor(db_root: Path) -> SafeExecutor:
    return SafeExecutor(db_root)


# --- Happy paths -------------------------------------------------------------


def test_select_returns_columns_and_rows(executor: SafeExecutor):
    result = executor.run(DB, "SELECT name, age FROM singer WHERE country = 'UK' ORDER BY age")
    assert result.ok, result.error
    assert result.columns == ("name", "age")
    assert result.rows == (("Dana", 19), ("Ava", 30))
    assert result.error is None
    assert result.elapsed_ms >= 0


def test_cte_and_union_and_aggregate_are_allowed(executor: SafeExecutor):
    for sql in (
        "WITH uk AS (SELECT * FROM singer WHERE country = 'UK') SELECT count(*) FROM uk",
        "SELECT name FROM singer UNION SELECT venue FROM concert",
        "SELECT country, count(*) FROM singer GROUP BY country",
        "SELECT 1;",  # trailing semicolon is fine — still one statement
    ):
        result = executor.run(DB, sql)
        assert result.ok, f"{sql!r} → {result.error}"


# --- Whitelist rejections ----------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO singer VALUES (9, 'Eve', 'DE', 20, 0)",
        "UPDATE singer SET age = 99",
        "DELETE FROM singer",
        "DROP TABLE singer",
        "CREATE TABLE pwn (x)",
        "PRAGMA journal_mode=DELETE",
        "VACUUM",
        "SELECT 1; DROP TABLE singer",
    ],
)
def test_non_select_is_rejected_before_execution(executor: SafeExecutor, sql: str):
    result = executor.run(DB, sql)
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("rejected:")


def test_unparseable_sql_is_rejected_with_reason(executor: SafeExecutor):
    result = executor.run(DB, "SELEC * FRM singer")
    assert not result.ok
    assert result.error is not None
    assert "rejected" in result.error


def test_empty_sql_is_rejected(executor: SafeExecutor):
    assert not executor.run(DB, "   ").ok


# --- Layer independence ------------------------------------------------------


def test_write_fails_even_if_whitelist_is_bypassed(
    executor: SafeExecutor, monkeypatch: pytest.MonkeyPatch, db_root: Path
):
    """Disable layer 2 (the parser whitelist) entirely: the read-only VFS,
    query_only pragma, and authorizer must still block the write."""
    monkeypatch.setattr(executor_module, "validate_sql", lambda sql: None)
    result = executor.run(DB, "INSERT INTO singer VALUES (9, 'Eve', 'DE', 20, 0)")
    assert not result.ok

    # And nothing was written.
    count = executor.run(DB, "SELECT count(*) FROM singer")
    assert count.ok
    assert count.rows == ((5,),)


# --- db_id hygiene -----------------------------------------------------------


@pytest.mark.parametrize("db_id", ["../../etc", "no_such_db", "a/b", ""])
def test_bad_db_ids_are_rejected(executor: SafeExecutor, db_id: str):
    result = executor.run(db_id, "SELECT 1")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("rejected: unknown database id")


# --- Resource bounds ---------------------------------------------------------


def test_recursive_bomb_times_out(db_root: Path):
    fast = SafeExecutor(db_root, timeout_s=0.2)
    result = fast.run(
        DB,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
    )
    assert not result.ok
    assert result.error is not None
    assert "timed out" in result.error
    assert result.elapsed_ms >= 200


def test_row_cap_truncates_and_flags(db_root: Path):
    capped = SafeExecutor(db_root, max_rows=3)
    result = capped.run(DB, "SELECT name FROM singer")
    assert result.ok
    assert result.truncated is True
    assert len(result.rows) == 3
