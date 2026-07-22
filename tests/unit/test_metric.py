"""Execution-accuracy metric against the fixture database.

SimpleNamespace stands in for dspy.Example / dspy.Prediction — the metric is
deliberately duck-typed so these tests need no dspy import.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dspyed.engine import SafeExecutor
from dspyed.eval import ExecutionAccuracy, gold_is_ordered
from dspyed.eval.metric import classify_error, gold_has_limit_without_order

DB = "mini_singer"


@pytest.fixture(scope="module")
def metric(db_root: Path) -> ExecutionAccuracy:
    return ExecutionAccuracy(SafeExecutor(db_root))


# --- Scoring -----------------------------------------------------------------


def test_gold_matches_itself(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT name FROM singer", DB, "SELECT name FROM singer")
    assert outcome.score == 1.0
    assert outcome.gold_error is None


def test_semantically_equal_sql_matches(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name FROM singer WHERE country = 'UK'",
        DB,
        "SELECT name FROM singer WHERE country IN ('UK')",
    )
    assert outcome.score == 1.0


def test_unordered_gold_forgives_row_order(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name FROM singer",
        DB,
        "SELECT name FROM singer ORDER BY age DESC",
    )
    assert outcome.score == 1.0


def test_ordered_gold_enforces_row_order(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name FROM singer ORDER BY age",
        DB,
        "SELECT name FROM singer ORDER BY age DESC",
    )
    assert outcome.score == 0.0


def test_wrong_result_scores_zero(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT count(*) FROM singer", DB, "SELECT 999")
    assert outcome.score == 0.0


def test_missing_distinct_scores_zero(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT DISTINCT name FROM singer",  # 'Ben' appears twice in the table
        DB,
        "SELECT name FROM singer",
    )
    assert outcome.score == 0.0


# --- Failure classification --------------------------------------------------


def test_pred_syntax_error_classified(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT 1", DB, "SELEC * FRM singer")
    assert outcome.score == 0.0
    assert outcome.error_class == "rejected"  # sqlglot rejects before SQLite sees it


def test_pred_missing_column_classified_as_schema(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT 1", DB, "SELECT nation FROM singer")
    assert outcome.score == 0.0
    assert outcome.error_class == "schema"


def test_empty_pred_scores_zero(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT 1", DB, "")
    assert outcome.score == 0.0
    assert outcome.error_class == "rejected"


def test_gold_failure_is_flagged_not_swallowed(metric: ExecutionAccuracy):
    outcome = metric.detailed("SELECT * FROM no_such_table", DB, "SELECT 1")
    assert outcome.score == 0.0
    assert outcome.gold_error is not None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("rejected: only SELECT queries are allowed, got PRAGMA", "rejected"),
        ("timed out after 5s while fetching rows", "timeout"),
        ('near "FRM": syntax error', "syntax"),
        ("no such column: singer.nation", "schema"),
        ("disk I/O error", "other"),
    ],
)
def test_classify_error(message: str, expected: str):
    assert classify_error(message) == expected


# --- Diagnostics -------------------------------------------------------------


def test_both_empty_matches_but_flagged(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name FROM singer WHERE age > 100",
        DB,
        "SELECT name FROM singer WHERE 1 = 0",
    )
    assert outcome.score == 1.0
    assert outcome.both_empty is True


def test_limit_without_order_flagged(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name FROM singer LIMIT 2",
        DB,
        "SELECT name FROM singer LIMIT 2",
    )
    assert "limit_no_order" in outcome.flags


def test_column_order_swap_is_loose_only(metric: ExecutionAccuracy):
    outcome = metric.detailed(
        "SELECT name, age FROM singer",
        DB,
        "SELECT age, name FROM singer",
    )
    assert outcome.score == 0.0  # strict headline
    assert outcome.matched_loose is True  # published as a diagnostic rate


# --- Gold SQL analysis -------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "ordered"),
    [
        ("SELECT name FROM singer ORDER BY age", True),
        ("SELECT name FROM singer", False),
        ("SELECT name FROM singer UNION SELECT venue FROM concert ORDER BY 1", True),
        ("not sql at all", False),
    ],
)
def test_gold_is_ordered(sql: str, ordered: bool):
    assert gold_is_ordered(sql) is ordered


def test_limit_without_order_detection():
    assert gold_has_limit_without_order("SELECT 1 LIMIT 3") is True
    assert gold_has_limit_without_order("SELECT 1 ORDER BY 1 LIMIT 3") is False


# --- DSPy-shaped call surface ------------------------------------------------


def test_call_shape_and_trace_mode(metric: ExecutionAccuracy):
    example = SimpleNamespace(sql="SELECT count(*) FROM singer", db_id=DB)
    good = SimpleNamespace(sql="SELECT count(*) FROM singer")
    bad = SimpleNamespace(sql="SELECT 0")

    assert metric(example, good) == 1.0
    assert metric(example, bad) == 0.0
    assert metric(example, good, trace=object()) is True
    assert metric(example, bad, trace=object()) is False
    assert metric(SimpleNamespace(), good) == 0.0  # malformed example scores, not raises
