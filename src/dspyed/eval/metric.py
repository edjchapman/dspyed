"""Execution accuracy — the headline metric.

Score = 1.0 iff the predicted SQL executes and its result set matches the
gold query's result set on the same database (multiset comparison; ordered
iff the gold query has a top-level ORDER BY). Everything else is 0.0 —
rejections, SQLite errors, and timeouts included, each recorded with an
error class so failure modes are analyzable.

Design notes:
- The gold query failing to execute is a *red flag*, not noise: Phase 1
  acceptance guarantees zero gold errors on our splits, so ``gold_error``
  being set later means the data or executor regressed.
- ``trace is not None`` signals optimizer bootstrapping (DSPy convention):
  we return a strict bool and skip diagnostics.
- Diagnostics (``matched_loose``, ``both_empty``, flags) never change the
  score; their aggregate rates are published next to the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
import sqlglot.errors
from sqlglot import Expr

from dspyed.engine import SafeExecutor, compare_results


def _parse_gold(sql: str) -> Expr | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return None


def gold_is_ordered(sql: str) -> bool:
    """True iff the gold query imposes a row order (top-level ORDER BY).

    sqlglot attaches ORDER BY to the outermost node for both plain selects
    and set operations, so one args lookup covers both shapes. An unparseable
    gold query (should never happen on our splits) compares unordered.
    """
    root = _parse_gold(sql)
    return root is not None and root.args.get("order") is not None


def gold_has_limit_without_order(sql: str) -> bool:
    """LIMIT with no ORDER BY: the gold result set itself is nondeterministic."""
    root = _parse_gold(sql)
    if root is None:
        return False
    return root.args.get("limit") is not None and root.args.get("order") is None


def classify_error(message: str) -> str:
    """Bucket an execution failure for the error taxonomy."""
    if message.startswith("rejected:"):
        return "rejected"
    if "timed out" in message:
        return "timeout"
    if "syntax error" in message:
        return "syntax"
    if "no such table" in message or "no such column" in message:
        return "schema"
    return "other"


@dataclass(frozen=True)
class MetricOutcome:
    """Per-example scoring record; the harness persists these verbatim."""

    score: float
    matched_loose: bool
    both_empty: bool
    flags: tuple[str, ...]
    error_class: str | None  # set iff the PREDICTED query failed to execute
    gold_error: str | None  # set iff the GOLD query failed — a red flag


def _get_str_attr(obj: object, name: str) -> str | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, str) else None


class ExecutionAccuracy:
    """Callable metric, DSPy-shaped: ``metric(example, pred, trace=None)``.

    ``example`` must carry ``sql`` (gold) and ``db_id``; ``pred`` must carry
    ``sql``. Anything missing scores 0.0 rather than raising — a malformed
    prediction is a wrong prediction, not a crash.
    """

    def __init__(self, executor: SafeExecutor) -> None:
        self._executor = executor

    def detailed(self, gold_sql: str, db_id: str, pred_sql: str | None) -> MetricOutcome:
        if not pred_sql or not pred_sql.strip():
            return MetricOutcome(
                score=0.0,
                matched_loose=False,
                both_empty=False,
                flags=(),
                error_class="rejected",
                gold_error=None,
            )

        gold = self._executor.run(db_id, gold_sql)
        if not gold.ok:
            return MetricOutcome(
                score=0.0,
                matched_loose=False,
                both_empty=False,
                flags=(),
                error_class=None,
                gold_error=gold.error,
            )

        pred = self._executor.run(db_id, pred_sql)
        if not pred.ok:
            return MetricOutcome(
                score=0.0,
                matched_loose=False,
                both_empty=False,
                flags=(),
                error_class=classify_error(pred.error or "unknown execution error"),
                gold_error=None,
            )

        flags: list[str] = []
        if gold_has_limit_without_order(gold_sql):
            flags.append("limit_no_order")
        if gold.truncated or pred.truncated:
            flags.append("row_capped")

        comparison = compare_results(
            list(gold.rows),
            list(pred.rows),
            ordered=gold_is_ordered(gold_sql),
        )
        return MetricOutcome(
            score=1.0 if comparison.matched else 0.0,
            matched_loose=comparison.matched_loose,
            both_empty=comparison.both_empty,
            flags=tuple(flags),
            error_class=None,
            gold_error=None,
        )

    def __call__(self, example: object, pred: object, trace: object | None = None) -> float | bool:
        gold_sql = _get_str_attr(example, "sql")
        db_id = _get_str_attr(example, "db_id")
        pred_sql = _get_str_attr(pred, "sql")
        if gold_sql is None or db_id is None:
            outcome_score = 0.0
        else:
            outcome_score = self.detailed(gold_sql, db_id, pred_sql).score
        if trace is not None:  # optimizer bootstrapping: strict bool (DSPy convention)
            return outcome_score >= 1.0
        return outcome_score
