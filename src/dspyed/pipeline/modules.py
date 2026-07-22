"""The TextToSQL pipeline module and SQL output hygiene.

Control flow: link (optional) → generate → execute → repair loop. The repair
loop fires ONLY on execution failure and its feedback is the verbatim
executor error — it improves *validity*, not correctness (there is no oracle
at inference time; correctness comes from linking, CoT, and compilation).
That asymmetry is stated in the eval writeup.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

import dspy

from dspyed.data.schema import DBSchema, render_subset
from dspyed.engine import SafeExecutor
from dspyed.pipeline.signatures import GenerateSQL, LinkSchema, RepairSQL

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_STATEMENT_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Attempt:
    """One generate-or-repair round: what ran and how it ended."""

    sql: str
    error: str | None
    latency_ms: float


def clean_sql(raw: str) -> str:
    """Normalize model output to (at most) one bare SQL statement.

    Order of operations: prefer a fenced block if present; then drop any
    leading prose by cutting at the first SELECT/WITH; then cut at the first
    complete statement (semicolons inside string literals are respected via
    sqlite3.complete_statement); finally strip the trailing semicolon. If
    nothing looks like SQL, return the stripped text — the executor's
    rejection message is more instructive than an empty string.
    """
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = _STATEMENT_START_RE.search(text)
    if start:
        text = text[start.start() :].strip()

    for index, char in enumerate(text):
        if char == ";" and sqlite3.complete_statement(text[: index + 1]):
            text = text[:index]
            break

    return text.strip().rstrip(";").strip()


class TextToSQL(dspy.Module):
    """link (optional) → generate → execute → repair (≤ ``max_repairs``).

    The linker returns typed ``list[str]`` identifiers, never schema text —
    the subset is re-rendered deterministically via ``render_subset``, so the
    LM cannot corrupt the schema and garbage degrades to the full render.

    ``schema_lookup`` maps db_id → DBSchema (needed to re-render subsets);
    the full rendered schema still arrives as the ``db_schema`` input so the
    module works on any example the harness builds.
    """

    def __init__(
        self,
        executor: SafeExecutor,
        schema_lookup: Callable[[str], DBSchema],
        *,
        use_linking: bool = True,
        use_cot: bool = True,
        max_repairs: int = 2,
    ) -> None:
        super().__init__()
        generate_cls = dspy.ChainOfThought if use_cot else dspy.Predict
        self.link = dspy.Predict(LinkSchema) if use_linking else None
        self.generate = generate_cls(GenerateSQL)
        self.repair = generate_cls(RepairSQL)
        self._executor = executor
        self._schema_lookup = schema_lookup
        self._max_repairs = max_repairs

    def _linked_schema(self, question: str, db_schema: str, db_id: str) -> str:
        if self.link is None:
            return db_schema
        selection = self.link(question=question, db_schema=db_schema).relevant_columns
        columns = [str(item) for item in selection] if isinstance(selection, list) else []
        return render_subset(self._schema_lookup(db_id), columns, fallback=db_schema)

    def forward(self, question: str, db_schema: str, db_id: str) -> dspy.Prediction:
        linked = self._linked_schema(question, db_schema, db_id)

        started = time.monotonic()
        sql = clean_sql(self.generate(question=question, db_schema=linked).sql)
        result = self._executor.run(db_id, sql)
        attempts = [Attempt(sql, result.error, (time.monotonic() - started) * 1000)]

        for _ in range(self._max_repairs):
            if result.ok:
                break
            started = time.monotonic()
            repaired = self.repair(
                question=question,
                db_schema=linked,
                failed_sql=sql,
                error=result.error or "unknown execution error",
            )
            sql = clean_sql(repaired.sql)
            result = self._executor.run(db_id, sql)
            attempts.append(Attempt(sql, result.error, (time.monotonic() - started) * 1000))

        return dspy.Prediction(
            sql=sql,
            ok=result.ok,
            columns=list(result.columns),
            rows=[list(row) for row in result.rows],
            error=result.error,
            attempts=attempts,
            linked_schema=linked,
        )
