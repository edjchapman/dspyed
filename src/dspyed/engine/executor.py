"""SafeExecutor: run model-generated SQL against benchmark SQLite files, safely.

Defense in depth — each layer exists because a different failure mode bypasses
the others:

1. db_id hygiene: ``^[A-Za-z0-9_]+$`` + must resolve inside the database root
   (no path traversal; unknown ids rejected before any file I/O).
2. Static whitelist BEFORE execution: sqlglot must parse exactly one statement
   whose root is a SELECT / set operation (CTEs included). PRAGMA, ATTACH,
   VACUUM, writes, and multi-statement input never reach SQLite.
3. Read-only at the VFS: ``mode=ro&immutable=1`` — benchmark DBs never change,
   and no write can succeed even if SQL slips past the parser.
4. ``PRAGMA query_only=ON`` — belt over braces, applied BEFORE the authorizer
   is installed (the authorizer denies SQLITE_PRAGMA, including our own).
5. Authorizer: deny-by-default; only read-class opcodes are allowed. Catches
   whatever the parser mis-modeled.
6. Wall-clock timeout via a progress handler: kills runaway cartesian joins
   and WITH RECURSIVE bombs — valid, read-only SQL that no other layer stops.
7. Row cap via chunked fetch: bounds memory; ``truncated=True`` marks capped
   results so the metric can flag the comparison as weakened.

Rejection reasons are human-readable strings: the same text feeds the repair
signature and GEPA feedback, so wording here is part of the training signal.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

from dspyed.engine.compare import RawRow

_DB_ID_RE: Final = re.compile(r"^[A-Za-z0-9_]+$")

# Root expression types that constitute a read query. Union/Except/Intersect
# cover set operations; a WITH ... SELECT parses as a Select carrying CTEs.
_SELECT_ROOTS: Final = (exp.Select, exp.Union, exp.Except, exp.Intersect)

# Authorizer opcodes allowed (deny-by-default). SQLITE_RECURSIVE is required
# for WITH RECURSIVE, which is legitimate read-only SQL — the timeout is the
# layer that handles its abuse. Fallback literals cover older Pythons that
# don't export a constant.
_ALLOWED_OPCODES: Final = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)

_PROGRESS_HANDLER_OPCODES: Final = 10_000  # VM instructions between deadline checks


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one guarded execution. ``error`` is None iff ``ok``."""

    ok: bool
    columns: tuple[str, ...]
    rows: tuple[RawRow, ...]
    error: str | None
    truncated: bool
    elapsed_ms: float


def _failure(error: str, elapsed_ms: float = 0.0) -> ExecResult:
    return ExecResult(
        ok=False, columns=(), rows=(), error=error, truncated=False, elapsed_ms=elapsed_ms
    )


def validate_sql(sql: str) -> str | None:
    """Static whitelist. Returns a rejection reason, or None if allowed.

    sqlglot is stricter than SQLite in places — a query it cannot parse is
    rejected here with a reason the repair loop can act on. sqlglot being
    *looser* than SQLite is fine: SQLite itself errors next.
    """
    if not sql or not sql.strip():
        return "rejected: empty SQL"
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as err:
        return f"rejected: not parseable as SQLite SQL ({err.errors[0]['description']})"
    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        return f"rejected: expected exactly one statement, got {len(statements)}"
    root = statements[0]
    if not isinstance(root, _SELECT_ROOTS):
        return f"rejected: only SELECT queries are allowed, got {root.key.upper()}"
    return None


class SafeExecutor:
    """Executes whitelisted SELECTs against SQLite files under ``db_root``.

    Layout contract (Spider's): ``<db_root>/<db_id>/<db_id>.sqlite``.
    A fresh connection per query — no shared state, safe under Evaluate's
    thread pool.
    """

    def __init__(
        self,
        db_root: Path,
        *,
        timeout_s: float = 5.0,
        max_rows: int = 10_000,
    ) -> None:
        self._db_root = db_root
        self._timeout_s = timeout_s
        self._max_rows = max_rows

    def database_path(self, db_id: str) -> Path | None:
        """Resolve a db_id to its SQLite file; None if invalid or absent."""
        if not _DB_ID_RE.match(db_id):
            return None
        path = self._db_root / db_id / f"{db_id}.sqlite"
        return path if path.is_file() else None

    def run(self, db_id: str, sql: str) -> ExecResult:
        path = self.database_path(db_id)
        if path is None:
            return _failure(f"rejected: unknown database id {db_id!r}")

        rejection = validate_sql(sql)
        if rejection is not None:
            return _failure(rejection)

        start = time.monotonic()
        deadline = start + self._timeout_s

        def _elapsed_ms() -> float:
            return (time.monotonic() - start) * 1000

        def _authorize(
            action: int,
            arg1: str | None,
            arg2: str | None,
            db_name: str | None,
            trigger: str | None,
        ) -> int:
            return sqlite3.SQLITE_OK if action in _ALLOWED_OPCODES else sqlite3.SQLITE_DENY

        def _check_deadline() -> int:
            return 1 if time.monotonic() > deadline else 0

        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        except sqlite3.Error as err:
            return _failure(f"connection error: {err}", _elapsed_ms())

        try:
            # Order is load-bearing: query_only must be set BEFORE the
            # authorizer exists, or the authorizer would deny our own PRAGMA.
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(_authorize)
            conn.set_progress_handler(_check_deadline, _PROGRESS_HANDLER_OPCODES)

            cursor = conn.execute(sql)
            columns = tuple(desc[0] for desc in cursor.description or ())

            rows: list[RawRow] = []
            truncated = False
            while True:
                if time.monotonic() > deadline:
                    return _failure(
                        f"timed out after {self._timeout_s:g}s while fetching rows",
                        _elapsed_ms(),
                    )
                chunk = cursor.fetchmany(min(1_000, self._max_rows - len(rows) + 1))
                if not chunk:
                    break
                rows.extend(tuple(row) for row in chunk)
                if len(rows) > self._max_rows:
                    del rows[self._max_rows :]
                    truncated = True
                    break

            return ExecResult(
                ok=True,
                columns=columns,
                rows=tuple(rows),
                error=None,
                truncated=truncated,
                elapsed_ms=_elapsed_ms(),
            )
        except sqlite3.Error as err:
            message = str(err)
            if "interrupted" in message:
                message = f"timed out after {self._timeout_s:g}s ({message})"
            return _failure(message, _elapsed_ms())
        finally:
            conn.close()
