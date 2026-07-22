"""FastAPI demo service.

Serves the compiled DSPy artifact over the curated benchmark databases.
Guardrail stack (defense in depth, mirroring the executor's philosophy):

1. No raw-SQL input path exists — users submit natural language only; the
   generated SQL runs solely under SafeExecutor against allowlisted DBs.
2. Question length cap (400 → 422) and db_id allowlist (404).
3. App-level LRU keyed (db_id, normalized question): example-chip clicks and
   repeat questions are free and instant, and cached answers stay served even
   past the daily cap.
4. Per-IP rate limit (slowapi) + a global daily LM-call cap: past the cap the
   service degrades to cached-only mode (503 with a clear message) instead of
   burning spend.
5. SMALL model only, bounded max_tokens; display rows capped (the executor
   caps at 10k regardless).

The artifact is a state-dict JSON loaded onto a freshly built program at
startup — never a pickle (supply-chain hygiene). Startup fails fast if it is
missing so a mis-deployed image cannot silently serve the uncompiled program.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import dspy
import sqlglot
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from dspyed import __version__
from dspyed.config import Settings, estimate_cost_usd
from dspyed.data.schema import render
from dspyed.engine import SafeExecutor
from dspyed.eval.harness import _SchemaCache, _usage_tokens
from dspyed.pipeline import build_program

MAX_QUESTION_CHARS = 300
DISPLAY_ROW_CAP = 200
LRU_SIZE = 512
DAILY_LM_CALL_CAP = int(os.environ.get("DSPYED_DAILY_CAP", "300"))

limiter = Limiter(key_func=get_remote_address)


class QueryRequest(BaseModel):
    db_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class _DailyBudget:
    """Global LM-call counter that resets at UTC midnight."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._day = ""
        self._count = 0

    def try_spend(self) -> bool:
        today = datetime.now(UTC).date().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    @property
    def used_today(self) -> int:
        return self._count


class _AnswerCache:
    """Bounded LRU of full response payloads, keyed (db_id, normalized question)."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    @staticmethod
    def key(db_id: str, question: str) -> tuple[str, str]:
        return (db_id, " ".join(question.lower().split()))

    def get(self, key: tuple[str, str]) -> dict[str, Any] | None:
        with self._lock:
            payload = self._data.get(key)
            if payload is not None:
                self._data.move_to_end(key)
            return payload

    def put(self, key: tuple[str, str], payload: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = payload
            self._data.move_to_end(key)
            while len(self._data) > self._size:
                self._data.popitem(last=False)


class DemoState:
    """Everything the endpoints need, built once at startup.

    The LM is held here and applied per-request via ``dspy.context`` (thread
    -local) rather than global ``dspy.configure`` — the global is owned by
    whichever thread configures first, which a threaded server must not rely
    on. Tests swap ``state.demo.lm`` for a DummyLM the same way.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lm = dspy.LM(settings.small_model, temperature=0.0, max_tokens=settings.max_tokens_cot)
        database_root = settings.data_root / "database"
        self.executor = SafeExecutor(
            database_root, timeout_s=settings.exec_timeout_s, max_rows=settings.exec_max_rows
        )
        self.schemas = _SchemaCache(database_root)
        self.db_ids = sorted(
            path.name for path in database_root.iterdir() if (path / f"{path.name}.sqlite").exists()
        )
        artifact = os.environ.get("DSPYED_DEMO_ARTIFACT", "artifacts/demo/program.json")
        self.artifact_name = Path(artifact).parent.name or "demo"
        self.program = build_program("p3", self.executor, self.schemas.schema)
        if not Path(artifact).exists():
            raise RuntimeError(
                f"demo artifact {artifact!r} not found — a deployment must never "
                "silently serve the uncompiled program"
            )
        self.program.load(artifact)
        self.budget = _DailyBudget(DAILY_LM_CALL_CAP)
        self.cache = _AnswerCache(LRU_SIZE)
        self.examples: dict[str, list[str]] = _load_canned_examples(self.db_ids)


def _load_canned_examples(db_ids: list[str]) -> dict[str, list[str]]:
    """Three starter questions per DB, from a committed JSON (built in Phase 5)."""
    path = Path("artifacts/demo/examples.json")
    if not path.exists():
        return {db_id: [] for db_id in db_ids}
    import json

    data = json.loads(path.read_text())
    return {db_id: data.get(db_id, [])[:3] for db_id in db_ids}


def _pretty_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite", write="sqlite", pretty=True)[0]
    except Exception:  # noqa: BLE001 — display nicety only; raw SQL is the fallback
        return sql


@asynccontextmanager
async def lifespan(app: FastAPI):
    import litellm

    litellm.drop_params = True
    app.state.demo = DemoState(Settings())
    yield


app = FastAPI(title="dspyed", version=__version__, lifespan=lifespan)
app.state.limiter = limiter
# cast: slowapi types its handler for RateLimitExceeded; starlette's stub wants Exception.
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))


def _demo(request: Request) -> DemoState:
    return request.app.state.demo


@app.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    demo = _demo(request)
    return {
        "status": "ok",
        "version": __version__,
        "artifact": demo.artifact_name,
        "model": demo.settings.small_model,
        "queries_today": demo.budget.used_today,
    }


@app.get("/api/dbs")
def list_dbs(request: Request) -> dict[str, Any]:
    demo = _demo(request)
    return {
        "dbs": [
            {"db_id": db_id, "tables": [t.name for t in demo.schemas.schema(db_id).tables]}
            for db_id in demo.db_ids
        ]
    }


@app.get("/api/schema/{db_id}")
def get_schema(db_id: str, request: Request) -> dict[str, str]:
    demo = _demo(request)
    if db_id not in demo.db_ids:
        raise HTTPException(status_code=404, detail=f"unknown db_id {db_id!r}")
    return {"db_id": db_id, "schema": render(demo.schemas.schema(db_id))}


@app.get("/api/examples/{db_id}")
def get_examples(db_id: str, request: Request) -> dict[str, Any]:
    demo = _demo(request)
    if db_id not in demo.db_ids:
        raise HTTPException(status_code=404, detail=f"unknown db_id {db_id!r}")
    return {"db_id": db_id, "questions": demo.examples.get(db_id, [])}


@app.post("/api/query")
@limiter.limit("10/minute")
def query(request: Request, body: QueryRequest) -> JSONResponse:
    demo = _demo(request)
    if body.db_id not in demo.db_ids:
        raise HTTPException(status_code=404, detail=f"unknown db_id {body.db_id!r}")

    cache_key = demo.cache.key(body.db_id, body.question)
    cached = demo.cache.get(cache_key)
    if cached is not None:
        return JSONResponse({**cached, "cached": True})

    if not demo.budget.try_spend():
        raise HTTPException(
            status_code=503,
            detail="Daily demo budget reached — previously asked questions still work "
            "(they're cached). Fresh questions resume tomorrow (UTC).",
        )

    started = time.monotonic()
    with dspy.context(lm=demo.lm, track_usage=True):
        prediction = demo.program(
            question=body.question,
            db_schema=render(demo.schemas.schema(body.db_id)),
            db_id=body.db_id,
        )
    latency_ms = round((time.monotonic() - started) * 1000)
    prompt_tokens, completion_tokens = _usage_tokens(prediction)

    sql = getattr(prediction, "sql", "") or ""
    rows = [list(row) for row in (getattr(prediction, "rows", None) or [])][:DISPLAY_ROW_CAP]
    attempts = getattr(prediction, "attempts", None) or []
    payload: dict[str, Any] = {
        "sql": sql,
        "pretty_sql": _pretty_sql(sql),
        "ok": bool(getattr(prediction, "ok", False)),
        "error": getattr(prediction, "error", None),
        "columns": list(getattr(prediction, "columns", None) or []),
        "rows": rows,
        "truncated": len(rows) == DISPLAY_ROW_CAP,
        "trace": {
            "linked_schema": getattr(prediction, "linked_schema", ""),
            "repairs": [{"sql": attempt.sql, "error": attempt.error} for attempt in attempts[:-1]],
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(
                estimate_cost_usd(demo.settings.small_model, prompt_tokens, completion_tokens), 6
            ),
            "latency_ms": latency_ms,
        },
        "cached": False,
    }
    demo.cache.put(cache_key, {key: value for key, value in payload.items() if key != "cached"})
    return JSONResponse(payload)


_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
