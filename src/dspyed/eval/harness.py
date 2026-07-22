"""Eval harness: run a program over a split, persist a self-describing results JSON.

Every results file must be reproducible evidence, so it records provenance
(git SHA, dspy version, model id, temperature, split SHA256, the price table
used) alongside per-example records and the summary. The README results
table is regenerated from these files — numbers never live in prose.

Orchestration only — scoring stays in metric.py (strict-typed); this module
touches dspy and is deliberately outside the strict boundary.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dspy

from dspyed.config import Settings, estimate_cost_usd
from dspyed.data.schema import DBSchema, introspect, render
from dspyed.data.spider import _sha256
from dspyed.data.splits import load_split
from dspyed.engine import SafeExecutor
from dspyed.eval.metric import ExecutionAccuracy
from dspyed.pipeline import build_program


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    program: str  # p0 | p1 | p2 | p3
    model: str  # "small" | "large"
    split: str = "dev_eval_200"
    limit: int | None = None
    temperature: float = 0.0


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _usage_tokens(prediction: object) -> tuple[int, int]:
    """Best-effort (prompt, completion) token counts from a Prediction."""
    get_usage = getattr(prediction, "get_lm_usage", None)
    if get_usage is None:
        return (0, 0)
    usage = get_usage() or {}
    prompt = sum(int(v.get("prompt_tokens", 0)) for v in usage.values())
    completion = sum(int(v.get("completion_tokens", 0)) for v in usage.values())
    return (prompt, completion)


class _SchemaCache:
    def __init__(self, database_root: Path) -> None:
        self._root = database_root
        self._cache: dict[str, DBSchema] = {}

    def schema(self, db_id: str) -> DBSchema:
        if db_id not in self._cache:
            self._cache[db_id] = introspect(self._root / db_id / f"{db_id}.sqlite", db_id)
        return self._cache[db_id]

    def rendered(self, db_id: str) -> str:
        return render(self.schema(db_id))


def run_eval(spec: RunSpec, settings: Settings, *, lm: object | None = None) -> dict[str, Any]:
    """Run one experiment; returns the results document (also written to disk).

    ``lm`` overrides the model (tests inject DummyLM); by default the spec's
    tier is resolved through Settings and DSPy's disk cache stays on.
    """
    model_id = settings.small_model if spec.model == "small" else settings.large_model
    if lm is None:
        lm = dspy.LM(model_id, temperature=spec.temperature, max_tokens=settings.max_tokens_cot)
    dspy.configure(lm=lm, track_usage=True)

    database_root = settings.data_root / "database"
    executor = SafeExecutor(
        database_root, timeout_s=settings.exec_timeout_s, max_rows=settings.exec_max_rows
    )
    schemas = _SchemaCache(database_root)
    metric = ExecutionAccuracy(executor)
    program = build_program(spec.program, executor, schemas.schema)

    examples = load_split(settings.data_root, settings.splits_dir, spec.split)
    if spec.limit is not None:
        examples = examples[: spec.limit]

    records: list[dict[str, Any]] = []  # dspy-facing orchestration: Any is honest here
    started = time.monotonic()
    for example in examples:
        example_started = time.monotonic()
        try:
            prediction = program(
                question=example.question,
                db_schema=schemas.rendered(example.db_id),
                db_id=example.db_id,
            )
            pred_sql = getattr(prediction, "sql", None)
            failure: str | None = None
        except Exception as err:  # noqa: BLE001 — a crashed example scores 0, run continues
            prediction, pred_sql, failure = None, None, f"{type(err).__name__}: {err}"
        latency_ms = (time.monotonic() - example_started) * 1000

        outcome = metric.detailed(example.sql, example.db_id, pred_sql)
        prompt_tokens, completion_tokens = _usage_tokens(prediction) if prediction else (0, 0)
        records.append(
            {
                "db_id": example.db_id,
                "question": example.question,
                "gold_sql": example.sql,
                "pred_sql": pred_sql,
                "program_failure": failure,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "n_attempts": len(getattr(prediction, "attempts", []) or []) if prediction else 0,
                **asdict(outcome),
            }
        )

    total = len(records)
    prompt_total = sum(int(r["prompt_tokens"]) for r in records)
    completion_total = sum(int(r["completion_tokens"]) for r in records)
    results: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "spec": asdict(spec),
        "provenance": {
            "git_sha": _git_sha(),
            "dspy_version": dspy.__version__,
            "model_id": model_id if lm is None or isinstance(lm, dspy.LM) else "dummy",
            "split_sha256": _sha256(settings.splits_dir / f"{spec.split}.json"),
            "prices_per_mtok": "see dspyed.config.PRICES_PER_MTOK at git_sha",
        },
        "summary": {
            "n": total,
            "execution_accuracy": round(sum(float(r["score"]) for r in records) / total, 4)
            if total
            else 0.0,
            "valid_sql_rate": round(
                sum(1 for r in records if r["error_class"] is None and r["program_failure"] is None)
                / total,
                4,
            )
            if total
            else 0.0,
            "matched_loose_rate": round(sum(1 for r in records if r["matched_loose"]) / total, 4)
            if total
            else 0.0,
            "both_empty_rate": round(sum(1 for r in records if r["both_empty"]) / total, 4)
            if total
            else 0.0,
            "error_classes": {
                cls: sum(1 for r in records if r["error_class"] == cls)
                for cls in ("rejected", "syntax", "schema", "timeout", "other")
            },
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "cost_usd": round(estimate_cost_usd(model_id, prompt_total, completion_total), 4),
            "wall_time_s": round(time.monotonic() - started, 1),
        },
        "records": records,
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.results_dir / f"{spec.experiment_id}.json"
    out_path.write_text(json.dumps(results, indent=1))
    return results
