"""Optimizer wrappers with a budget guard and artifact persistence.

Every call signature here was verified against the installed dspy 3.2.1
source (see the plan's Appendix A): BFRS must ALWAYS receive an explicit
valset (None silently reuses the trainset — leakage); MIPROv2 with ``auto``
must NOT receive num_trials; GEPA takes exactly one budget knob and requires
a reflection_lm; its metric must bind five positional arguments.

The budget guard is formula-backed, not vibes-backed: projected program
executions come from each optimizer's own source-derived math, priced with
the measured $/execution from committed baseline results, plus a LARGE-model
allowance for proposer/reflection calls. A start that projects past the
remaining cap is refused.

Artifacts are state-dict JSONs (demos + instructions only). Optimizer extras
(candidate scores, trial logs) are NOT in the state dict, so the interesting
ones are persisted into the compile-run metadata JSON alongside the artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import dspy

from dspyed.config import PRICES_PER_MTOK, Settings
from dspyed.data.schema import render
from dspyed.data.spider import SpiderExample
from dspyed.data.splits import load_split
from dspyed.engine import SafeExecutor
from dspyed.eval.harness import _git_sha, _SchemaCache
from dspyed.eval.metric import ExecutionAccuracy

OPTIMIZERS = ("bfrs", "mipro", "gepa")

# Source-derived projection constants (dspy 3.2.1; plan Appendix A):
_BFRS_CANDIDATES = 8  # num_candidate_programs → C+3 valset evals
_MIPRO_LIGHT_TRIALS = {1: 10, 3: 31}  # int(max(2*(P*2)*log2(6), 9)) per predictor count
_LARGE_CALL_ALLOWANCE = {"bfrs": 0, "mipro": 23, "gepa": 40}


class GepaExecutionMetric:
    """GEPA metric: execution-accuracy score + execution-grounded feedback text.

    Must bind five positional args (GEPA inspects the signature); the score
    with and without ``pred_name`` is identical by construction (deterministic
    metric), satisfying GEPA's consistency rule.
    """

    def __init__(self, executor: SafeExecutor) -> None:
        self._executor = executor
        self._metric = ExecutionAccuracy(executor)

    def _feedback(self, gold_sql: str, db_id: str, pred_sql: str | None, score: float) -> str:
        if not pred_sql or not pred_sql.strip():
            return "No SQL statement was produced."
        pred = self._executor.run(db_id, pred_sql)
        if not pred.ok:
            return f"The SQL failed to execute. Error: {pred.error}"
        if score >= 1.0:
            return "Correct: the query executed and returned the expected result set."
        gold = self._executor.run(db_id, gold_sql)
        message = (
            f"Executed but returned the wrong result: {len(pred.rows)} rows "
            f"vs {len(gold.rows)} expected."
        )
        outcome = self._metric.detailed(gold_sql, db_id, pred_sql)
        if outcome.matched_loose:
            message += (
                " The values match but column order differs — select columns in the order "
                "the question implies."
            )
        return message

    def __call__(
        self,
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> Any:
        gold_sql: str = gold.sql
        db_id: str = gold.db_id
        pred_sql = getattr(pred, "sql", None)
        outcome = self._metric.detailed(gold_sql, db_id, pred_sql)
        if outcome.gold_error is not None:
            return dspy.Prediction(
                score=0.0, feedback="Gold query failed to execute (data regression)."
            )
        return dspy.Prediction(
            score=outcome.score,
            feedback=self._feedback(gold_sql, db_id, pred_sql, outcome.score),
        )


def to_dspy_examples(examples: list[SpiderExample], schemas: _SchemaCache) -> list[dspy.Example]:
    return [
        dspy.Example(
            question=example.question,
            db_schema=render(schemas.schema(example.db_id)),
            db_id=example.db_id,
            sql=example.sql,
        ).with_inputs("question", "db_schema", "db_id")
        for example in examples
    ]


def spent_to_date(results_dir: Path) -> float:
    total = 0.0
    for path in results_dir.glob("*.json"):
        try:
            total += float(json.loads(path.read_text())["summary"]["cost_usd"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return total


def _measured_small_per_execution(results_dir: Path) -> float:
    """Max observed $/example across SMALL baselines — a conservative unit price."""
    prices = []
    for exp in ("E01", "E02", "E03", "E04"):
        path = results_dir / f"{exp}.json"
        if path.exists():
            summary = json.loads(path.read_text())["summary"]
            if summary["n"]:
                prices.append(summary["cost_usd"] / summary["n"])
    return max(prices) if prices else 0.004  # fallback: ~2x the observed worst case


def _large_per_call(settings: Settings) -> float:
    prompt_price, completion_price = PRICES_PER_MTOK[settings.large_model]
    # Proposer/reflection calls are long-form: assume ~6k in / 2k out.
    return (6_000 * prompt_price + 2_000 * completion_price) / 1_000_000


def project_compile_cost(
    optimizer: str,
    settings: Settings,
    *,
    predictors: int,
    valset_size: int,
    trainset_size: int,
    baseline_accuracy: float = 0.75,
) -> float:
    """Projected USD for one compile run (plan Appendix A formulas)."""
    demos_needed = min(4 / max(baseline_accuracy, 0.05), trainset_size)
    if optimizer == "bfrs":
        executions = (_BFRS_CANDIDATES + 3) * valset_size + (_BFRS_CANDIDATES + 1) * demos_needed
    elif optimizer == "mipro":
        trials = _MIPRO_LIGHT_TRIALS.get(predictors, 31)
        executions = valset_size + 35 * trials + valset_size * (trials // 5 + 1) + 6 * demos_needed
    elif optimizer == "gepa":
        trials = _MIPRO_LIGHT_TRIALS.get(predictors, 31)
        executions = valset_size + 30 + 35 * trials + ((trials + 1) // 5 + 1) * valset_size
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")
    small_cost = executions * _measured_small_per_execution(settings.results_dir)
    large_cost = _LARGE_CALL_ALLOWANCE[optimizer] * _large_per_call(settings)
    return small_cost + large_cost


def _build_optimizer(
    optimizer: str, metric: Any, executor: SafeExecutor, settings: Settings
) -> Any:
    from dspy.teleprompt import GEPA, BootstrapFewShotWithRandomSearch, MIPROv2

    if optimizer == "bfrs":
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=_BFRS_CANDIDATES,
            num_threads=8,
            max_errors=100,
        )
    if optimizer == "mipro":
        return MIPROv2(
            metric=metric,
            auto="light",
            prompt_model=dspy.LM(
                settings.large_model, temperature=1.0, max_tokens=settings.max_tokens_cot * 4
            ),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_threads=8,
            max_errors=100,
            seed=9,
        )
    if optimizer == "gepa":
        return GEPA(
            metric=GepaExecutionMetric(executor),
            auto="light",
            reflection_lm=dspy.LM(settings.large_model, temperature=1.0, max_tokens=32_000),
            num_threads=8,
            track_stats=True,
            log_dir=str(settings.artifacts_dir / "gepa_logs"),
            seed=0,
        )
    raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")


def compile_program(
    experiment_id: str,
    optimizer: str,
    program_name: str,
    settings: Settings,
) -> dict[str, Any]:
    """Budget-check, compile, save the artifact + compile-run metadata."""
    import litellm

    from dspyed.pipeline import build_program

    litellm.drop_params = True
    task_lm = dspy.LM(settings.small_model, temperature=0.0, max_tokens=settings.max_tokens_cot)
    dspy.configure(lm=task_lm, track_usage=False)

    database_root = settings.data_root / "database"
    executor = SafeExecutor(
        database_root, timeout_s=settings.exec_timeout_s, max_rows=settings.exec_max_rows
    )
    schemas = _SchemaCache(database_root)
    student = build_program(program_name, executor, schemas.schema)
    predictors = len(student.predictors())

    trainset = to_dspy_examples(
        load_split(settings.data_root, settings.splits_dir, "train_200"), schemas
    )
    valset = to_dspy_examples(
        load_split(settings.data_root, settings.splits_dir, "optim_val_100"), schemas
    )

    projected = project_compile_cost(
        optimizer,
        settings,
        predictors=predictors,
        valset_size=len(valset),
        trainset_size=len(trainset),
    )
    spent = spent_to_date(settings.results_dir)
    if spent + projected > settings.budget_cap_usd:
        raise RuntimeError(
            f"budget guard: projected ${projected:.2f} + spent ${spent:.2f} "
            f"exceeds the ${settings.budget_cap_usd:.0f} cap — refusing to start"
        )

    opt = _build_optimizer(optimizer, ExecutionAccuracy(executor), executor, settings)
    started = time.monotonic()
    compiled = opt.compile(student, trainset=trainset, valset=valset)
    wall_time_s = round(time.monotonic() - started, 1)

    artifact_dir = settings.artifacts_dir / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "program.json"
    compiled.save(str(artifact_path))

    # Optimizer extras are not part of the state dict — persist the useful ones.
    extras: dict[str, Any] = {}
    candidates = getattr(compiled, "candidate_programs", None)
    if candidates:
        extras["candidate_scores"] = [
            candidate.get("score") for candidate in candidates if isinstance(candidate, dict)
        ][:20]
    detailed = getattr(compiled, "detailed_results", None)
    if detailed is not None:
        best = getattr(detailed, "best_idx", None)
        extras["gepa_best_idx"] = best

    metadata = {
        "experiment_id": experiment_id,
        "optimizer": optimizer,
        "program": program_name,
        "task_model": settings.small_model,
        "projected_cost_usd": round(projected, 2),
        "wall_time_s": wall_time_s,
        "git_sha": _git_sha(),
        "dspy_version": dspy.__version__,
        "trainset": "train_200",
        "valset": "optim_val_100",
        "extras": extras,
    }
    (artifact_dir / "compile_run.json").write_text(json.dumps(metadata, indent=1))
    return metadata
