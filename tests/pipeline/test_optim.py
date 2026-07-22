"""Optimizer plumbing — all offline: GEPA metric contract, budget math, artifact eval."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.utils.dummies import DummyLM

import dspyed.eval.harness as harness_module
from dspyed.config import Settings
from dspyed.data.spider import SpiderExample
from dspyed.engine import SafeExecutor
from dspyed.eval.harness import RunSpec, run_eval
from dspyed.optim import GepaExecutionMetric, project_compile_cost
from dspyed.optim.compile import spent_to_date

DB = "mini_singer"


# --- GEPA metric contract ----------------------------------------------------


@pytest.fixture(scope="module")
def gepa_metric(db_root: Path) -> GepaExecutionMetric:
    return GepaExecutionMetric(SafeExecutor(db_root))


def test_binds_five_positional_args(gepa_metric: GepaExecutionMetric):
    # GEPA does exactly this bind check before accepting a metric.
    inspect.signature(gepa_metric).bind(None, None, None, None, None)


def _gold(sql: str) -> SimpleNamespace:
    return SimpleNamespace(sql=sql, db_id=DB)


def test_correct_prediction_feedback(gepa_metric: GepaExecutionMetric):
    result = gepa_metric(
        _gold("SELECT name FROM singer"), SimpleNamespace(sql="SELECT name FROM singer")
    )
    assert result.score == 1.0
    assert "Correct" in result.feedback


def test_execution_error_feedback_carries_verbatim_error(gepa_metric: GepaExecutionMetric):
    result = gepa_metric(_gold("SELECT 1"), SimpleNamespace(sql="SELECT nation FROM singer"))
    assert result.score == 0.0
    assert "no such column" in result.feedback


def test_wrong_result_feedback_carries_row_counts(gepa_metric: GepaExecutionMetric):
    result = gepa_metric(_gold("SELECT name FROM singer"), SimpleNamespace(sql="SELECT 1"))
    assert result.score == 0.0
    assert "1 rows" in result.feedback
    assert "5 expected" in result.feedback


def test_column_order_hint(gepa_metric: GepaExecutionMetric):
    result = gepa_metric(
        _gold("SELECT name, age FROM singer"), SimpleNamespace(sql="SELECT age, name FROM singer")
    )
    assert result.score == 0.0
    assert "column order" in result.feedback


def test_missing_sql_feedback(gepa_metric: GepaExecutionMetric):
    result = gepa_metric(_gold("SELECT 1"), SimpleNamespace())
    assert result.score == 0.0
    assert "No SQL" in result.feedback


def test_score_consistent_with_and_without_pred_name(gepa_metric: GepaExecutionMetric):
    gold, pred = _gold("SELECT name FROM singer"), SimpleNamespace(sql="SELECT name FROM singer")
    assert gepa_metric(gold, pred).score == gepa_metric(gold, pred, None, "generate", None).score


# --- Budget math -------------------------------------------------------------


@pytest.fixture
def priced_results(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    doc = {"summary": {"n": 200, "cost_usd": 0.40}}  # $0.002 / example
    (results / "E03.json").write_text(json.dumps(doc))
    return results


def test_projected_costs_are_sane(priced_results: Path):
    settings = Settings(results_dir=priced_results)
    costs = {
        name: project_compile_cost(name, settings, predictors=3, valset_size=100, trainset_size=200)
        for name in ("bfrs", "mipro", "gepa")
    }
    for name, cost in costs.items():
        assert 0.5 < cost < 25, f"{name} projected ${cost:.2f} — outside sanity band"
    assert costs["bfrs"] < costs["mipro"]  # fewer executions by construction


def test_unknown_optimizer_raises(priced_results: Path):
    with pytest.raises(ValueError, match="unknown optimizer"):
        project_compile_cost(
            "sgd",
            Settings(results_dir=priced_results),
            predictors=1,
            valset_size=10,
            trainset_size=10,
        )


def test_spent_to_date_sums_summaries(priced_results: Path):
    assert spent_to_date(priced_results) == pytest.approx(0.40)
    assert spent_to_date(priced_results / "missing") == 0.0


# --- Artifact-loading eval path ----------------------------------------------


def test_run_eval_loads_artifact(tmp_path: Path, db_root: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "spider"
    db_dir = root / "database" / DB
    db_dir.mkdir(parents=True)
    db_dir.joinpath(f"{DB}.sqlite").write_bytes((db_root / DB / f"{DB}.sqlite").read_bytes())
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    splits_dir.joinpath("dev_eval_200.json").write_text("{}")
    settings = Settings(data_root=root, splits_dir=splits_dir, results_dir=tmp_path / "results")

    examples = [SpiderExample(question="names?", sql="SELECT name FROM singer", db_id=DB)]
    monkeypatch.setattr(harness_module, "load_split", lambda *args: examples)

    # Save an artifact from a customized program, then eval via RunSpec.artifact.
    from dspyed.data.schema import introspect
    from dspyed.pipeline import build_program

    executor = SafeExecutor(root / "database")
    schema = introspect(db_dir / f"{DB}.sqlite", DB)
    donor = build_program("p0", executor, lambda _: schema)
    donor.generate.signature = donor.generate.signature.with_instructions("ARTIFACT MARKER")  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    artifact = tmp_path / "program.json"
    donor.save(str(artifact))

    spec = RunSpec(
        experiment_id="TEST-artifact", program="p0", model="small", artifact=str(artifact)
    )
    results = run_eval(spec, settings, lm=DummyLM([{"sql": "SELECT name FROM singer"}]))
    assert results["summary"]["execution_accuracy"] == 1.0
    assert (tmp_path / "results" / "TEST-artifact.json").exists()
