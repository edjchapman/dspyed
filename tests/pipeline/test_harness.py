"""Harness behavior under DummyLM: results JSON shape, scoring, provenance."""

import json
from pathlib import Path

import pytest
from dspy.utils.dummies import DummyLM

import dspyed.eval.harness as harness_module
from dspyed.config import Settings
from dspyed.data.spider import SpiderExample
from dspyed.eval.harness import RunSpec, run_eval


@pytest.fixture
def fake_spider(tmp_path: Path, db_root: Path) -> Settings:
    root = tmp_path / "spider"
    db_dir = root / "database" / "mini_singer"
    db_dir.mkdir(parents=True)
    db_dir.joinpath("mini_singer.sqlite").write_bytes(
        (db_root / "mini_singer" / "mini_singer.sqlite").read_bytes()
    )
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    splits_dir.joinpath("dev_eval_200.json").write_text("{}")  # provenance hash target only
    return Settings(data_root=root, splits_dir=splits_dir, results_dir=tmp_path / "results")


def test_run_eval_scores_and_persists(fake_spider: Settings, monkeypatch: pytest.MonkeyPatch):
    examples = [
        SpiderExample(question="names?", sql="SELECT name FROM singer", db_id="mini_singer"),
        SpiderExample(question="count?", sql="SELECT count(*) FROM singer", db_id="mini_singer"),
    ]
    monkeypatch.setattr(harness_module, "load_split", lambda *args: examples)

    lm = DummyLM(
        [
            {"sql": "SELECT name FROM singer"},  # correct
            {"sql": "SELECT 999"},  # wrong result
        ]
    )
    spec = RunSpec(experiment_id="TEST-smoke", program="p0", model="small", limit=None)
    results = run_eval(spec, fake_spider, lm=lm)

    summary = results["summary"]
    assert summary["n"] == 2
    assert summary["execution_accuracy"] == 0.5
    assert summary["valid_sql_rate"] == 1.0  # both executed, one was just wrong

    persisted = json.loads((fake_spider.results_dir / "TEST-smoke.json").read_text())
    assert persisted["summary"] == summary
    assert persisted["provenance"]["dspy_version"]
    record = persisted["records"][0]
    for key in ("db_id", "gold_sql", "pred_sql", "score", "latency_ms", "error_class"):
        assert key in record


def test_run_eval_survives_program_crash(fake_spider: Settings, monkeypatch: pytest.MonkeyPatch):
    examples = [
        SpiderExample(question="names?", sql="SELECT name FROM singer", db_id="mini_singer")
    ]
    monkeypatch.setattr(harness_module, "load_split", lambda *args: examples)

    def explode(*args, **kwargs):
        raise RuntimeError("LM meltdown")

    monkeypatch.setattr(harness_module, "build_program", lambda *a: explode)

    results = run_eval(
        RunSpec(experiment_id="TEST-crash", program="p0", model="small"),
        fake_spider,
        lm=DummyLM([{"sql": "unused"}]),
    )
    record = results["records"][0]
    assert record["score"] == 0.0
    assert "LM meltdown" in record["program_failure"]
    assert results["summary"]["valid_sql_rate"] == 0.0
