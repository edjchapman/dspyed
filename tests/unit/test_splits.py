"""Split construction: determinism, db diversity, gold filtering, drift guard."""

import json
import random
from pathlib import Path

import pytest

import dspyed.data.splits as splits_module
from dspyed.data.spider import SPLIT_JSON, SpiderExample, _sha256
from dspyed.data.splits import _round_robin, _select, build_splits, load_split
from dspyed.engine import SafeExecutor


def _examples(*specs: tuple[str, str]) -> list[SpiderExample]:
    return [
        SpiderExample(question=f"q{i}: {sql}", sql=sql, db_id=db_id)
        for i, (db_id, sql) in enumerate(specs)
    ]


SIX = _examples(
    ("mini_singer", "SELECT name FROM singer"),
    ("mini_singer", "SELECT count(*) FROM singer"),
    ("mini_singer", "SELECT venue FROM concert"),
    ("mini_pets", "SELECT name FROM pets"),
    ("mini_pets", "SELECT count(*) FROM pets"),
    ("mini_pets", "SELECT species FROM pets"),
)


def test_round_robin_is_deterministic_per_seed():
    first = list(_round_robin(SIX, random.Random(13)))
    second = list(_round_robin(SIX, random.Random(13)))
    assert first == second
    assert sorted(first) == list(range(6))  # a permutation, nothing lost


def test_round_robin_alternates_databases():
    order = list(_round_robin(SIX, random.Random(13)))
    first_two_dbs = {SIX[index].db_id for index in order[:2]}
    assert first_two_dbs == {"mini_singer", "mini_pets"}


def test_select_filters_failing_and_empty_golds(db_root: Path):
    examples = _examples(
        ("mini_singer", "SELECT name FROM singer"),
        ("mini_singer", "SELECT name FROM singer WHERE 1 = 0"),  # empty result
        ("mini_singer", "SELECT nope FROM singer"),  # execution error
        ("mini_pets", "SELECT name FROM pets"),
    )
    executor = SafeExecutor(db_root)
    # Only the two healthy golds are pickable — filtering is lazy, so demand
    # more than exist: both bad candidates must be visited and skipped.
    with pytest.raises(RuntimeError, match="only 2 candidates available"):
        _select(examples, random.Random(13), 3, executor=executor)

    picked, _ = _select(examples, random.Random(13), 2, executor=executor)
    assert {examples[index].sql for index in picked} == {
        "SELECT name FROM singer",
        "SELECT name FROM pets",
    }


def test_select_raises_when_pool_is_too_small():
    with pytest.raises(RuntimeError, match="wanted 99"):
        _select(SIX, random.Random(13), 99, executor=None)


@pytest.fixture
def tiny_root(tmp_path: Path, db_root: Path) -> Path:
    """A miniature `data/spider` layout backed by the fixture databases."""
    root = tmp_path / "spider"
    (root / "database").mkdir(parents=True)
    for db_id in ("mini_singer", "mini_pets"):
        dest = root / "database" / db_id
        dest.mkdir()
        dest.joinpath(f"{db_id}.sqlite").write_bytes(
            (db_root / db_id / f"{db_id}.sqlite").read_bytes()
        )
    for split, examples in {"train": SIX, "dev": SIX[:4]}.items():
        records = [{"question": e.question, "query": e.sql, "db_id": e.db_id} for e in examples]
        (root / SPLIT_JSON[split]).write_text(json.dumps(records))
    manifest = {"sha256": {s: _sha256(root / SPLIT_JSON[s]) for s in SPLIT_JSON}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return root


def test_build_and_load_roundtrip(tiny_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(splits_module, "TRAIN_SIZE", 3)
    monkeypatch.setattr(splits_module, "VAL_SIZE", 2)
    monkeypatch.setattr(splits_module, "DEV_EVAL_SIZE", 2)
    splits_dir = tiny_root / "splits"

    summary = build_splits(tiny_root, splits_dir, seed=13)
    assert summary["sizes"] == {"train_200": 3, "optim_val_100": 2, "dev_eval_200": 2}

    train = load_split(tiny_root, splits_dir, "train_200")
    val = load_split(tiny_root, splits_dir, "optim_val_100")
    assert not {e.question for e in train} & {e.question for e in val}  # disjoint

    dev = load_split(tiny_root, splits_dir, "dev_eval_200")
    assert {e.db_id for e in dev} == {"mini_singer", "mini_pets"}  # spans databases


def test_load_split_detects_source_drift(tiny_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(splits_module, "TRAIN_SIZE", 3)
    monkeypatch.setattr(splits_module, "VAL_SIZE", 2)
    monkeypatch.setattr(splits_module, "DEV_EVAL_SIZE", 2)
    splits_dir = tiny_root / "splits"
    build_splits(tiny_root, splits_dir, seed=13)

    records = json.loads((tiny_root / SPLIT_JSON["dev"]).read_text())
    records[0]["question"] = "tampered"
    (tiny_root / SPLIT_JSON["dev"]).write_text(json.dumps(records))

    with pytest.raises(RuntimeError, match="drifted"):
        load_split(tiny_root, splits_dir, "dev_eval_200")
