"""Seeded, committed example-id splits.

The leakage rule (a headline rigor point): optimizer train and val both come
from Spider TRAIN; the dev set is touched only by evaluation.

| split          | source | size | gold filter              |
|----------------|--------|------|--------------------------|
| train_200      | train  | 200  | executes, returns ≥1 row |
| optim_val_100  | train  | 100  | executes, returns ≥1 row |
| dev_eval_200   | dev    | 200  | none (never filtered)    |

Selection is a seeded round-robin over db_id groups so every split spans many
databases. Train-side candidates whose gold SQL fails or returns zero rows
are skipped (empty-result golds make weak bootstrap demos); the two train
splits are disjoint by construction (one 300-example stream, sliced 200/100).

The committed files carry (idx, db_id, sha1(question)) per example: loaders
re-assert the hash against the source JSON, so silent upstream drift fails
loudly instead of quietly changing the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Final

from dspyed.data.spider import SpiderExample, load_examples, verify_manifest
from dspyed.engine import SafeExecutor

TRAIN_SIZE: Final = 200  # optimizer trainset
VAL_SIZE: Final = 100  # optimizer internal valset (disjoint: one pool, sliced)
DEV_EVAL_SIZE: Final = 200


def _question_sha1(question: str) -> str:
    return hashlib.sha1(question.encode()).hexdigest()  # noqa: S324 — drift guard, not crypto


def _round_robin(examples: list[SpiderExample], rng: random.Random) -> Iterator[int]:
    """Yield original indices, cycling across db_id groups (seeded order)."""
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.db_id, []).append(index)
    keys = sorted(groups)  # sort first: iteration order must not depend on file order
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                yield groups[key].pop()


def _gold_returns_rows(executor: SafeExecutor, example: SpiderExample) -> bool:
    result = executor.run(example.db_id, example.sql)
    return result.ok and len(result.rows) > 0


def _select(
    examples: list[SpiderExample],
    rng: random.Random,
    size: int,
    *,
    executor: SafeExecutor | None,
) -> tuple[list[int], int]:
    """Pick ``size`` indices round-robin; filter via executor when given."""
    picked: list[int] = []
    skipped = 0
    for index in _round_robin(examples, rng):
        if executor is not None and not _gold_returns_rows(executor, examples[index]):
            skipped += 1
            continue
        picked.append(index)
        if len(picked) == size:
            break
    if len(picked) < size:
        raise RuntimeError(f"only {len(picked)} candidates available, wanted {size}")
    return picked, skipped


def _split_record(
    examples: list[SpiderExample],
    indices: list[int],
    *,
    source_split: str,
    seed: int,
    gold_filter: str,
) -> dict[str, object]:
    return {
        "dataset": "spider",
        "source_split": source_split,
        "seed": seed,
        "created": date.today().isoformat(),
        "filter": gold_filter,
        "ids": [
            {
                "idx": index,
                "db_id": examples[index].db_id,
                "question_sha1": _question_sha1(examples[index].question),
            }
            for index in indices
        ],
    }


def build_splits(root: Path, splits_dir: Path, *, seed: int = 13) -> dict[str, object]:
    """Build and write all three split files. Returns a summary."""
    verify_manifest(root)
    executor = SafeExecutor(root / "database")
    rng = random.Random(seed)  # noqa: S311 — seeded, reproducible sampling; not crypto

    train = load_examples(root, "train")
    pool, skipped = _select(train, rng, TRAIN_SIZE + VAL_SIZE, executor=executor)

    dev = load_examples(root, "dev")
    dev_picked, _ = _select(dev, rng, DEV_EVAL_SIZE, executor=None)

    splits_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train_200": _split_record(
            train,
            pool[:TRAIN_SIZE],
            source_split="train",
            seed=seed,
            gold_filter="gold_executes_nonempty",
        ),
        "optim_val_100": _split_record(
            train,
            pool[TRAIN_SIZE:],
            source_split="train",
            seed=seed,
            gold_filter="gold_executes_nonempty",
        ),
        "dev_eval_200": _split_record(
            dev, dev_picked, source_split="dev", seed=seed, gold_filter="none"
        ),
    }
    for name, record in files.items():
        (splits_dir / f"{name}.json").write_text(json.dumps(record, indent=1))

    return {
        "seed": seed,
        "train_skipped_by_filter": skipped,
        "sizes": {name: len(record["ids"]) for name, record in files.items()},  # type: ignore[arg-type]
        "distinct_dbs": {
            name: len({entry["db_id"] for entry in record["ids"]})  # type: ignore[index, union-attr]
            for name, record in files.items()
        },
    }


def load_split(root: Path, splits_dir: Path, name: str) -> list[SpiderExample]:
    """Load a committed split, re-asserting question hashes against the source."""
    record = json.loads((splits_dir / f"{name}.json").read_text())
    source = load_examples(root, record["source_split"])
    examples: list[SpiderExample] = []
    for entry in record["ids"]:
        example = source[entry["idx"]]
        if _question_sha1(example.question) != entry["question_sha1"]:
            raise RuntimeError(
                f"split {name!r} idx {entry['idx']} drifted from the source dataset — "
                "the committed split no longer matches what `dspyed download` fetched"
            )
        examples.append(example)
    return examples
