"""Phase 1 acceptance: gold-vs-gold execution accuracy = 100% on dev_eval_200.

If the measuring instruments (executor + comparison + metric) cannot score
the ground truth perfectly, every downstream number is noise. This runs only
when the real dataset is present locally (`make data splits`); CI skips it.
"""

from pathlib import Path

import pytest

from dspyed.data.splits import load_split
from dspyed.engine import SafeExecutor
from dspyed.eval import ExecutionAccuracy

ROOT = Path("data/spider")
SPLITS_DIR = Path("data/splits")

pytestmark = pytest.mark.skipif(
    not (ROOT / "MANIFEST.json").exists() or not (SPLITS_DIR / "dev_eval_200.json").exists(),
    reason="Spider dataset/splits not downloaded (run `make data splits`)",
)


def test_gold_vs_gold_is_100_percent():
    examples = load_split(ROOT, SPLITS_DIR, "dev_eval_200")
    assert len(examples) == 200

    metric = ExecutionAccuracy(SafeExecutor(ROOT / "database"))
    outcomes = [metric.detailed(e.sql, e.db_id, e.sql) for e in examples]

    gold_errors = [
        (e.db_id, o.gold_error) for e, o in zip(examples, outcomes, strict=True) if o.gold_error
    ]
    assert gold_errors == []

    score = sum(o.score for o in outcomes)
    assert score == 200.0, f"gold-vs-gold only {score}/200 — instruments are broken"


def test_train_splits_have_zero_gold_errors():
    executor = SafeExecutor(ROOT / "database")
    metric = ExecutionAccuracy(executor)
    for name in ("train_200", "optim_val_100"):
        examples = load_split(ROOT, SPLITS_DIR, name)
        outcomes = [metric.detailed(e.sql, e.db_id, e.sql) for e in examples]
        assert all(o.score == 1.0 for o in outcomes), f"{name}: gold self-match failed"
