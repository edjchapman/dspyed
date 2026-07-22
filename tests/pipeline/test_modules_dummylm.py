"""Pipeline behavior under DummyLM — fully offline, deterministic.

DummyLM's sequential mode returns canned outputs in order, which lets each
test script an exact conversation: link → generate → (repair …). The
executor and databases are real; only the LM is fake.
"""

from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from dspyed.data.schema import DBSchema, introspect, render
from dspyed.engine import SafeExecutor
from dspyed.pipeline import TextToSQL, build_program, clean_sql

DB = "mini_singer"


# --- clean_sql ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1;  ", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1;\n```", "SELECT 1"),
        ("Here is the query:\nSELECT name FROM singer", "SELECT name FROM singer"),
        ("SELECT 1; DROP TABLE singer;", "SELECT 1"),
        ("SELECT ';' AS semi FROM singer", "SELECT ';' AS semi FROM singer"),
        ("WITH t AS (SELECT 1) SELECT * FROM t", "WITH t AS (SELECT 1) SELECT * FROM t"),
        ("no sql here at all", "no sql here at all"),
    ],
)
def test_clean_sql(raw: str, expected: str):
    assert clean_sql(raw) == expected


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def schema(db_root: Path) -> DBSchema:
    return introspect(db_root / DB / f"{DB}.sqlite", DB)


@pytest.fixture(scope="module")
def rendered(schema: DBSchema) -> str:
    return render(schema)


@pytest.fixture
def executor(db_root: Path) -> SafeExecutor:
    return SafeExecutor(db_root)


def _pipeline(executor: SafeExecutor, schema: DBSchema, *, max_repairs: int = 2) -> TextToSQL:
    return TextToSQL(
        executor,
        lambda db_id: schema,
        use_linking=True,
        use_cot=False,  # Predict keeps DummyLM scripts to exactly one field per turn
        max_repairs=max_repairs,
    )


# --- TextToSQL control flow --------------------------------------------------


def test_happy_path_single_attempt(executor: SafeExecutor, schema: DBSchema, rendered: str):
    lm = DummyLM(
        [
            {"relevant_columns": ["singer.name"]},
            {"sql": "SELECT name FROM singer"},
        ]
    )
    with dspy.context(lm=lm):
        prediction = _pipeline(executor, schema)(question="names?", db_schema=rendered, db_id=DB)

    assert prediction.ok is True
    assert prediction.sql == "SELECT name FROM singer"
    assert len(prediction.attempts) == 1
    assert prediction.attempts[0].error is None
    assert "Table singer" in prediction.linked_schema
    assert "concert" not in prediction.linked_schema  # linking narrowed the schema


def test_repair_fires_on_execution_error(executor: SafeExecutor, schema: DBSchema, rendered: str):
    lm = DummyLM(
        [
            {"relevant_columns": ["singer.name"]},
            {"sql": "SELECT nation FROM singer"},  # no such column
            {"sql": "SELECT country FROM singer"},  # the repair
        ]
    )
    with dspy.context(lm=lm):
        prediction = _pipeline(executor, schema)(
            question="countries?", db_schema=rendered, db_id=DB
        )

    assert prediction.ok is True
    assert prediction.sql == "SELECT country FROM singer"
    assert len(prediction.attempts) == 2
    assert "no such column" in prediction.attempts[0].error
    assert prediction.attempts[1].error is None


def test_repair_stops_at_cap(executor: SafeExecutor, schema: DBSchema, rendered: str):
    lm = DummyLM(
        [
            {"relevant_columns": ["singer.name"]},
            {"sql": "SELECT nation FROM singer"},
            {"sql": "SELECT still_wrong FROM singer"},
            {"sql": "SELECT wrong_again FROM singer"},
        ]
    )
    with dspy.context(lm=lm):
        prediction = _pipeline(executor, schema, max_repairs=2)(
            question="?", db_schema=rendered, db_id=DB
        )

    assert prediction.ok is False
    assert len(prediction.attempts) == 3  # initial + 2 repairs, then stop
    assert prediction.error is not None


def test_garbage_linker_output_degrades_to_full_schema(
    executor: SafeExecutor, schema: DBSchema, rendered: str
):
    lm = DummyLM(
        [
            {"relevant_columns": ["nonsense.blah"]},
            {"sql": "SELECT name FROM singer"},
        ]
    )
    with dspy.context(lm=lm):
        prediction = _pipeline(executor, schema)(question="?", db_schema=rendered, db_id=DB)

    assert prediction.linked_schema == rendered  # fallback, never an empty schema
    assert prediction.ok is True


def test_no_repair_variant_stops_after_first_failure(
    executor: SafeExecutor, schema: DBSchema, rendered: str
):
    lm = DummyLM(
        [
            {"relevant_columns": ["singer.name"]},
            {"sql": "SELECT nation FROM singer"},
        ]
    )
    pipeline = TextToSQL(executor, lambda _: schema, use_linking=True, use_cot=False, max_repairs=0)
    with dspy.context(lm=lm):
        prediction = pipeline(question="?", db_schema=rendered, db_id=DB)

    assert prediction.ok is False
    assert len(prediction.attempts) == 1


# --- Baseline ladder ---------------------------------------------------------


def test_p0_returns_cleaned_sql(executor: SafeExecutor, schema: DBSchema, rendered: str):
    program = build_program("p0", executor, lambda _: schema)
    lm = DummyLM([{"sql": "```sql\nSELECT count(*) FROM singer;\n```"}])
    with dspy.context(lm=lm):
        prediction = program(question="how many?", db_schema=rendered, db_id=DB)
    assert prediction.sql == "SELECT count(*) FROM singer"


def test_build_program_rejects_unknown_name(executor: SafeExecutor, schema: DBSchema):
    with pytest.raises(ValueError, match="unknown program"):
        build_program("p9", executor, lambda _: schema)


# --- Persistence -------------------------------------------------------------


def test_save_load_round_trip(
    tmp_path: Path, executor: SafeExecutor, schema: DBSchema, rendered: str
):
    program = _pipeline(executor, schema)
    # dspy's stubs don't model Predict.signature mutation (the optimizers do
    # exactly this at compile time — it's the mechanism artifacts persist).
    program.generate.signature = program.generate.signature.with_instructions(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        "CUSTOMIZED INSTRUCTIONS FOR ROUND-TRIP TEST"
    )
    path = tmp_path / "program.json"
    program.save(path)

    restored = _pipeline(executor, schema)
    restored.load(path)

    assert "CUSTOMIZED INSTRUCTIONS" in restored.generate.signature.instructions  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    lm = DummyLM([{"relevant_columns": ["singer.name"]}, {"sql": "SELECT name FROM singer"}])
    with dspy.context(lm=lm):
        prediction = restored(question="names?", db_schema=rendered, db_id=DB)
    assert prediction.ok is True
