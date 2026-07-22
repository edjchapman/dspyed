"""The baseline ladder — each rung isolates one design choice.

| name | composition                          | isolates              |
|------|--------------------------------------|-----------------------|
| p0   | Predict(GenerateSQL), full schema    | floor: naive prompting|
| p1   | ChainOfThought(GenerateSQL)          | + reasoning           |
| p2   | linking → CoT (no repair)            | + schema linking      |
| p3   | linking → CoT → execute → repair x2  | + self-repair (full)  |

Every program returns a Prediction carrying a cleaned ``sql`` field, so the
eval harness and metric are uniform across rungs (the metric executes the
SQL itself regardless of whether the program did).
"""

from __future__ import annotations

from collections.abc import Callable

import dspy

from dspyed.data.schema import DBSchema
from dspyed.engine import SafeExecutor
from dspyed.pipeline.modules import TextToSQL, clean_sql
from dspyed.pipeline.signatures import GenerateSQL

PROGRAM_NAMES = ("p0", "p1", "p2", "p3")


class SimpleGenerate(dspy.Module):
    """Bare generation over the full schema — the P0/P1 rungs."""

    def __init__(self, *, use_cot: bool) -> None:
        super().__init__()
        generate_cls = dspy.ChainOfThought if use_cot else dspy.Predict
        self.generate = generate_cls(GenerateSQL)

    def forward(self, question: str, db_schema: str, db_id: str) -> dspy.Prediction:
        raw = self.generate(question=question, db_schema=db_schema).sql
        return dspy.Prediction(sql=clean_sql(raw))


def build_program(
    name: str,
    executor: SafeExecutor,
    schema_lookup: Callable[[str], DBSchema],
) -> dspy.Module:
    match name:
        case "p0":
            return SimpleGenerate(use_cot=False)
        case "p1":
            return SimpleGenerate(use_cot=True)
        case "p2":
            return TextToSQL(executor, schema_lookup, use_linking=True, use_cot=True, max_repairs=0)
        case "p3":
            return TextToSQL(executor, schema_lookup, use_linking=True, use_cot=True, max_repairs=2)
        case _:
            raise ValueError(f"unknown program {name!r}; expected one of {PROGRAM_NAMES}")
