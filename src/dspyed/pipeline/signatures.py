"""Task definitions — the "what", separated from prompts (the "how").

Docstrings and field descriptions here are the *seed* instructions; the
optimizers rewrite/extend what surrounds them. Field name ``db_schema`` (not
``schema``) is deliberate: DSPy signatures are pydantic models, and ``schema``
shadows a ``BaseModel`` attribute.
"""

from __future__ import annotations

import dspy


class LinkSchema(dspy.Signature):
    """Identify which tables and columns of the database are needed to answer the question."""

    question: str = dspy.InputField()
    db_schema: str = dspy.InputField(desc="Full schema: tables, columns with types, PK/FK links")
    relevant_columns: list[str] = dspy.OutputField(
        desc='Needed columns as "table.column"; include join-key columns'
    )


class GenerateSQL(dspy.Signature):
    """Write one SQLite SELECT query that answers the question using only the given schema.

    Output only SQL — no markdown fences, no explanation.
    """

    question: str = dspy.InputField()
    db_schema: str = dspy.InputField(desc="Relevant schema: tables, columns, types, PK/FK links")
    sql: str = dspy.OutputField(desc="A single SQLite-dialect SELECT statement")


class RepairSQL(dspy.Signature):
    """The SQL failed to execute on the SQLite database.

    Produce a corrected query that answers the question. Change only what is
    necessary to fix the error.
    """

    question: str = dspy.InputField()
    db_schema: str = dspy.InputField()
    failed_sql: str = dspy.InputField()
    error: str = dspy.InputField(desc="Verbatim SQLite error message or safety-rejection reason")
    sql: str = dspy.OutputField(desc="The corrected single SQLite SELECT statement")
