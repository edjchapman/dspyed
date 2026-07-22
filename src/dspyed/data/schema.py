"""Schema introspection and rendering.

Schemas are introspected directly from the SQLite files (PRAGMA table_info /
foreign_key_list) rather than from Spider's tables.json — the schema shown to
the LM is derived from the exact file its SQL will execute against. One
source of truth, no drift.

Render format (compact, unambiguous, pinned by golden tests):

    Table singer (singer_id INTEGER PK, name TEXT, country TEXT)
    Table concert (concert_id INTEGER PK, venue TEXT, singer_id INTEGER)
    FK: concert.singer_id -> singer.singer_id

Rendering rules:
- Declared column types are uppercased and truncated at "(" (VARCHAR(100) →
  VARCHAR); an empty declared type renders as ANY.
- FK lines appear only when BOTH endpoint tables are rendered.
- ``render_subset`` includes the WHOLE table when any of its columns is
  selected (join keys and PKs never go missing), ignores unknown names, and
  falls back to the full render when the selection resolves to nothing —
  a garbage linker output degrades to "no linking", never to "no schema".
- Wide tables are capped with a ``(+N more columns)`` marker; a total
  character budget degrades gracefully (drop types first, then drop trailing
  tables with a marker) so a pathological schema cannot blow the context.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_COLS_PER_TABLE: Final = 25
CHAR_BUDGET: Final = 6_000


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_keys: tuple[str, ...]


@dataclass(frozen=True)
class DBSchema:
    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...]

    def table_names(self) -> set[str]:
        return {table.name.lower() for table in self.tables}


def _normalize_type(declared: str) -> str:
    cleaned = declared.strip().upper().split("(")[0].strip()
    return cleaned or "ANY"


def _quote_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def introspect(db_path: Path, db_id: str) -> DBSchema:
    """Read the schema from a SQLite file (read-only, no side effects)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        tables: list[Table] = []
        foreign_keys: list[ForeignKey] = []
        pk_by_table: dict[str, tuple[str, ...]] = {}

        for name in names:
            info = conn.execute(f"PRAGMA table_info({_quote_ident(name)})").fetchall()
            columns = tuple(
                Column(name=str(row[1]), type=_normalize_type(str(row[2]))) for row in info
            )
            pks = tuple(str(row[1]) for row in sorted(info, key=lambda r: r[5]) if row[5] > 0)
            tables.append(Table(name=name, columns=columns, primary_keys=pks))
            pk_by_table[name] = pks

        for name in names:
            for row in conn.execute(f"PRAGMA foreign_key_list({_quote_ident(name)})"):
                ref_table, from_col, to_col = str(row[2]), str(row[3]), row[4]
                if to_col is None:
                    # Implicit reference to the target's primary key.
                    ref_pks = pk_by_table.get(ref_table, ())
                    to_col = ref_pks[0] if ref_pks else "rowid"
                foreign_keys.append(
                    ForeignKey(
                        table=name, column=from_col, ref_table=ref_table, ref_column=str(to_col)
                    )
                )

        return DBSchema(db_id=db_id, tables=tuple(tables), foreign_keys=tuple(foreign_keys))
    finally:
        conn.close()


def _render_table(table: Table, *, with_types: bool) -> str:
    shown = table.columns[:MAX_COLS_PER_TABLE]
    hidden = len(table.columns) - len(shown)
    parts: list[str] = []
    for column in shown:
        piece = column.name if not with_types else f"{column.name} {column.type}"
        if column.name in table.primary_keys:
            piece += " PK"
        parts.append(piece)
    if hidden > 0:
        parts.append(f"(+{hidden} more columns)")
    return f"Table {table.name} ({', '.join(parts)})"


def _render(
    tables: tuple[Table, ...],
    foreign_keys: tuple[ForeignKey, ...],
    *,
    char_budget: int,
) -> str:
    def build(*, with_types: bool, table_subset: tuple[Table, ...]) -> str:
        # FK lines are derived from the subset actually rendered — a table
        # dropped by budget degradation must take its FK lines with it.
        rendered_names = {table.name.lower() for table in table_subset}
        fk_lines = [
            f"FK: {fk.table}.{fk.column} -> {fk.ref_table}.{fk.ref_column}"
            for fk in foreign_keys
            if fk.table.lower() in rendered_names and fk.ref_table.lower() in rendered_names
        ]
        lines = [_render_table(table, with_types=with_types) for table in table_subset]
        dropped = len(tables) - len(table_subset)
        if dropped > 0:
            lines.append(f"(+{dropped} more tables)")
        return "\n".join([*lines, *fk_lines])

    full = build(with_types=True, table_subset=tables)
    if len(full) <= char_budget:
        return full

    typeless = build(with_types=False, table_subset=tables)
    if len(typeless) <= char_budget:
        return typeless

    # Still over: drop widest tables (descending column count) until it fits.
    keep = list(tables)
    while len(keep) > 1:
        widest = max(keep, key=lambda table: len(table.columns))
        keep.remove(widest)
        candidate = build(with_types=False, table_subset=tuple(keep))
        if len(candidate) <= char_budget:
            return candidate
    return build(with_types=False, table_subset=tuple(keep))


def render(schema: DBSchema, *, char_budget: int = CHAR_BUDGET) -> str:
    """Full schema render, within the character budget."""
    return _render(schema.tables, schema.foreign_keys, char_budget=char_budget)


def render_subset(
    schema: DBSchema,
    selected_columns: list[str],
    *,
    fallback: str,
    char_budget: int = CHAR_BUDGET,
) -> str:
    """Render only the tables touched by ``selected_columns`` ("table.column").

    Unknown identifiers are ignored; an empty resolution returns ``fallback``
    (normally the full render) — linker garbage can only widen the schema,
    never lose it.
    """
    known = schema.table_names()
    wanted: set[str] = set()
    for identifier in selected_columns:
        table = identifier.split(".", 1)[0].strip().lower()
        if table in known:
            wanted.add(table)
    if not wanted:
        return fallback
    tables = tuple(table for table in schema.tables if table.name.lower() in wanted)
    return _render(tables, schema.foreign_keys, char_budget=char_budget)
