"""Schema introspection + rendering: golden strings and degradation rules."""

import sqlite3
from pathlib import Path

import pytest

from dspyed.data.schema import introspect, render, render_subset

GOLDEN_FULL = (
    "Table singer (singer_id INTEGER PK, name TEXT, country TEXT, age INTEGER, net_worth REAL)\n"
    "Table concert (concert_id INTEGER PK, venue TEXT, singer_id INTEGER)\n"
    "FK: concert.singer_id -> singer.singer_id"
)


@pytest.fixture(scope="module")
def mini_schema(db_root: Path):
    return introspect(db_root / "mini_singer" / "mini_singer.sqlite", "mini_singer")


def test_full_render_matches_golden(mini_schema):
    assert render(mini_schema) == GOLDEN_FULL


def test_subset_pulls_whole_table_and_drops_dangling_fks(mini_schema):
    rendered = render_subset(mini_schema, ["singer.name"], fallback=GOLDEN_FULL)
    assert "Table singer" in rendered
    assert "net_worth REAL" in rendered  # whole table, not just the named column
    assert "concert" not in rendered
    assert "FK:" not in rendered  # FK endpoint missing → line dropped


def test_subset_keeps_fk_when_both_endpoints_present(mini_schema):
    rendered = render_subset(
        mini_schema, ["singer.singer_id", "concert.venue"], fallback=GOLDEN_FULL
    )
    assert "FK: concert.singer_id -> singer.singer_id" in rendered


def test_subset_is_case_insensitive(mini_schema):
    rendered = render_subset(mini_schema, ["SINGER.NAME"], fallback=GOLDEN_FULL)
    assert "Table singer" in rendered
    assert "concert" not in rendered


@pytest.mark.parametrize("selection", [[], ["unknown.x"], ["not-even-dotted"]])
def test_subset_falls_back_on_garbage(mini_schema, selection):
    assert render_subset(mini_schema, selection, fallback="FALLBACK") == "FALLBACK"


def test_wide_table_is_capped(tmp_path: Path):
    db_path = tmp_path / "wide.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        columns = ", ".join(f"c{i} INTEGER" for i in range(30))
        conn.execute(f"CREATE TABLE wide ({columns})")
        conn.commit()
    finally:
        conn.close()
    schema = introspect(db_path, "wide")
    rendered = render(schema)
    assert "(+5 more columns)" in rendered
    assert "c24 INTEGER" in rendered
    assert "c25" not in rendered


def test_char_budget_degrades_gracefully(mini_schema):
    typeless = render(mini_schema, char_budget=160)
    assert "INTEGER" not in typeless  # types dropped first...
    assert "Table singer" in typeless  # ...but every table still present

    tiny = render(mini_schema, char_budget=80)
    assert "(+1 more tables)" in tiny  # then widest tables drop, with a marker
    assert "singer_id" in tiny  # concert (narrower) survives
    assert "FK:" not in tiny  # a dropped table takes its FK lines with it
