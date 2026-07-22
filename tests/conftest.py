"""Global test configuration: the zero-network guarantee.

Every test runs with sockets disabled unless it explicitly opts in with the
`llm` marker — CI runs `pytest -m "not llm"` with no secrets configured, so a
test that tries to reach the network fails loudly instead of silently
depending on it.
"""

import sqlite3
from pathlib import Path

import pytest
from pytest_socket import disable_socket, enable_socket


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "llm" in item.keywords:
        enable_socket()
    else:
        disable_socket()


# --- Fixture databases -------------------------------------------------------
# A "mini-spider": built from checked-in DDL at session start (no binaries in
# git), laid out exactly like Spider (<root>/<db_id>/<db_id>.sqlite) so the
# executor's path contract is exercised for real.

_MINI_SINGER_DDL = """
CREATE TABLE singer (
    singer_id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT,
    age INTEGER,
    net_worth REAL
);
CREATE TABLE concert (
    concert_id INTEGER PRIMARY KEY,
    venue TEXT,
    singer_id INTEGER REFERENCES singer(singer_id)
);
INSERT INTO singer VALUES
    (1, 'Ava',  'UK',   30, 1000000.5),
    (2, 'Ben',  'US',   45, 250000.0),
    (3, 'Caro', NULL,   30, NULL),
    (4, 'Ben',  'FR',   61, 90000.25),
    (5, 'Dana', 'UK',   19, 12.3456789);
INSERT INTO concert VALUES
    (10, 'Roundhouse', 1),
    (11, 'Barbican',   1),
    (12, 'Palais',     4);
"""

_MINI_PETS_DDL = """
CREATE TABLE pets (
    pet_id INTEGER PRIMARY KEY,
    name TEXT,
    species TEXT
);
INSERT INTO pets VALUES
    (1, 'Rex',    'dog'),
    (2, 'Whisky', 'cat'),
    (3, 'Bubbles','fish');
"""

MINI_DBS = {"mini_singer": _MINI_SINGER_DDL, "mini_pets": _MINI_PETS_DDL}


@pytest.fixture(scope="session")
def db_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mini-spider")
    for db_id, ddl in MINI_DBS.items():
        db_dir = root / db_id
        db_dir.mkdir()
        conn = sqlite3.connect(db_dir / f"{db_id}.sqlite")
        try:
            conn.executescript(ddl)
            conn.commit()
        finally:
            conn.close()
    return root
