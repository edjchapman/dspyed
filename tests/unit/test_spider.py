"""Spider acquisition plumbing — all offline: zip hygiene, parquet → JSON, manifest."""

import json
import zipfile
from pathlib import Path

import pytest

from dspyed.data.spider import (
    SPLIT_JSON,
    _convert_parquet,
    _extract_databases,
    _sha256,
    load_examples,
    verify_manifest,
)


def test_extract_takes_only_wellformed_database_members(tmp_path: Path):
    zip_path = tmp_path / "spider_data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("spider_data/database/good_db/good_db.sqlite", b"db-bytes")
        archive.writestr("spider_data/database/good_db/schema.sql", b"ddl")  # wrong file
        archive.writestr("spider_data/test_database/tst/tst.sqlite", b"skip")  # test set
        archive.writestr("spider_data/database/../evil/evil.sqlite", b"slip")  # traversal
        archive.writestr("spider_data/database/bad id/bad id.sqlite", b"space")  # bad id
        archive.writestr("spider_data/database/deep/nested/deep.sqlite", b"deep")  # depth

    root = tmp_path / "root"
    count = _extract_databases(zip_path, root)

    assert count == 1
    extracted = sorted(path.relative_to(root) for path in root.rglob("*.sqlite"))
    assert extracted == [Path("database/good_db/good_db.sqlite")]
    assert (root / "database" / "good_db" / "good_db.sqlite").read_bytes() == b"db-bytes"


def test_parquet_roundtrips_to_examples(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "question": ["How many singers?", "List venues"],
            "query": ["SELECT count(*) FROM singer", "SELECT venue FROM concert"],
            "db_id": ["mini_singer", "mini_singer"],
            "extra_toks": [["ignored"], ["ignored"]],  # extra columns are dropped
        }
    )
    parquet_path = tmp_path / "part.parquet"
    pq.write_table(table, parquet_path)

    root = tmp_path
    count = _convert_parquet(parquet_path, root / SPLIT_JSON["dev"])
    assert count == 2

    examples = load_examples(root, "dev")
    assert examples[0].question == "How many singers?"
    assert examples[0].sql == "SELECT count(*) FROM singer"
    assert examples[0].db_id == "mini_singer"


def test_verify_manifest_detects_drift(tmp_path: Path):
    for name in SPLIT_JSON.values():
        (tmp_path / name).write_text("[]")
    manifest = {"sha256": {split: _sha256(tmp_path / name) for split, name in SPLIT_JSON.items()}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))

    verify_manifest(tmp_path)  # clean state passes

    (tmp_path / SPLIT_JSON["dev"]).write_text('[{"tampered": true}]')
    with pytest.raises(RuntimeError, match="drifted"):
        verify_manifest(tmp_path)


def test_verify_manifest_requires_download_first(tmp_path: Path):
    with pytest.raises(RuntimeError, match="dspyed download"):
        verify_manifest(tmp_path)
