"""Spider dataset acquisition, validation, and loading.

Sources (resolved during planning, pinned here):
- Questions + gold SQL: the ``xlangai/spider`` HF dataset (parquet; the
  canonical 7000 train / 1034 dev rows). Converted ONCE to plain JSON at
  download time — pyarrow is a dev-only dependency, everything downstream
  (including the Docker image) reads JSON with stdlib.
- SQLite databases: ``HAL-9001/spider-databases`` (the official
  ``spider_data.zip``). Manual fallback: place the zip at
  ``<root>/spider_data.zip`` yourself and re-run — the pipeline is identical
  from there.

``download`` is idempotent (a valid MANIFEST.json short-circuits) and ends by
validating: expected counts, every dev db_id resolving to a SQLite file, an
integrity-check sample, and SHA256s recorded in the manifest so other
machines can assert they got the same data.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

QA_REPO: Final = "xlangai/spider"
QA_FILES: Final = {
    "train": "spider/train-00000-of-00001.parquet",
    "dev": "spider/validation-00000-of-00001.parquet",
}
DB_REPO: Final = "HAL-9001/spider-databases"
DB_ZIP: Final = "spider_data.zip"

SPLIT_JSON: Final = {"train": "train_spider.json", "dev": "dev.json"}
EXPECTED_COUNTS: Final = {"train": 7000, "dev": 1034}
_DB_ID_RE: Final = re.compile(r"^[A-Za-z0-9_]+$")
_INTEGRITY_SAMPLE: Final = 5


@dataclass(frozen=True)
class SpiderExample:
    """One benchmark example. ``sql`` is the gold query (JSON key: "query")."""

    question: str
    sql: str
    db_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_parquet(parquet_path: Path, dest_json: Path) -> int:
    import pyarrow.parquet as pq  # dev-only dep; never imported at runtime

    table = pq.read_table(parquet_path, columns=["question", "query", "db_id"])
    records = [
        {"question": question, "query": query, "db_id": db_id}
        for question, query, db_id in zip(
            table.column("question").to_pylist(),
            table.column("query").to_pylist(),
            table.column("db_id").to_pylist(),
            strict=True,
        )
    ]
    dest_json.write_text(json.dumps(records, indent=1))
    return len(records)


def _extract_databases(zip_path: Path, root: Path) -> int:
    """Extract only ``*/database/<db_id>/<db_id>.sqlite`` members.

    ``test_database/`` is deliberately skipped (unused, ~doubles the size),
    and db_ids are validated against the executor's regex — which is also the
    zip-slip guard: no separators, no traversal.
    """
    extracted: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if "database" not in parts or "test_database" in parts:
                continue
            anchor = parts.index("database")
            if len(parts) != anchor + 3:  # database/<db_id>/<file>
                continue
            db_id, filename = parts[anchor + 1], parts[anchor + 2]
            if not _DB_ID_RE.match(db_id) or filename != f"{db_id}.sqlite":
                continue
            dest = root / "database" / db_id / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src:
                dest.write_bytes(src.read())
            extracted.add(db_id)
    return len(extracted)


def _validate(root: Path, counts: dict[str, int], db_count: int) -> dict[str, object]:
    for split, expected in EXPECTED_COUNTS.items():
        if counts[split] != expected:
            raise RuntimeError(f"{split}: expected {expected} examples, got {counts[split]}")

    missing = sorted(
        {example.db_id for example in load_examples(root, "dev")}
        - {path.name for path in (root / "database").iterdir()}
    )
    if missing:
        raise RuntimeError(f"dev databases missing from archive: {missing[:5]} …")

    for db_dir in sorted((root / "database").iterdir())[:_INTEGRITY_SAMPLE]:
        conn = sqlite3.connect(f"file:{db_dir / (db_dir.name + '.sqlite')}?mode=ro", uri=True)
        try:
            status = conn.execute("PRAGMA integrity_check").fetchone()
            if status is None or status[0] != "ok":
                raise RuntimeError(f"integrity_check failed for {db_dir.name}: {status}")
        finally:
            conn.close()

    return {
        "sources": {"qa": QA_REPO, "databases": f"{DB_REPO}/{DB_ZIP}"},
        "counts": counts,
        "database_count": db_count,
        "sha256": {split: _sha256(root / SPLIT_JSON[split]) for split in SPLIT_JSON},
    }


def manifest_path(root: Path) -> Path:
    return root / "MANIFEST.json"


def download(root: Path, *, force: bool = False) -> dict[str, object]:
    """Fetch, normalize, and validate the dataset under ``root``. Idempotent."""
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path(root).exists() and not force:
        return json.loads(manifest_path(root).read_text())

    from huggingface_hub import hf_hub_download  # lazy: network-side only

    counts = {
        split: _convert_parquet(
            Path(hf_hub_download(QA_REPO, repo_file, repo_type="dataset")),
            root / SPLIT_JSON[split],
        )
        for split, repo_file in QA_FILES.items()
    }

    local_zip = root / DB_ZIP  # manual-fallback location, preferred if present
    if not local_zip.exists():
        local_zip = Path(hf_hub_download(DB_REPO, DB_ZIP, repo_type="dataset"))
    db_count = _extract_databases(local_zip, root)

    manifest = _validate(root, counts, db_count)
    manifest_path(root).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(root: Path) -> None:
    """Assert the on-disk JSONs match the manifest (drift guard for loaders)."""
    if not manifest_path(root).exists():
        raise RuntimeError(f"no MANIFEST.json under {root} — run `dspyed download` first")
    manifest = json.loads(manifest_path(root).read_text())
    recorded: dict[str, str] = manifest["sha256"]
    for split, json_name in SPLIT_JSON.items():
        actual = _sha256(root / json_name)
        if actual != recorded[split]:
            raise RuntimeError(
                f"{json_name} drifted from MANIFEST.json — re-run `dspyed download --force`"
            )


def load_examples(root: Path, split: str) -> list[SpiderExample]:
    records = json.loads((root / SPLIT_JSON[split]).read_text())
    return [
        SpiderExample(question=r["question"], sql=r["query"], db_id=r["db_id"]) for r in records
    ]
