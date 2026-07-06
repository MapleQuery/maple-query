"""Unit tests for the rows_runner JSONL write path.

The key invariant: `row` must be written as a JSON object, not a
JSON-encoded string. BigQuery ingests a native JSON object into a
`JSON`-typed column as an object tree; a string scalar would require
a `PARSE_JSON(STRING(row))` unwrap on every query.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from warehouse_load.config.settings import Settings
from warehouse_load.core.rows_runner import _DocWorkPaths, _parse_one_pass
from warehouse_load.types import DocumentRow, SniffResult

_SCHEMAS_DIR = (
    Path(__file__).resolve().parents[4] / "infra" / "terraform" / "schemas"
)


def _doc(doc_id: str = "a" * 64) -> DocumentRow:
    return DocumentRow(
        document_id=doc_id,
        organization_code="fin",
        source_url=f"https://example.org/{doc_id[:8]}.csv",
        gcs_uri=f"gs://bucket/{doc_id[:8]}.csv",
        file_format="csv",
        declared_format="CSV",
        checksum="x" * 64,
        resource_last_modified=None,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        schemas_dir=_SCHEMAS_DIR,
        runlog_local_dir=tmp_path,
        body_min_run=5,
        header_lookback=3,
    )


def _sniff() -> SniffResult:
    return SniffResult(delimiter=",", encoding="utf-8", sniff_bytes=256)


def _write_csv(tmp_path: Path, content: bytes) -> Path:
    p = tmp_path / "fixture.csv"
    p.write_bytes(content)
    return p


def _log() -> object:
    return structlog.get_logger("test")


_SIMPLE_CSV = (
    b"col_a,col_b\n"
    + b"\n".join(f"val_{i},{i}".encode() for i in range(8))
    + b"\n"
)

_NULL_CELL_CSV = (
    b"col_a,col_b\n"
    b"hello,world\n"
    b",\n"
)


def test_row_is_written_as_object(tmp_path: Path) -> None:
    """The staging JSONL must contain `"row": {...}` not `"row": "{...}"`."""
    raw_blob = _write_csv(tmp_path, _SIMPLE_CSV)
    paths = _DocWorkPaths(raw_blob=raw_blob)
    settings = _settings(tmp_path)
    doc = _doc()
    sniff = _sniff()

    result = _parse_one_pass(
        doc=doc, paths=paths, sniff=sniff, settings=settings, log=_log()
    )

    assert result.final_status == "loaded"
    assert paths.staging_jsonl is not None

    with paths.staging_jsonl.open(encoding="utf-8") as f:
        first_line = f.readline().strip()

    record = json.loads(first_line)
    assert isinstance(record["row"], dict), (
        f"row must be a dict (native JSON object), got {type(record['row'])!r}. "
        "If this is a str, the loader is double-encoding the row field."
    )
    assert "col_a" in record["row"]
    assert "col_b" in record["row"]


def test_null_cell_round_trips_as_none(tmp_path: Path) -> None:
    """Empty CSV cells must land as JSON null, not the string 'null'."""
    raw_blob = _write_csv(tmp_path, _NULL_CELL_CSV)
    paths = _DocWorkPaths(raw_blob=raw_blob)
    settings = _settings(tmp_path)
    doc = _doc()
    sniff = _sniff()

    result = _parse_one_pass(
        doc=doc, paths=paths, sniff=sniff, settings=settings, log=_log()
    )

    assert result.final_status == "loaded"
    assert paths.staging_jsonl is not None

    lines = [
        json.loads(line)
        for line in paths.staging_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Second body row has two empty cells.
    null_row = lines[1]["row"]
    assert isinstance(null_row, dict)
    assert null_row["col_a"] is None
    assert null_row["col_b"] is None
