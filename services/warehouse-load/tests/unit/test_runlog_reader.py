"""§12.1 — runlog reader handles fixture lines, parse errors, extras."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from warehouse_load.core.runlog_reader import iter_runlog_rows


def _row_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "country_code": "ca",
        "source_code": "ckan-opencanada",
        "organization_code": "fin",
        "document_id": "a" * 64,
        "source_url": "https://example.org/x.csv",
        "gcs_uri": None,
        "checksum": "b" * 64,
        "etag": None,
        "http_last_modified": None,
        "resource_last_modified": None,
        "file_format": "csv",
        "declared_format": "CSV",
        "language": "en",
        "title": "T",
        "document_type": None,
        "subjects": ["economics_and_industry"],
        "published_date": None,
        "metadata_modified": "2026-05-01T00:00:00+00:00",
        "ingested_at": "2026-06-01T00:00:00+00:00",
        "ingestion_status": "success",
        "quarantine_reason": None,
        "run_id": "run-1",
    }
    base.update(overrides)
    return base


def test_reader_yields_rows(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(_row_dict()) + "\n")

    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None))
    assert len(events) == 1
    assert events[0].error is None
    assert events[0].row is not None
    assert events[0].row.document_id == "a" * 64


def test_reader_ignores_extra_fields(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(_row_dict(new_field="value-from-the-future")) + "\n")

    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None))
    assert events[0].error is None


def test_reader_surfaces_json_parse_errors(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text("not json at all\n" + json.dumps(_row_dict()) + "\n")

    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None))
    assert len(events) == 2
    assert events[0].error is not None
    assert "json" in events[0].error.error
    assert events[1].row is not None


def test_reader_surfaces_pydantic_validation_errors(tmp_path: Path) -> None:
    """A row with an unknown language Literal should surface as a parse error,
    not blow up the iterator."""
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(_row_dict(language="zz")) + "\n")

    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None))
    assert events[0].error is not None
    assert "schema" in events[0].error.error


def test_reader_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text("\n" + json.dumps(_row_dict()) + "\n\n")

    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None))
    assert len(events) == 1


def test_reader_since_filter_drops_old_rows(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        json.dumps(_row_dict(ingested_at="2026-05-01T00:00:00+00:00")) + "\n"
        + json.dumps(_row_dict(ingested_at="2026-06-01T00:00:00+00:00")) + "\n",
    )

    cutoff = datetime(2026, 5, 15, tzinfo=UTC)
    events = list(iter_runlog_rows(local_dir=tmp_path, gcs_prefix=None, since=cutoff))
    assert len(events) == 1
    assert events[0].row is not None
    assert events[0].row.ingested_at >= cutoff


def test_reader_requires_a_source() -> None:
    import pytest
    with pytest.raises(ValueError, match="at least one of"):
        list(iter_runlog_rows(local_dir=None, gcs_prefix=None))
