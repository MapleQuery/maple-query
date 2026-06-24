"""RawRunlogRow validators."""
from __future__ import annotations

from datetime import UTC, datetime

from warehouse_load.types import RawRunlogRow


def _payload(**overrides: object) -> dict[str, object]:
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


def test_naive_ingested_at_is_anchored_to_utc() -> None:
    """Naive timestamps in the runlog are assumed UTC at parse time so
    downstream `--since` and dedupe tie-break comparisons don't raise
    TypeError on tz-aware vs naive."""
    row = RawRunlogRow.model_validate(_payload(ingested_at="2026-06-01T00:00:00"))
    assert row.ingested_at.tzinfo is not None
    assert row.ingested_at == datetime(2026, 6, 1, tzinfo=UTC)


def test_naive_metadata_modified_is_anchored_to_utc() -> None:
    row = RawRunlogRow.model_validate(_payload(metadata_modified="2026-05-01T00:00:00"))
    assert row.metadata_modified.tzinfo is not None


def test_tzaware_input_is_preserved() -> None:
    """Already-aware inputs round-trip without re-anchoring."""
    row = RawRunlogRow.model_validate(_payload(ingested_at="2026-06-01T00:00:00+02:00"))
    assert row.ingested_at.tzinfo is not None
    assert row.ingested_at.utcoffset() is not None
