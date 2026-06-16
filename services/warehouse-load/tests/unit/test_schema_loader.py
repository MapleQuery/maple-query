"""Schema files in infra/terraform/schemas/ parse via SchemaField."""
from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_load.core.schema_loader import load_schema


def test_raw_documents_schema_parses(schemas_dir: Path) -> None:
    fields = load_schema(schemas_dir / "raw_documents.json")
    assert fields, "raw_documents.json should not be empty"
    for f in fields:
        assert f.name
        assert f.field_type
        assert f.mode


@pytest.mark.parametrize("name", ["raw_documents.json", "raw_rows.json", "raw_column_index.json"])
def test_all_raw_schemas_parse(schemas_dir: Path, name: str) -> None:
    fields = load_schema(schemas_dir / name)
    assert fields


def test_load_schema_rejects_non_array(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "not_an_array"}')
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_schema(bad)


def test_raw_documents_owns_expected_columns(schemas_dir: Path) -> None:
    """Guard against the schema drifting away from what the loader writes."""
    fields = load_schema(schemas_dir / "raw_documents.json")
    field_names = {f.name for f in fields}

    expected_subset = {
        "country_code", "source_code", "organization_code",
        "document_id", "source_url", "gcs_uri",
        "file_format", "language", "subjects",
        "metadata_modified", "ingested_at", "ingestion_status",
        # Content-loader columns inserted as NULL/'pending':
        "preamble_rows", "header_confidence", "load_status",
        "load_attempted_at", "load_error", "row_count",
    }
    missing = expected_subset - field_names
    assert not missing, f"raw_documents.json missing expected columns: {sorted(missing)}"
