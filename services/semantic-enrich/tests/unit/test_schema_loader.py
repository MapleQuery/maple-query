"""Schema-loader smoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from semantic_enrich.config.settings import _find_schemas_dir
from semantic_enrich.core.schema_loader import (
    assert_datasets_schema,
    load_schema,
)


def test_load_semantic_datasets_schema() -> None:
    path = _find_schemas_dir() / "semantic_datasets.json"
    schema = load_schema(path)
    by_name = {f.name: f for f in schema}
    assert "package_id" in by_name
    assert "embedding" in by_name
    emb = by_name["embedding"]
    assert emb.field_type == "FLOAT64"
    assert emb.mode == "REPEATED"


def test_assert_datasets_schema_passes_on_real_file() -> None:
    schema = load_schema(_find_schemas_dir() / "semantic_datasets.json")
    assert_datasets_schema(schema)  # does not raise


def test_assert_datasets_schema_fails_without_embedding() -> None:
    schema = load_schema(_find_schemas_dir() / "raw_documents.json")
    with pytest.raises(AssertionError):
        assert_datasets_schema(schema)


def test_assert_datasets_schema_fails_without_representative() -> None:
    schema = [
        f
        for f in load_schema(_find_schemas_dir() / "semantic_datasets.json")
        if f.name != "representative_document_id"
    ]
    with pytest.raises(AssertionError, match="representative_document_id"):
        assert_datasets_schema(schema)


def test_load_schema_rejects_non_list(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_schema(bad)
