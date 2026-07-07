"""End-to-end `columns-load` against a FakeBqClient."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.columns_load import (
    ColumnsLoadRequest,
    _build_merge_sql,
    run_load,
)

from .conftest import FakeBqClient

# Load validates staged embeddings against the OpenAI embedding dim;
# fixtures must match it or pre-load validation rejects them.
_DIM = Settings.model_fields["openai_embedding_dim"].default


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
    )


def _staged(
    package_id: str,
    column_name: str,
    *,
    embedding: list[float] | None,
    generation_failed: bool = False,
    description: str | None = None,
) -> dict:
    return {
        "package_id": package_id,
        "column_name": column_name,
        "semantic_type": "text",
        "description": description or ("a canned description, " * 3),
        "sample_values": ["v1", "v2"],
        "embedding": embedding,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "generation_model": "fake",
        "generation_model_commit": None,
        "generation_run_id": "r1",
        "generation_failed": generation_failed,
        "failure_reason": "chunk_count_exceeded_cap" if generation_failed else None,
        "dry_run": False,
    }


def _seed(tmp_path: Path, run_id: str, rows: list[dict]) -> None:
    columns_dir = tmp_path / run_id / "columns"
    columns_dir.mkdir(parents=True)
    with (columns_dir / "000.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_three_columns_inserts(tmp_path: Path) -> None:
    rows = [
        _staged("pkg-a", "col_x", embedding=[0.1] * _DIM),
        _staged("pkg-a", "col_y", embedding=[0.2] * _DIM),
        _staged("pkg-b", "col_z", embedding=[0.3] * _DIM),
    ]
    _seed(tmp_path, "r1", rows)
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.columns`", [])

    summary = run_load(
        request=ColumnsLoadRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.rows_inserted == 3
    assert summary.rows_updated == 0
    assert summary.rows_unchanged == 0
    assert summary.embedding_null_count == 0
    assert summary.failure_marker_count == 0

    # Staging table created, MERGE executed, table dropped.
    assert any(
        name.startswith("proj.semantic._columns_staging_")
        for name in bq.staging_tables
    )
    assert bq.deleted_tables
    merge_calls = [
        c for c in bq.calls if c["kind"] == "execute" and "MERGE INTO" in c["sql"]
    ]
    assert len(merge_calls) == 1


def test_load_excludes_failure_markers_and_embedding_null(tmp_path: Path) -> None:
    rows = [
        _staged("pkg-a", "col_x", embedding=[0.1] * _DIM),
        _staged("pkg-b", "col_y", embedding=None),  # embed pending
        _staged("pkg-c", "__failure_marker__", embedding=None,
                generation_failed=True,
                description="GENERATION_FAILED_PLACEHOLDER_DESCRIPTION_MIN_LEN_OK"),
    ]
    _seed(tmp_path, "r1", rows)
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.columns`", [])

    summary = run_load(
        request=ColumnsLoadRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.coalesced_row_count == 3
    assert summary.failure_marker_count == 1
    assert summary.embedding_null_count == 1
    assert summary.rows_staged == 1
    assert summary.rows_inserted == 1


def test_load_dry_run_no_touches(tmp_path: Path) -> None:
    _seed(tmp_path, "r1", [_staged("pkg-a", "x", embedding=[0.1] * _DIM)])
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.columns`", [])
    summary = run_load(
        request=ColumnsLoadRequest(run_id="r1", dry_run=True),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.dry_run is True
    assert summary.rows_inserted == 1
    assert bq.staging_tables == {}
    assert bq.deleted_tables == []


def test_load_rejects_short_description(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        "r1",
        [_staged("pkg-a", "x", embedding=[0.1] * _DIM, description="short")],
    )
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.columns`", [])
    with pytest.raises(RuntimeError, match="description_too_short"):
        run_load(
            request=ColumnsLoadRequest(run_id="r1", dry_run=False),
            settings=_settings(tmp_path),
            bq=bq,
        )


def test_merge_sql_no_delete_clause() -> None:
    """v1 is one-shot per parent §9; the MERGE must not delete other
    packages' rows on a scoped re-run."""
    sql = _build_merge_sql(
        target="p.semantic.columns", staging="p.semantic._stg"
    )
    assert "DELETE" not in sql
    assert "WHEN NOT MATCHED BY SOURCE" not in sql
