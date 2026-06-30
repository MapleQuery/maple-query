"""Unit tests for `core.columns_load`.

Covers the MERGE SQL shape (§10.2) and pre-load validation (§10.4).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
import structlog

from semantic_enrich.core.columns_load import (
    _build_merge_sql,
    _preload_validate,
)
from semantic_enrich.types import StagedColumnRow


def _row(
    *,
    description: str = "x" * 40,
    embedding: list[float] | None = None,
    column_name: str = "col_a",
) -> StagedColumnRow:
    return StagedColumnRow(
        package_id="pkg-1",
        column_name=column_name,
        semantic_type="text",
        description=description,
        sample_values=[],
        embedding=embedding if embedding is not None else [0.1] * 1024,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        generation_model="fake",
        generation_model_commit=None,
        generation_run_id="r1",
        generation_failed=False,
        failure_reason=None,
        dry_run=False,
    )


def test_merge_sql_shape() -> None:
    sql = _build_merge_sql(target="p.semantic.columns", staging="p.semantic._stg")
    assert "MERGE INTO `p.semantic.columns` t" in sql
    assert "USING `p.semantic._stg` s" in sql
    # Composite key.
    assert "ON t.package_id  = s.package_id" in sql
    assert "AND t.column_name = s.column_name" in sql
    assert "WHEN MATCHED AND s.generated_at > t.generated_at" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    # No DELETE clause (§10.2 invariant 3).
    assert "DELETE" not in sql
    # All target columns appear in the UPDATE SET.
    for col in (
        "semantic_type", "description", "sample_values",
        "embedding", "generated_at",
    ):
        assert col in sql


def test_preload_validate_accepts_clean_row() -> None:
    log = structlog.get_logger()
    _preload_validate(
        rows=[_row()], embedding_dim=1024, run_id="r1", log=log
    )


def test_preload_validate_rejects_short_description() -> None:
    log = structlog.get_logger()
    with pytest.raises(RuntimeError, match="description_too_short"):
        _preload_validate(
            rows=[_row(description="too short")],
            embedding_dim=1024,
            run_id="r1",
            log=log,
        )


def test_preload_validate_rejects_long_description() -> None:
    log = structlog.get_logger()
    with pytest.raises(RuntimeError, match="description_too_long"):
        _preload_validate(
            rows=[_row(description="x" * 601)],
            embedding_dim=1024,
            run_id="r1",
            log=log,
        )


def test_preload_validate_rejects_wrong_embedding_dim() -> None:
    log = structlog.get_logger()
    with pytest.raises(RuntimeError, match="wrong_embedding_dim"):
        _preload_validate(
            rows=[_row(embedding=[0.1] * 512)],
            embedding_dim=1024,
            run_id="r1",
            log=log,
        )


def test_preload_validate_rejects_empty_column_name() -> None:
    log = structlog.get_logger()
    with pytest.raises(RuntimeError, match="empty_column_name"):
        _preload_validate(
            rows=[_row(column_name="")],
            embedding_dim=1024,
            run_id="r1",
            log=log,
        )
