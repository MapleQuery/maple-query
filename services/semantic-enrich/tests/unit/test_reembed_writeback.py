"""Assert the reembed MERGE SQL shape.

Only `embedding` and `generated_at` should ever be in the UPDATE
clause. `summary`, `grain`, `measures`, `dimensions`,
`date_range_start`, `date_range_end`, `semantic_type`, `description`,
`sample_values` must NOT appear anywhere in the SQL — a reembed pass
that touches those columns would silently drop generation output.
"""
from __future__ import annotations

from semantic_enrich.core.reembed import (
    _build_columns_merge_sql,
    _build_datasets_merge_sql,
)

_DATASETS_MUTABLE_FIELDS = (
    "summary",
    "grain",
    "measures",
    "dimensions",
    "date_range_start",
    "date_range_end",
)
_COLUMNS_MUTABLE_FIELDS = ("semantic_type", "description", "sample_values")


def test_datasets_merge_updates_only_embedding_and_generated_at() -> None:
    sql = _build_datasets_merge_sql("proj.semantic.datasets", "proj.semantic._stg")
    assert "MERGE INTO `proj.semantic.datasets` t" in sql
    assert "USING `proj.semantic._stg` s" in sql
    assert "ON t.package_id = s.package_id" in sql
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert "embedding = s.embedding" in sql
    assert "generated_at = CURRENT_TIMESTAMP()" in sql
    # No INSERT branch — reembed should never insert.
    assert "WHEN NOT MATCHED" not in sql
    for field in _DATASETS_MUTABLE_FIELDS:
        assert field not in sql, f"datasets reembed MERGE touches `{field}`"


def test_columns_merge_updates_only_embedding_and_generated_at() -> None:
    sql = _build_columns_merge_sql("proj.semantic.columns", "proj.semantic._stg")
    assert "MERGE INTO `proj.semantic.columns` t" in sql
    assert "USING `proj.semantic._stg` s" in sql
    assert "ON t.package_id  = s.package_id" in sql
    assert "AND t.column_name = s.column_name" in sql
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert "embedding = s.embedding" in sql
    assert "generated_at = CURRENT_TIMESTAMP()" in sql
    assert "WHEN NOT MATCHED" not in sql
    for field in _COLUMNS_MUTABLE_FIELDS:
        assert field not in sql, f"columns reembed MERGE touches `{field}`"
