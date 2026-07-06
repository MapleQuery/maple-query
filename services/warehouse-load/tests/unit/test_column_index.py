"""SQL rendering for `core/column_index.refresh_column_index`."""
from __future__ import annotations

from typing import Any

import pytest

from tests.integration.conftest import FakeBqClient
from warehouse_load.core import column_index as column_index_mod


def _logger() -> Any:
    import structlog

    return structlog.get_logger("test")


def test_refresh_emits_create_or_replace_with_cap() -> None:
    bq = FakeBqClient()
    # Two query_rows pre-loads: COUNT(*) for unique_cols, then SUM for total.
    bq.query_results = [[{"n": 100}], [{"total": 5000}]]
    # The implementation actually calls count_rows for unique_cols
    # (which reads from target_rows). Pre-seed those instead.
    bq.target_rows = {f"col_{i}": {} for i in range(100)}
    bq.query_results = [[{"total": 5000}]]

    column_index_mod.refresh_column_index(
        bq=bq, rows_table="proj.raw.rows",
        column_index_table="proj.raw.column_index",
        doc_ids_cap=1000, log=_logger(), run_id="run-x",
    )

    assert len(bq.query_calls) >= 1
    sql = bq.query_calls[0]
    assert "CREATE OR REPLACE TABLE `proj.raw.column_index`" in sql
    assert "LIMIT 1000" in sql
    assert "JSON_KEYS(row)" in sql
    assert "overflow_truncated" in sql


def test_refresh_rejects_bad_table_id() -> None:
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="invalid BQ identifier"):
        column_index_mod.refresh_column_index(
            bq=bq, rows_table="proj`.raw.rows",
            column_index_table="proj.raw.column_index",
            doc_ids_cap=1000, log=_logger(), run_id="run-x",
        )


def test_refresh_rejects_zero_cap() -> None:
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="doc_ids_cap"):
        column_index_mod.refresh_column_index(
            bq=bq, rows_table="proj.raw.rows",
            column_index_table="proj.raw.column_index",
            doc_ids_cap=0, log=_logger(), run_id="run-x",
        )


def test_refresh_returns_counts() -> None:
    bq = FakeBqClient()
    bq.target_rows = {f"col_{i}": {} for i in range(7)}
    bq.query_results = [[{"total": 42}]]

    result = column_index_mod.refresh_column_index(
        bq=bq, rows_table="proj.raw.rows",
        column_index_table="proj.raw.column_index",
        doc_ids_cap=500, log=_logger(), run_id="run-x",
    )
    assert result.unique_cols == 7
    assert result.total_doc_col_pairs == 42
