"""SQL generation for the batch MERGE."""
from __future__ import annotations

import pytest

from tests.integration.conftest import FakeBqClient
from warehouse_load.core import rows_merge


def test_flush_batch_emits_delete_then_insert_then_truncate() -> None:
    bq = FakeBqClient()
    rows_merge.flush_batch(
        bq=bq,
        staging_table="proj.raw.rows_staging",
        target_table="proj.raw.rows",
    )
    assert len(bq.query_calls) == 1
    sql = bq.query_calls[0]
    assert sql.startswith("BEGIN")
    assert "MERGE INTO `proj.raw.rows` t" in sql
    assert "USING `proj.raw.rows_staging` s" in sql
    assert "WHEN MATCHED THEN DELETE" in sql
    assert "INSERT INTO `proj.raw.rows`" in sql
    assert "TRUNCATE TABLE `proj.raw.rows_staging`" in sql
    # Ordering: delete must precede insert must precede truncate.
    assert sql.index("WHEN MATCHED THEN DELETE") < sql.index("INSERT INTO")
    assert sql.index("INSERT INTO") < sql.index("TRUNCATE TABLE")


def test_assert_staging_empty_returns_zero_for_empty_table() -> None:
    bq = FakeBqClient()
    count = rows_merge.assert_staging_empty(
        bq=bq, staging_table="proj.raw.rows_staging",
    )
    assert count == 0


def test_assert_staging_empty_returns_count() -> None:
    bq = FakeBqClient()
    bq.target_rows = {str(i): {} for i in range(5)}
    count = rows_merge.assert_staging_empty(
        bq=bq, staging_table="proj.raw.rows_staging",
    )
    assert count == 5


def test_flush_batch_rejects_bad_table_id() -> None:
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="invalid BQ identifier"):
        rows_merge.flush_batch(
            bq=bq,
            staging_table="proj`.raw.rows_staging",
            target_table="proj.raw.rows",
        )
