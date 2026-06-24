"""Batch MERGE from `raw.rows_staging` into `raw.rows`.

The staging table is shared across the batch — per-doc workers
append JSONL files into it (`WRITE_APPEND` is atomic per load job),
then once the batch's docs have all finished, this module issues a
single MERGE+TRUNCATE script to land the rows in `raw.rows`.

DELETE-then-INSERT semantics (PRD §8.2) make per-doc replay
correct: any doc in the staging batch has its prior rows in
`raw.rows` deleted, then the new rows inserted. The two statements
run as a BQ multi-statement script so a crash between them doesn't
leave staging polluted.
"""
from __future__ import annotations

import re

from warehouse_load.clients.bq import BqClient

_BQ_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def assert_staging_empty(*, bq: BqClient, staging_table: str) -> int:
    """Return the row count in `staging_table`.

    The runner calls this at startup and refuses to proceed if the
    count is non-zero (PRD §8.0 single-runner-at-a-time guard). We
    don't fail here — the runner emits the structured event with
    the recovery command before exiting.
    """
    _validate_table_id(staging_table)
    return bq.count_rows(staging_table)


def flush_batch(
    *,
    bq: BqClient,
    staging_table: str,
    target_table: str,
) -> None:
    """MERGE all staging rows into the target, then TRUNCATE staging.

    DELETE-then-INSERT: every doc in the staging set loses its prior
    rows in the target, then gains the new ones. The MERGE keys on
    `document_id`, so multi-doc batches land correctly in one pass.

    BQ multi-statement scripts run as one transaction at the script
    level. The TRUNCATE only fires if the MERGE succeeded — a crash
    in the MERGE leaves staging populated, which the next runner's
    §8.0 precondition will catch.
    """
    _validate_table_id(staging_table)
    _validate_table_id(target_table)

    sql = f"""\
BEGIN
  MERGE INTO `{target_table}` t
  USING `{staging_table}` s
    ON t.document_id = s.document_id
  WHEN MATCHED THEN DELETE;

  INSERT INTO `{target_table}` (document_id, row_index, row, loaded_at)
  SELECT document_id, row_index, row, loaded_at FROM `{staging_table}`;

  TRUNCATE TABLE `{staging_table}`;
END;
"""
    bq.execute(sql)


def _validate_table_id(table_id: str) -> None:
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected project.dataset.table, got {table_id!r}")
    for part in parts:
        if not _BQ_IDENT_RE.fullmatch(part):
            raise ValueError(f"invalid BQ identifier segment {part!r} in {table_id!r}")
