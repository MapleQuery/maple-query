"""Rebuild `raw.column_index` from `raw.rows`.

`CREATE OR REPLACE TABLE` is atomic at the BQ level: readers either
see the prior table or the new one, never a partial state. The
~280K rows the index produces (PRD §10) are trivial to rewrite, so
the simpler full-rebuild beats incremental maintenance.

Cap on `document_ids`: bounded to `doc_ids_cap` (default 1000). When
the cap fires, `overflow_truncated=TRUE` signals to the agent that
the doc-id list is incomplete and the full set is recoverable from
a `raw.rows` scan.

`JSON_KEYS` is BQ Preview as of 2026-Q1; `row` is a BQ JSON type so
it can be passed directly without a STRING/PARSE_JSON round-trip.
"""
from __future__ import annotations

import re
import time
from typing import Any

from warehouse_load.clients.bq import BqClient
from warehouse_load.types import ColumnIndexRefreshResult

_BQ_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def refresh_column_index(
    *,
    bq: BqClient,
    rows_table: str,
    column_index_table: str,
    doc_ids_cap: int,
    log: Any,
    run_id: str,
) -> ColumnIndexRefreshResult:
    """Rebuild `column_index_table` from `rows_table`.

    Returns a `ColumnIndexRefreshResult` with the row counts. The
    counts come from a follow-up COUNT query — BQ doesn't return
    row counts on `CREATE OR REPLACE`. Single extra scan is cheap;
    the value to the operator (seeing the index grow as the corpus
    loads) is higher than the cost.
    """
    _validate_table_id(rows_table)
    _validate_table_id(column_index_table)
    if doc_ids_cap <= 0:
        raise ValueError(f"doc_ids_cap must be positive, got {doc_ids_cap}")

    log.info("column_index_refresh_start", run_id=run_id)
    started = time.monotonic()

    sql = _render_refresh_sql(
        rows_table=rows_table,
        column_index_table=column_index_table,
        doc_ids_cap=doc_ids_cap,
    )
    bq.execute(sql)

    unique_cols = bq.count_rows(column_index_table)
    total_doc_col_pairs = _sum_file_count(bq=bq, column_index_table=column_index_table)

    duration_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "column_index_refresh_done",
        run_id=run_id,
        unique_cols=unique_cols,
        total_doc_col_pairs=total_doc_col_pairs,
        duration_ms=duration_ms,
    )
    return ColumnIndexRefreshResult(
        unique_cols=unique_cols,
        total_doc_col_pairs=total_doc_col_pairs,
        duration_ms=duration_ms,
    )


def _render_refresh_sql(
    *,
    rows_table: str,
    column_index_table: str,
    doc_ids_cap: int,
) -> str:
    """Render the `CREATE OR REPLACE` script.

    `doc_ids_cap` is inlined (it's an int, not user input) so the
    statement runs as a single non-parameterised script — cleaner
    for the operator if they want to copy-paste it from logs.
    """
    return f"""\
CREATE OR REPLACE TABLE `{column_index_table}` AS
WITH keys_per_doc AS (
  SELECT DISTINCT
    document_id,
    k AS col_name
  FROM `{rows_table}`,
       UNNEST(JSON_KEYS(row)) AS k
),
agg AS (
  SELECT
    col_name,
    LOWER(col_name) AS col_name_lower,
    COUNT(DISTINCT document_id) AS file_count,
    ARRAY_AGG(DISTINCT document_id LIMIT {doc_ids_cap}) AS document_ids,
    COUNT(DISTINCT document_id) > {doc_ids_cap} AS overflow_truncated
  FROM keys_per_doc
  GROUP BY col_name
)
SELECT
  col_name,
  col_name_lower,
  file_count,
  document_ids,
  overflow_truncated,
  CURRENT_TIMESTAMP() AS refreshed_at
FROM agg
"""


def _sum_file_count(*, bq: BqClient, column_index_table: str) -> int:
    """Total `SUM(file_count)`. Approximates "total (doc, column) pairs"
    — useful as a sanity check that the rebuild produced meaningful
    output, not the precise truth (would need to re-scan `raw.rows`).
    """
    sql = f"SELECT COALESCE(SUM(file_count), 0) AS total FROM `{column_index_table}`"
    for row in bq.query_rows(sql):
        return int(row["total"])
    return 0


def _validate_table_id(table_id: str) -> None:
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected project.dataset.table, got {table_id!r}")
    for part in parts:
        if not _BQ_IDENT_RE.fullmatch(part):
            raise ValueError(f"invalid BQ identifier segment {part!r} in {table_id!r}")
