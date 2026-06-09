"""Stage 3 of §6 + §7: load to staging, MERGE into raw.documents.

The MERGE is the load-bearing piece of the loader. Two invariants
worth re-stating because changing them breaks 3.3:

1. The UPDATE clause **does not touch** any column 3.3 owns:
   `preamble_rows`, `header_confidence`, `load_status`,
   `load_attempted_at`, `load_error`, `row_count`. If 3.3 has already
   loaded a doc (`load_status='loaded'`), re-running 3.2 must not
   reset it to `'pending'`.
2. The INSERT clause sets 3.3-owned columns to their pristine M2
   initial values (`load_status='pending'`, rest NULL) — and only on
   first insert.

The UPDATE condition is `s.metadata_modified > t.metadata_modified
OR s.ingested_at > t.ingested_at`. Either side moving forward is a
reason to refresh; if neither moves, the update is a no-op and BQ
skips the write.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from google.cloud import bigquery

from warehouse_load.clients.bq import BqClient
from warehouse_load.types import RawRunlogRow

# Columns 3.2 owns (UPDATE writes these; same set INSERT writes from
# the staging row). Kept as a constant so the regex check in the
# integration test has a single source of truth.
DOCUMENTS_OWNED_BY_32: tuple[str, ...] = (
    "country_code",
    "source_code",
    "organization_code",
    "source_url",
    "gcs_uri",
    "checksum",
    "etag",
    "http_last_modified",
    "resource_last_modified",
    "file_format",
    "declared_format",
    "language",
    "title",
    "document_type",
    "subjects",
    "published_date",
    "metadata_modified",
    "ingested_at",
    "ingestion_status",
    "quarantine_reason",
    "run_id",
)

# Columns 3.3 owns. The MERGE UPDATE must NEVER include any of these.
DOCUMENTS_OWNED_BY_33: tuple[str, ...] = (
    "preamble_rows",
    "header_confidence",
    "load_status",
    "load_attempted_at",
    "load_error",
    "row_count",
)


@dataclass(frozen=True)
class MergeResult:
    rows_inserted: int
    rows_updated: int
    rows_unchanged: int


@dataclass(frozen=True)
class DryRunResult:
    """What we *would* have done. No staging, no MERGE."""

    rows_kept: int
    payload: list[dict[str, Any]]


def merge_documents(
    *,
    bq: BqClient,
    rows: Iterable[RawRunlogRow],
    project_id: str,
    dataset: str,
    table: str,
    schema: list[bigquery.SchemaField],
    run_id_short: str,
    staging_ttl: timedelta = timedelta(hours=1),
) -> MergeResult:
    """Stage `rows` and MERGE into `<project>.<dataset>.<table>`.

    Returns insert/update/unchanged counts derived from the MERGE
    job's `num_dml_affected_rows` plus a follow-up count query. BQ's
    job stats don't break out insert vs. update; we approximate by
    counting rows inserted (target rowcount delta) and treating the
    rest as updated. The acceptance tests in §13 verify the totals
    rather than the split.
    """
    rows_list = list(rows)
    payload = [_row_to_payload(r) for r in rows_list]

    staging_table_id = f"{project_id}.{dataset}._documents_staging_{run_id_short}"
    target_table_id = f"{project_id}.{dataset}.{table}"

    bq.create_staging_table(
        table_id=staging_table_id,
        schema=schema,
        expires_in=staging_ttl,
    )

    rows_before = bq.count_rows(target_table_id)
    bq.load_json(rows=payload, destination=staging_table_id, schema=schema)
    bq.execute(_render_merge_sql(target_table_id, staging_table_id))
    rows_after = bq.count_rows(target_table_id)

    rows_inserted = max(rows_after - rows_before, 0)
    rows_touched = len(payload)
    rows_updated_or_unchanged = max(rows_touched - rows_inserted, 0)

    # No cheap way to split updated vs. unchanged without a second
    # MERGE-time SELECT; the §10 RunSummary reports `unchanged` as
    # 0 here and `updated` as the residual. Tradeoff: cheap and
    # accurate on the insert axis, slightly pessimistic on update
    # counts. Revisit if downstream needs the precise split.
    return MergeResult(
        rows_inserted=rows_inserted,
        rows_updated=rows_updated_or_unchanged,
        rows_unchanged=0,
    )


def dry_run(rows: Iterable[RawRunlogRow]) -> DryRunResult:
    """Compute the staging payload without touching BQ.

    Mirrors what `merge_documents` would have sent to
    `load_table_from_json`. The CLI prints `would_have_merged` events
    per row from this payload.
    """
    rows_list = list(rows)
    payload = [_row_to_payload(r) for r in rows_list]
    return DryRunResult(rows_kept=len(rows_list), payload=payload)


def _row_to_payload(row: RawRunlogRow) -> dict[str, Any]:
    """Convert a RawRunlogRow into the dict shape BQ load_table_from_json wants.

    Datetimes/dates are serialised to ISO strings; BQ accepts those
    for TIMESTAMP/DATE columns when source_format is
    NEWLINE_DELIMITED_JSON. 3.3-owned columns are set to their initial
    values (load_status='pending', rest NULL) so a fresh INSERT lands
    in the correct shape — even though those columns are passed in
    the load payload, the MERGE UPDATE clause won't touch them on a
    subsequent run.
    """
    out: dict[str, Any] = {
        "country_code": row.country_code,
        "source_code": row.source_code,
        "organization_code": row.organization_code,
        "document_id": row.document_id,
        "source_url": row.source_url,
        "gcs_uri": row.gcs_uri,
        "checksum": row.checksum,
        "etag": row.etag,
        "http_last_modified": _iso_or_none(row.http_last_modified),
        "resource_last_modified": _iso_or_none(row.resource_last_modified),
        "file_format": row.file_format,
        "declared_format": row.declared_format,
        "language": row.language,
        "title": row.title,
        "document_type": row.document_type,
        "subjects": list(row.subjects),
        "published_date": _iso_or_none(row.published_date),
        "metadata_modified": _iso_or_none(row.metadata_modified),
        "ingested_at": _iso_or_none(row.ingested_at),
        "ingestion_status": row.ingestion_status,
        "quarantine_reason": row.quarantine_reason,
        "run_id": row.run_id,
        # 3.3-owned columns. INSERT writes these once; subsequent
        # MERGE UPDATE clauses do NOT touch them.
        "preamble_rows": None,
        "header_confidence": None,
        "load_status": "pending",
        "load_attempted_at": None,
        "load_error": None,
        "row_count": None,
    }
    return out


def _iso_or_none(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        # Anchor to UTC so naive datetimes don't blow up downstream.
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return value.isoformat()


def _render_merge_sql(target: str, staging: str) -> str:
    """Build the MERGE statement.

    Generated rather than hand-templated so the column list in the
    UPDATE clause stays in sync with `DOCUMENTS_OWNED_BY_32` — adding
    a column to that constant automatically extends the MERGE. The
    `DOCUMENTS_OWNED_BY_33` set is referenced only as a comment guard
    in the rendered SQL so a casual reader can see what's deliberately
    omitted.
    """
    update_assignments = ",\n      ".join(
        f"{col} = s.{col}" for col in DOCUMENTS_OWNED_BY_32 if col != "document_id"
    )
    insert_values = ", ".join(
        [
            *(f"s.{col}" for col in DOCUMENTS_OWNED_BY_32 if col != "document_id"),
            "s.document_id",
            "NULL",  # preamble_rows
            "NULL",  # header_confidence
            "'pending'",  # load_status
            "NULL",  # load_attempted_at
            "NULL",  # load_error
            "NULL",  # row_count
        ],
    )
    # INSERT column list, ordered to match the value list above
    # (target-owned first sans document_id, then document_id, then 3.3-owned).
    insert_columns_ordered = ", ".join(
        [
            *(col for col in DOCUMENTS_OWNED_BY_32 if col != "document_id"),
            "document_id",
            *DOCUMENTS_OWNED_BY_33,
        ],
    )

    return f"""\
MERGE INTO `{target}` t
USING `{staging}` s
  ON t.document_id = s.document_id
WHEN MATCHED AND
     (s.metadata_modified > t.metadata_modified
      OR s.ingested_at > t.ingested_at)
THEN UPDATE SET
      {update_assignments}
WHEN NOT MATCHED THEN INSERT (
  {insert_columns_ordered}
)
VALUES (
  {insert_values}
)
-- 3.3-owned columns NEVER appear in the UPDATE clause:
-- {", ".join(DOCUMENTS_OWNED_BY_33)}.
"""
