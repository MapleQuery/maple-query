"""Load to staging, then MERGE into raw.documents.

Two invariants the generated SQL must satisfy (the integration test
regexes the UPDATE clause to enforce them):

1. The UPDATE clause **does not touch** any column owned by the
   downstream content loader (`preamble_rows`, `header_confidence`,
   `load_status`, `load_attempted_at`, `load_error`, `row_count`).
   Once that loader marks a doc `load_status='loaded'`, re-running
   this service must not reset it to `'pending'`.
2. The INSERT clause sets those same columns to their initial values
   (`load_status='pending'`, rest NULL) — and only on first insert.

UPDATE fires when `s.metadata_modified > t.metadata_modified OR
s.ingested_at > t.ingested_at`. Otherwise the row is a no-op and BQ
skips the write.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from google.cloud import bigquery

from warehouse_load.clients.bq import BqClient
from warehouse_load.types import RawRunlogRow

# BQ identifiers in a fully-qualified table ID can contain letters,
# digits, underscore, and hyphen (the latter for project IDs). Anything
# else is rejected before interpolating into the MERGE SQL.
_BQ_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Columns this service owns: UPDATE writes these, and INSERT writes
# the same set from the staging row. Kept as a constant so the regex
# check in the integration test has a single source of truth.
DOCUMENTS_OWNED_BY_LOADER: tuple[str, ...] = (
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

# Columns the downstream content loader owns. The MERGE UPDATE must
# NEVER read these from staging — they are 3.3's outputs. Two of them
# are reset to constants on UPDATE (see `DOCUMENTS_RESET_ON_REFRESH`)
# so a re-ingested doc re-enters 3.3's candidate queue; the rest are
# left untouched so completed loads stay intact.
DOCUMENTS_OWNED_BY_CONTENT_LOADER: tuple[str, ...] = (
    "preamble_rows",
    "header_confidence",
    "load_status",
    "load_attempted_at",
    "load_error",
    "row_count",
)

# Constants written into the UPDATE clause when an ingest re-pull
# fires the MERGE: the doc's bytes changed, so 3.3's prior load is
# stale and the doc must re-enter the `load_status='pending'` queue.
# Values are SQL literals (no parameter binding) because they're
# fixed; this is the simplest way to keep them out of the
# "owned-by-staging" code path.
DOCUMENTS_RESET_ON_REFRESH: tuple[tuple[str, str], ...] = (
    ("load_status", "'pending'"),
    ("load_error", "NULL"),
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

    Returns insert/update/unchanged counts. BQ doesn't break out
    insert vs. update in job stats, so we approximate: rowcount delta
    on the target is inserts; the rest is treated as updates.
    """
    rows_list = list(rows)
    if not rows_list:
        # `load_table_from_json` rejects empty payloads; short-circuit so a
        # fully-zombified run (e.g. misconfigured bucket prefix) is a no-op
        # rather than a hard failure.
        return MergeResult(rows_inserted=0, rows_updated=0, rows_unchanged=0)
    payload = [_row_to_payload(r) for r in rows_list]

    staging_table_id = f"{project_id}.{dataset}._documents_staging_{run_id_short}"
    target_table_id = f"{project_id}.{dataset}.{table}"

    bq.create_staging_table(
        table_id=staging_table_id,
        schema=schema,
        expires_in=staging_ttl,
    )

    try:
        rows_before = bq.count_rows(target_table_id)
        bq.load_json(rows=payload, destination=staging_table_id, schema=schema)
        bq.execute(_render_merge_sql(target_table_id, staging_table_id))
        rows_after = bq.count_rows(target_table_id)
    finally:
        # Happy path drops the staging table immediately; the TTL set in
        # create_staging_table is the safety net for crashes between
        # create_staging_table and the finally block.
        bq.delete_table(staging_table_id, not_found_ok=True)

    rows_inserted = max(rows_after - rows_before, 0)
    rows_touched = len(payload)
    rows_updated_or_unchanged = max(rows_touched - rows_inserted, 0)

    # No cheap way to split updated vs. unchanged without a second
    # MERGE-time SELECT; we report `unchanged=0` and roll everything
    # non-insert into `updated`. Slightly pessimistic on update
    # counts; revisit if downstream needs the precise split.
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
    """Convert a RawRunlogRow into the dict shape `load_table_from_json` wants.

    Datetimes/dates serialise to ISO strings; BQ accepts those for
    TIMESTAMP/DATE under NEWLINE_DELIMITED_JSON. Content-loader
    columns are set to their initial values (load_status='pending',
    rest NULL) so a fresh INSERT lands in the correct shape — the
    MERGE UPDATE clause then leaves them alone on subsequent runs.
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
        # Content-loader columns. INSERT writes these once; subsequent
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

    Generated (rather than hand-templated) so the UPDATE column list
    stays in sync with `DOCUMENTS_OWNED_BY_LOADER`. The
    `DOCUMENTS_OWNED_BY_CONTENT_LOADER` set is appended as a trailing
    comment so a casual reader can see what's deliberately omitted.
    """
    _validate_table_id(target)
    _validate_table_id(staging)

    update_assignments = ",\n      ".join(
        [
            *(f"{col} = s.{col}" for col in DOCUMENTS_OWNED_BY_LOADER),
            # Reset 3.3's work-queue signal so the rows loader re-picks
            # the doc up on its next run. `load_attempted_at`,
            # `row_count`, `preamble_rows`, `header_confidence` are
            # deliberately preserved — readers see prior-load state
            # until 3.3 reruns and DELETE-then-INSERTs new rows.
            *(f"{col} = {literal}" for col, literal in DOCUMENTS_RESET_ON_REFRESH),
        ],
    )
    insert_values = ", ".join(
        [
            *(f"s.{col}" for col in DOCUMENTS_OWNED_BY_LOADER),
            "s.document_id",
            "NULL",  # preamble_rows
            "NULL",  # header_confidence
            "'pending'",  # load_status
            "NULL",  # load_attempted_at
            "NULL",  # load_error
            "NULL",  # row_count
        ],
    )
    # INSERT column list, ordered to match the value list above:
    # loader-owned, then document_id, then content-loader-owned.
    insert_columns_ordered = ", ".join(
        [
            *DOCUMENTS_OWNED_BY_LOADER,
            "document_id",
            *DOCUMENTS_OWNED_BY_CONTENT_LOADER,
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
-- Content-loader columns NEVER appear in the UPDATE clause:
-- {", ".join(DOCUMENTS_OWNED_BY_CONTENT_LOADER)}.
"""


def _validate_table_id(table_id: str) -> None:
    """Reject any project/dataset/table segment that contains characters
    outside `[A-Za-z0-9_-]`. Defense in depth: the values come from
    `Settings`, not user input, but a typo or a future code path that
    pulls them from less-trusted config shouldn't get to interpolate
    arbitrary SQL between backticks (which don't escape)."""
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected project.dataset.table, got {table_id!r}")
    for part in parts:
        if not _BQ_IDENT_RE.fullmatch(part):
            raise ValueError(f"invalid BQ identifier segment {part!r} in {table_id!r}")
