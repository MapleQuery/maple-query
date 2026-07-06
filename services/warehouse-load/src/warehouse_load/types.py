"""Shared type aliases, dataclasses, and the runlog pydantic model.

Pure data shapes only — no business logic, no client SDKs imported
here. Lives at the bottom of the layer stack so every other module
can depend on it.

Closed-enum sets (`LANGUAGES`, `INGESTION_STATUSES`, `FILE_FORMATS`)
are exported alongside the pydantic `Literal[...]` declarations so
the enum-drift test can read from a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# Closed-enum sets. The enum-drift test compares these against the
# union of values seen in real runlog files. Update both the
# Literal[...] declarations below and the matching set in lockstep.
LANGUAGES: frozenset[str] = frozenset({"en", "fr", "unknown"})
INGESTION_STATUSES: frozenset[str] = frozenset({"success", "quarantined", "failed"})
# `file_format` is what ingest emits pre-filter — the kitchen-sink
# set the model must accept. The loader narrows to "csv" downstream;
# values that show up in the runlog but aren't here mean ingest
# added a format or we missed one. Fix the drift before loading.
FILE_FORMATS: frozenset[str] = frozenset(
    {
        "csv", "tsv",
        "xlsx", "xls", "ods",
        "json", "xml",
        "zip", "7z", "tar", "gz",
        "html", "htm",
        "pdf", "docx", "doc", "txt", "rtf",
        "unknown",
    },
)


class RawRunlogRow(BaseModel):
    """One JSONL row from `services/ingest/runlog/*.jsonl`.

    Shape matches ingest's `DocumentRow` in
    `services/ingest/src/ingest/types.py`. `extra="ignore"` is
    forward-compat for new ingest fields; the key-drift test catches
    them so additions don't silently bypass the warehouse.
    """

    model_config = ConfigDict(extra="ignore")

    country_code: str
    source_code: str
    organization_code: str

    document_id: str
    source_url: str
    # Best-effort link to the CKAN parent package. None for resources
    # whose CKAN parent has been deleted since ingest, or for resources
    # harvested from non-CKAN sources. The BQ column is permanently
    # NULLABLE for the same reason — see raw_documents.json.
    package_id: str | None = None
    gcs_uri: str | None = None

    checksum: str | None = None
    etag: str | None = None
    http_last_modified: datetime | None = None
    resource_last_modified: datetime | None = None

    file_format: str
    declared_format: str | None = None

    language: Literal["en", "fr", "unknown"]
    title: str | None = None
    document_type: str | None = None
    subjects: list[str] = []
    published_date: date | None = None
    metadata_modified: datetime

    ingested_at: datetime
    ingestion_status: Literal["success", "quarantined", "failed"]
    quarantine_reason: str | None = None
    run_id: str

    @field_validator(
        "http_last_modified",
        "resource_last_modified",
        "metadata_modified",
        "ingested_at",
        mode="after",
    )
    @classmethod
    def _anchor_naive_to_utc(cls, value: datetime | None) -> datetime | None:
        # Ingest writes tz-aware (+00:00) today; this guards downstream
        # comparisons (--since, dedupe tie-break) against TypeError if a
        # naive timestamp ever slips through.
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


@dataclass(frozen=True)
class DocumentRow:
    """A `raw.documents` row, re-hydrated for the rows loader.

    Carries only the columns 3.3 actually reads (the candidate-query
    projection plus the columns 3.3 writes back). The full row shape
    lives in `infra/terraform/schemas/raw_documents.json`.
    """

    document_id: str
    organization_code: str
    source_url: str
    gcs_uri: str | None
    file_format: str
    declared_format: str | None
    checksum: str | None
    resource_last_modified: datetime | None


# Header confidence is a closed enum on the wire; pinned so the
# document_status module and the column-index test share one set.
HEADER_CONFIDENCES: frozenset[str] = frozenset({"single", "multi_row", "low"})

# load_status values written by 3.3. The schema is permissive (STRING),
# but the loader only ever emits these. Pinned here so test assertions
# read against one source of truth.
LOAD_STATUSES: frozenset[str] = frozenset(
    {"pending", "loaded", "blob_missing", "parse_failed"},
)


@dataclass(frozen=True)
class SniffResult:
    """Result of the CSV header-sample sniff (`core/csv_sniff.py`)."""

    delimiter: Literal[",", "\t", ";"]
    encoding: str
    sniff_bytes: int


@dataclass(frozen=True)
class HeaderResult:
    """Result of the header-detection state machine (`core/header_detect.py`).

    `header_rows` is the 1 or 2 source rows used to compose the keys;
    `body_start_index` is the 0-based row index where the body begins
    (i.e. polars' `skip_rows` argument). `keys` is the list of
    normalised column keys in column order — the row-stream consumer
    pairs each cell against the matching `keys[i]`.
    """

    body_start_index: int
    header_rows: tuple[tuple[str, ...], ...]
    preamble_rows: tuple[tuple[str, ...], ...]
    confidence: Literal["single", "multi_row", "low"]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class RowsRunRequest:
    """Per-invocation intent for `warehouse-load rows`. Built by the CLI."""

    limit_orgs: tuple[str, ...]
    limit_documents: tuple[str, ...]
    status: str  # candidate-query filter; default "pending"
    force: bool  # replay already-loaded docs
    concurrency: int | None  # None = Settings default
    dry_run: bool
    refresh_column_index: bool


@dataclass(frozen=True)
class RowsRunSummary:
    """End-of-run roll-up. Printed as JSON; same shape as the
    `rows_load_finish` log event."""

    run_id: str
    dry_run: bool
    candidate_count: int
    docs_loaded: int
    docs_blob_missing: int
    docs_parse_failed: int
    docs_header_low_confidence: int
    docs_skipped_already_loaded: int
    rows_merged: int
    column_index_refreshed: bool
    duration_ms: int


@dataclass(frozen=True)
class ColumnIndexRefreshResult:
    """End-of-refresh summary for `core/column_index.py`."""

    unique_cols: int
    total_doc_col_pairs: int
    duration_ms: int


@dataclass(frozen=True)
class DocumentsRunSummary:
    """End-of-run roll-up. Printed as JSON; same shape as the
    `documents_load_finish` log event."""

    run_id: str
    dry_run: bool
    runlog_files_seen: int
    runlog_rows_seen: int
    runlog_parse_errors: int
    rows_filtered_not_csv: int
    rows_filtered_not_success: int
    rows_filtered_blob_missing: int
    rows_deduped: int
    rows_kept: int
    documents_inserted: int
    documents_updated: int
    documents_unchanged: int
    # True when --no-bucket-check was used (or no bucket was configured
    # in dry-run). Lets downstream tooling distinguish "0 zombies
    # because none existed" from "0 zombies because we didn't check."
    bucket_check_skipped: bool
    duration_ms: int
