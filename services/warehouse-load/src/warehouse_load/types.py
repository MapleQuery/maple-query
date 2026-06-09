"""Shared type aliases, dataclasses, and the runlog pydantic model.

Pure data shapes only — no business logic, no client SDKs imported
here. Lives at the bottom of the layer stack so every other module
can depend on it.

Closed-enum sets (`LANGUAGES`, `INGESTION_STATUSES`, `FILE_FORMATS`)
are exported alongside the pydantic `Literal[...]` declarations so the
§11.3 value-drift CI check has a single source of truth to read from.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Closed-enum sets. The §11.3 drift check compares these against
# the union of values seen in real runlog files. Update both the
# Literal[...] declarations below and the matching set in lockstep.
LANGUAGES: frozenset[str] = frozenset({"en", "fr", "unknown"})
INGESTION_STATUSES: frozenset[str] = frozenset({"success", "quarantined", "failed"})
# `file_format` is what 2.2 emits pre-filter — the kitchen-sink set the
# model must accept. 3.2 narrows to "csv" at §6.1; values that show up
# in the runlog but aren't here mean either 2.2 added a format or we
# missed one. Either way, fix the drift before loading.
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

    Shape matches 2.2's `DocumentRow` (`services/ingest/src/ingest/
    types.py`). `extra="ignore"` is forward-compat for new fields
    added by 2.2; the §11.2 key-drift CI check catches them so
    additions don't silently bypass the warehouse.
    """

    model_config = ConfigDict(extra="ignore")

    country_code: str
    source_code: str
    organization_code: str

    document_id: str
    source_url: str
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


@dataclass(frozen=True)
class DocumentsRunSummary:
    """End-of-run roll-up. Printed as JSON; shape mirrors §10.1's
    `documents_load_finish` event payload."""

    run_id: str
    dry_run: bool
    runlog_files_seen: int
    runlog_rows_seen: int
    runlog_parse_errors: int
    rows_filtered_not_csv: int
    rows_filtered_not_success: int
    rows_deduped: int
    rows_kept: int
    documents_inserted: int
    documents_updated: int
    documents_unchanged: int
    duration_ms: int
