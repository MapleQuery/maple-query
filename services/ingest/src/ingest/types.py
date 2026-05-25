"""Shared type aliases + frozen dataclasses used across layers.

Per PRD §3, this module only carries pure data shapes — no business
logic, no external imports beyond stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

QuarantineReason = Literal[
    "download_failed",
    "oversize",
    "truncated_body",
    "unreadable_encoding",
    "path_collision",
]


@dataclass(frozen=True)
class DocumentRow:
    """One ingested-resource record.

    In Phase A1 these are serialised as JSONL to a local run log
    (see `core/runlog.py`). Phase A2 loads the JSONL into
    `raw.documents` in BigQuery; the shape matches the PRD §14.2 schema.
    """

    country_code: str
    source_code: str
    organization_code: str

    document_id: str
    source_url: str
    gcs_uri: str | None  # NULL when ingestion failed before upload

    checksum: str | None  # sha256 hex of body; NULL on failure
    etag: str | None
    http_last_modified: datetime | None
    resource_last_modified: datetime | None

    file_format: str
    declared_format: str | None

    language: str  # 'en' | 'fr' | 'unknown'
    title: str | None
    document_type: str | None
    subjects: list[str]
    published_date: date | None
    metadata_modified: datetime

    ingested_at: datetime
    ingestion_status: str  # 'success' | 'quarantined' | 'failed'
    quarantine_reason: str | None
    run_id: str
