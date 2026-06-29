"""Shared type aliases + frozen dataclasses used across layers.

This module only carries pure data shapes — no business logic, no
external imports beyond stdlib. Other layers (`config`, `clients`,
`core`) import from here.
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

    Serialised as JSONL to a local run log (see `core/runlog.py`). A
    follow-up loader reads the JSONL into BigQuery's `raw.documents`
    table — the shape here matches the eventual table schema so the
    loader is a straight read-and-insert.
    """

    country_code: str
    source_code: str
    organization_code: str

    document_id: str
    source_url: str
    package_id: str  # CKAN package UUID (Dataset.id); parent of this resource.
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
