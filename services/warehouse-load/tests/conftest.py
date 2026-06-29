"""Shared test fixtures.

`REPO_ROOT` finds the repo root by walking up looking for the
top-level marker (`.git`). The runlog drift tests and the e2e test
all need it.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from warehouse_load.types import RawRunlogRow


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("could not find repo root (no .git ancestor)")


REPO_ROOT: Path = _find_repo_root()


@pytest.fixture
def ingest_runlog_dir() -> Path:
    return REPO_ROOT / "services" / "ingest" / "runlog"


@pytest.fixture
def schemas_dir() -> Path:
    return REPO_ROOT / "infra" / "terraform" / "schemas"


def make_row(
    *,
    source_url: str = "https://example.org/data.csv",
    document_id: str | None = None,
    file_format: str = "csv",
    ingestion_status: str = "success",
    ingested_at: datetime | None = None,
    metadata_modified: datetime | None = None,
    organization_code: str = "fin",
    package_id: str | None = "d2dcdf2a-3a1f-4f3c-8c0a-3b5f0e0a1c7e",
) -> RawRunlogRow:
    """Minimal RawRunlogRow builder for unit tests."""
    if document_id is None:
        document_id = "a" * 64
    if ingested_at is None:
        ingested_at = datetime(2026, 6, 1, tzinfo=UTC)
    if metadata_modified is None:
        metadata_modified = datetime(2026, 5, 1, tzinfo=UTC)
    return RawRunlogRow(
        country_code="ca",
        source_code="ckan-opencanada",
        organization_code=organization_code,
        document_id=document_id,
        source_url=source_url,
        package_id=package_id,
        gcs_uri=None,
        checksum="b" * 64,
        etag=None,
        http_last_modified=None,
        resource_last_modified=None,
        file_format=file_format,
        declared_format="CSV",
        language="en",
        title="Example",
        document_type=None,
        subjects=["economics_and_industry"],
        published_date=None,
        metadata_modified=metadata_modified,
        ingested_at=ingested_at,
        ingestion_status=ingestion_status,  # type: ignore[arg-type]
        quarantine_reason=None,
        run_id="run-1",
    )
