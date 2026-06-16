"""Deploy-time configuration loaded from environment variables.

Per-run intent (--dry-run, --since, --limit-orgs) is CLI-only: env
for things that change between deploys, CLI for things that change
between invocations.

The default `runlog_local_dir` walks up from cwd looking for
`services/ingest/runlog/` so `uv run warehouse-load documents`
works from anywhere inside the repo without setting an env var.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import find_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_ingest_runlog_dir() -> Path:
    """Walk up from cwd looking for `services/ingest/runlog/`."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "services" / "ingest" / "runlog"
        if candidate.is_dir():
            return candidate
    # Fall through; the runlog reader will raise a clear error if it's
    # actually used.
    return Path("services/ingest/runlog")


def _find_schemas_dir() -> Path:
    """Walk up from cwd looking for `infra/terraform/schemas/`."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "infra" / "terraform" / "schemas"
        if candidate.is_dir():
            return candidate
    return Path("infra/terraform/schemas")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHLOAD_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    # Project-wide concept — same value across every service in this
    # repo — so we accept both the service-prefixed and the
    # unprefixed env names. WHLOAD_GCP_PROJECT_ID wins if both are set.
    gcp_project_id: str = Field(
        validation_alias=AliasChoices("WHLOAD_GCP_PROJECT_ID", "GCP_PROJECT_ID"),
    )

    bq_dataset_raw: str = "raw"
    bq_documents_table: str = "documents"

    # Runlog source. Both can be set; local first, then GCS.
    runlog_local_dir: Path | None = Field(default_factory=_find_ingest_runlog_dir)
    runlog_gcs_prefix: str | None = None  # e.g. "gs://maplequery-raw/runlog/"

    # Schema source of truth, shared with Terraform's google_bigquery_table.
    schemas_dir: Path = Field(default_factory=_find_schemas_dir)

    # Behaviour
    batch_size: int = 5_000
    max_retries: int = 3

    # Run identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
