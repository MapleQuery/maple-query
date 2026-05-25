"""Deploy-time configuration loaded from environment variables.

Per-run intent (subject, formats, dry-run, limit-orgs) is **not** here —
that lives on the CLI as a `RunRequest`. Env is for things that change
between deploys; CLI is for things that change between invocations.

BigQuery-related settings (dataset / table names) are absent for now;
they return when the BQ catalog task lands.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=None)

    gcp_project_id: str

    gcs_bucket: str = "maplequery-raw"

    sources_config_path: Path = Path("infra/ingest_sources.yaml")

    # The pipeline appends per-resource records here; the follow-up BQ
    # loader reads them to populate `raw.documents`.
    runlog_dir: Path = Path("runlog")

    request_timeout_seconds: float = 60.0
    inter_request_delay_seconds: float = 0.5
    max_retries: int = 3

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
