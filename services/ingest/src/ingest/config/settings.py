"""Deploy-time configuration loaded from environment variables.

Per-run intent (subject, formats, dry-run, limit-orgs) is **not** here —
that lives on the CLI as a `RunRequest`. See PRD 2.2 §5.1, §11.1.

BigQuery-related settings (dataset / table names) are absent in Phase A1;
they'll return in A2 when the BQ catalog task lands.
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

    # Phase A1 writes the per-resource run log here. A2 reads it to populate BQ.
    runlog_dir: Path = Path("runlog")

    max_file_size_mb: int = 512
    request_timeout_seconds: float = 60.0
    inter_request_delay_seconds: float = 0.5
    max_retries: int = 3

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
