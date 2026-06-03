"""Deploy-time configuration loaded from environment variables.

Per-run intent (subject, formats, dry-run, limit-orgs) is **not** here —
that lives on the CLI as a `RunRequest`. Env is for things that change
between deploys; CLI is for things that change between invocations.

A `.env` file at the repo root (or anywhere up the cwd chain) is also
loaded — so you can set `INGEST_GCP_PROJECT_ID=...` once and stop
typing it on every invocation. Real environment variables still
override `.env` values; `.env` overrides the class defaults.

BigQuery-related settings (dataset / table names) are absent for now;
they return when the BQ catalog task lands.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import find_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_sources_yaml() -> Path:
    """Walk up from cwd looking for `infra/ingest_sources.yaml`.

    Lets the operator run `uv run ingest` from `services/ingest/` (or
    anywhere else inside the repo) without having to set
    `INGEST_SOURCES_CONFIG_PATH` by hand. Falls back to the bare
    relative path if no candidate is found — `load_sources()` will then
    raise a clear FileNotFoundError pointing at the missing path.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "infra" / "ingest_sources.yaml"
        if candidate.is_file():
            return candidate
    return Path("infra/ingest_sources.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    # Project-wide concept — same value across every service in this repo —
    # so we accept both the service-prefixed and the unprefixed env names.
    # `INGEST_GCP_PROJECT_ID` wins if both are set.
    gcp_project_id: str = Field(
        validation_alias=AliasChoices("INGEST_GCP_PROJECT_ID", "GCP_PROJECT_ID"),
    )

    gcs_bucket: str = "maplequery-raw"

    sources_config_path: Path = Field(default_factory=_find_sources_yaml)

    # The pipeline appends per-resource records here; the follow-up BQ
    # loader reads them to populate `raw.documents`.
    runlog_dir: Path = Path("runlog")

    request_timeout_seconds: float = 60.0
    inter_request_delay_seconds: float = 0.5
    max_retries: int = 3

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
