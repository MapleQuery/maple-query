"""Deploy-time configuration from environment variables (prefix CURATE_).

Per-run intent (--dry-run, --out) is CLI-only."""

from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import find_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        env_prefix="CURATE_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    gcp_project_id: str = Field(
        default="",
        validation_alias=AliasChoices("CURATE_GCP_PROJECT_ID", "GCP_PROJECT_ID"),
    )
    bq_dataset_curated: str = "curated"
    gcs_bucket: str = "maplequery-raw"

    # openparliament.ca is volunteer-run: identify ourselves and pace
    # requests. Only the per-person detail fetches are paced; the few
    # paged list calls are not worth slowing.
    openparliament_base: str = "https://api.openparliament.ca"
    openparliament_user_agent: str = "MapleQuery-curate/0.1 (+https://maple-query.vercel.app; weekly batch)"
    detail_requests_per_second: float = 2.0
    request_timeout_seconds: float = 30.0

    # Where ingest lands the Elections Canada contributions archive
    # (`ingest -s government_and_politics -f csv --limit-orgs elections
    # --accept-archives`). The newest object under this prefix whose
    # name contains `contributions_object_contains` is read.
    contributions_prefix: str = "raw/country=ca/source=ckan-opencanada/organization=elections/"
    contributions_object_contains: str = "od_cntrbtn_de_e"

    # House of Commons members' expenditure reports (one CSV a quarter).
    ourcommons_base: str = "https://www.ourcommons.ca"
    ourcommons_user_agent: str = (
        "Mozilla/5.0 (compatible; maplequery-curate/0.1; +https://maple-query.vercel.app)"
    )
    ourcommons_requests_per_second: float = 1.0

    schemas_dir: Path = Field(default_factory=_find_schemas_dir)
    staging_ttl_hours: int = 6
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
