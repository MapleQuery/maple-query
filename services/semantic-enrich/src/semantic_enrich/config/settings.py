"""Deploy-time configuration loaded from environment variables.

Prefix `WHENRICH_` matches the WHLOAD_ / WHINGEST_ pattern in sibling
services. Per-run intent — `--run-id`, `--limit-packages`,
`--limit-package-ids`, `--limit-orgs`, `--dry-run` — is CLI-only.
"""
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


def _find_staging_dir() -> Path:
    """Default to `services/semantic-enrich/stage/` relative to repo root.

    Walks up looking for the service dir so `uv run semantic-enrich ...`
    invoked from anywhere inside the repo lands its JSONL in the same
    place that the rsync runbook expects.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "services" / "semantic-enrich"
        if candidate.is_dir():
            return candidate / "stage"
    return Path("stage")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHENRICH_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    # ── 4.3 fields ──
    generation_model: str = "Qwen/Qwen2.5-14B-Instruct"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda"

    # Optional HF cache override. Falls through to the HF default
    # (`$HF_HOME` or `~/.cache/huggingface`) when unset.
    hf_cache_dir: Path | None = None

    # ── 4.4 additions ──

    # Project-wide concept — same value across every service in this
    # repo — so we accept both the service-prefixed and the unprefixed
    # env names. WHENRICH_GCP_PROJECT_ID wins if both are set.
    gcp_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WHENRICH_GCP_PROJECT_ID", "GCP_PROJECT_ID"),
    )

    # BQ identifiers.
    bq_dataset_raw: str = "raw"
    bq_documents_table: str = "documents"
    bq_rows_table: str = "rows"
    bq_dataset_semantic: str = "semantic"
    bq_datasets_table: str = "datasets"

    # Generation tunables.
    generation_max_tokens: int = 800
    generation_temperature: float = 0.0
    generation_dtype: str = "bfloat16"

    # Embedding tunables.
    embedding_dim: int = 1024
    embedding_batch_size: int = 64

    # Sampling.
    sample_rows_per_package: int = 10
    sample_column_cap: int = 40

    # Staging.
    staging_dir: Path = Field(default_factory=_find_staging_dir)
    flush_every_n_packages: int = 500

    # Behaviour knobs.
    max_retries: int = 3
    dry_run: bool = False

    # Schema source of truth.
    schemas_dir: Path = Field(default_factory=_find_schemas_dir)

    # Run identity. A new UUID per process by default; operators
    # override with --run-id to resume an existing stage dir.
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
