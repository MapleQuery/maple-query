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
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_ingest_runlog_dir() -> Path | None:
    """Walk up from cwd looking for `services/ingest/runlog/`.

    Returns None on fall-through so the reader/entrypoint can fail
    loudly (or fall back to GCS) rather than silently iterating a
    non-existent relative path and reporting zero rows.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "services" / "ingest" / "runlog"
        if candidate.is_dir():
            return candidate
    return None


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

    # Tables the rows loader reads/writes. Names live here so test
    # harnesses and the CLI both pick them up from one place.
    bq_rows_table: str = "rows"
    bq_rows_staging_table: str = "rows_staging"
    bq_column_index_table: str = "column_index"

    # Rows-loader concurrency + safety belts. All knobs are env-tunable
    # without code changes (prefix WHLOAD_); defaults match the PRD.
    rows_concurrency: int = 4
    max_bytes_per_doc: int = 600 * 1024 * 1024  # 600 MB; max observed is 533 MB
    max_rows_per_doc: int = 50_000_000  # safety belt; no observed file exceeds 10M
    sniff_buffer_bytes: int = 8192  # first-line delimiter sniff sample
    per_doc_timeout_seconds: int = 900  # 15 min wall-clock ceiling per doc

    # Header detection thresholds (PRD §6.6). Tunable post-backfill if
    # the empirical `header_confidence='low'` rate exceeds 5%.
    body_min_run: int = 20  # stable-run length that anchors body_start_index
    header_lookback: int = 5  # rows above body considered as header / preamble
    body_modal_match_ratio: float = 0.80  # fraction of slice that must match modal
    header_max_cell_chars: int = 200  # any longer = body-like

    # Batch flush trigger: flush when ANY of these thresholds trips.
    # Rows bounds work-loss on MERGE failure; bytes bounds disk
    # footprint of pending JSONLs in /tmp; files bounds open-FD count
    # and per-batch UPDATE round-trips. Whichever fires first wins.
    rows_staging_flush_threshold: int = 500_000
    # 2 GiB total across pending JSONLs. With concurrency=4 and 600 MB
    # max per doc, worst-case in-flight + pending stays comfortable.
    rows_staging_flush_bytes_threshold: int = 2 * 1024 * 1024 * 1024
    # 32 pending JSONL files. Caps the per-flush `record_load_outcome`
    # UPDATE storm and the open-FD pressure between flushes.
    rows_staging_flush_files_threshold: int = 32

    # raw.column_index document_ids cap. 1000 matches 3.1 §4.4.
    column_index_doc_ids_cap: int = 1000

    # Runlog source. Both can be set; local first, then GCS.
    runlog_local_dir: Path | None = Field(default_factory=_find_ingest_runlog_dir)
    runlog_gcs_prefix: str | None = None  # e.g. "gs://maplequery-raw/runlog/"

    # GCS corpus root, used by the documents loader as the source of
    # truth for blob existence. Rows whose `gcs_uri` is absent from
    # this prefix are dropped at load time as `blob_missing`.
    bucket_prefix: str = "gs://maplequery-raw/raw/"

    # Schema source of truth, shared with Terraform's google_bigquery_table.
    schemas_dir: Path = Field(default_factory=_find_schemas_dir)

    # Behaviour
    batch_size: int = 5_000
    max_retries: int = 3

    # Run identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("bucket_prefix")
    @classmethod
    def _validate_bucket_prefix(cls, value: str) -> str:
        if not value.startswith("gs://"):
            raise ValueError(f"bucket_prefix must be a gs:// URI, got {value!r}")
        without_scheme = value[len("gs://"):]
        bucket, _, _ = without_scheme.partition("/")
        if not bucket:
            raise ValueError(f"bucket_prefix missing bucket name: {value!r}")
        if not value.endswith("/"):
            raise ValueError(f"bucket_prefix must end with '/': {value!r}")
        return value
