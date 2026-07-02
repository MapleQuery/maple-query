"""Deploy-time configuration loaded from environment variables.

Prefix `WHENRICH_` matches the WHLOAD_ / WHINGEST_ pattern in sibling
services. Per-run intent — `--run-id`, `--limit-packages`,
`--limit-package-ids`, `--limit-orgs`, `--dry-run` — is CLI-only.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import find_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_schemas_dir() -> Path:
    """Walk up from cwd looking for `infra/terraform/schemas/`."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "infra" / "terraform" / "schemas"
        if candidate.is_dir():
            return candidate
    return Path("infra/terraform/schemas")


def _find_service_dir() -> Path:
    """Return `services/semantic-enrich/` relative to the repo root.

    Walks up from cwd looking for the service dir so any file default
    keyed off it resolves to the same absolute path whether the CLI is
    invoked from the repo root or from a subdir.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "services" / "semantic-enrich"
        if candidate.is_dir():
            return candidate
    return Path("services") / "semantic-enrich"


def _find_staging_dir() -> Path:
    """Default to `services/semantic-enrich/stage/` relative to repo root."""
    return _find_service_dir() / "stage"


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
    bq_columns_table: str = "columns"

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

    # `datasets-extract` fan-out. Each package costs two BQ queries
    # (column-union + sample-rows); running them serially against
    # ~3,700 packages takes ~2-3 hours at ~1.5s round-trip per query.
    # 16 concurrent workers cut that to ~12 minutes and stay well
    # under BQ's per-user concurrent-jobs ceiling.
    extract_concurrency: int = 16

    # ── 4.5 columns enrichment ──
    # Per §7.3: 100 columns/chunk balances output-token budget against
    # call count. A 1,383-column outlier becomes 14 chunks.
    column_chunk_size: int = 100
    # Wide-package safety belt (§7.4). 20 x 100 = 2,000 columns —
    # ~45% headroom over the corpus's 1,383-column max.
    column_chunk_max_chunks_per_package: int = 20
    # ≤10 distinct sample values per column (parent §10).
    column_sample_values_cap: int = 10
    # Column-name allowlist (§5.3). Admits identifier-ish, hyphens,
    # dots, slashes, spaces. Rejects quotes, backslashes, backticks,
    # `$`-prefixed paths, and anything ≥ 200 chars.
    column_name_allowlist_re: str = r"^[A-Za-z0-9_][A-Za-z0-9_\-./ ]{0,199}$"
    # Single-retry temperature on per-chunk invariant violation (§8.3).
    # `temperature=0` is deterministic — a retry would reproduce the
    # same failure — so a small bump samples a different completion.
    column_chunk_retry_temperature: float = 0.2

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

    # ── 4.7 OpenAI embedding swap ──
    # After 4.7 lands, datasets-embed / columns-embed and the reembed
    # subcommands call OpenAI text-embedding-3-small (1536-dim).
    #
    # Both the WHENRICH_-prefixed and unprefixed forms are accepted;
    # the prefixed form wins, same posture as gcp_project_id.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WHENRICH_OPENAI_API_KEY", "OPENAI_API_KEY"
        ),
    )
    openai_embedding_model: str = "text-embedding-3-small"
    # Sanity-check knob: the reembed pass and the retargeted embed
    # subcommands assert every vector matches this before writing.
    # A mismatch means the operator pointed at a different model and
    # forgot to update the dim.
    openai_embedding_dim: int = 1536
    openai_embedding_batch_size: int = 128
    openai_request_timeout_s: float = 30.0
    openai_max_retries: int = 3

    # ── 4.6 retrieval-validation harness ──
    # OpenAI generation config. The embedding config above is reused as-is.
    openai_generation_model: str = "gpt-4o"
    openai_generation_temperature: float = 0.0
    openai_generation_max_tokens: int = 1024

    # Retrieval knobs. See PRD §6.4 for the k rationale.
    eval_k_packages: int = 5
    eval_k_columns: int = 15

    # SQL guard knobs. Cost cap default catches hallucinated full-scan
    # queries against `raw.rows` (~50 GB unfiltered) cheaply; operators
    # raise it deliberately via --max-bytes-billed.
    eval_max_bytes_billed: int = 50 * 1024 * 1024 * 1024
    eval_query_timeout_ms: int = 30_000
    eval_dry_run_timeout_ms: int = 10_000
    eval_row_limit: int = 100
    eval_allowed_datasets: tuple[str, ...] = ("raw", "semantic")
    eval_forbidden_keywords: tuple[str, ...] = (
        "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP",
        "ALTER", "GRANT", "REVOKE", "TRUNCATE", "CALL",
    )

    # Paths. Defaults land at the committed fixture + template locations
    # relative to the repo root; walked up from cwd like staging_dir.
    eval_questions_path: Path = Field(
        default_factory=lambda: _find_service_dir() / "eval" / "questions.yaml"
    )
    eval_prompt_template: Path = Field(
        default_factory=lambda: (
            _find_service_dir() / "eval" / "prompts" / "sql_generation.j2"
        )
    )
    eval_reports_dir: Path = Field(
        default_factory=lambda: _find_service_dir() / "eval" / "reports"
    )

    # Run identity. A new UUID per process by default; separate from
    # `run_id` so an eval run inside a larger session gets its own id.
    eval_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
