"""Shared type aliases and dataclasses.

Pure data shapes only. No model SDKs imported here so it can sit at
the bottom of the layer stack and import-free into the smoke-test
result path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pydantic

# Outlines exposes the model object through a factory rather than a
# stable public class. Treat it as opaque at our boundary — the
# function signatures document intent without locking us to a private
# import path that could shift between minor releases.
GenerationModel = Any
EmbeddingModel = Any


class MaxTokensExceededError(RuntimeError):
    """Raised when constrained-JSON generation produced malformed output.

    Outlines enforces JSON-Schema conformance per-token; a parse or
    schema-validation failure means the decoder hit `max_tokens` before
    closing the structure. Callers should re-run with a larger budget,
    not retry blindly.
    """


@dataclass(frozen=True)
class SmokeResult:
    """End-of-smoke roll-up. Maps to exit codes in the CLI:

    - `ok=True`                                  → exit 0
    - `ok=False` and `precondition_failure`      → exit 2
    - any uncaught exception in the runner       → exit 1 (handled by CLI)
    """

    ok: bool
    precondition_failure: str | None
    generation_output: dict[str, Any] | None
    embedding_dim: int | None
    embedding_norm: float | None
    duration_ms: int


# ── 4.4 input/output shapes ──


class PackageResource(pydantic.BaseModel):
    """One resource row from raw.documents, projected to the fields
    the per-package extract loop needs."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str | None = None
    subjects: tuple[str, ...] = ()
    organization_code: str
    file_format: str
    resource_last_modified: datetime | None = None
    row_count: int | None = None


class PackageInputs(pydantic.BaseModel):
    """One package's worth of inputs to the generation pass.

    Materialised by `datasets-extract` (laptop) and consumed by
    `datasets-generate` (GPU). Round-trips through JSONL on disk; no
    BQ client needed by the consumer.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    package_id: str
    resources: tuple[PackageResource, ...]
    column_names: tuple[str, ...]
    column_names_truncated_to: int | None
    representative_document_id: str
    sample_rows: tuple[dict[str, str | None], ...]


class DatasetCard(pydantic.BaseModel):
    """LLM output for one package. Constrained by
    `DATASET_CARD_GUIDED_JSON` at generation time."""

    model_config = pydantic.ConfigDict(extra="forbid")

    package_id: str
    summary: str = pydantic.Field(min_length=50, max_length=1200)
    grain: str | None = None
    measures: list[str] = pydantic.Field(default_factory=list, max_length=20)
    dimensions: list[str] = pydantic.Field(default_factory=list, max_length=20)
    date_range_start: date | None = None
    date_range_end: date | None = None

    @pydantic.field_validator("grain", mode="before")
    @classmethod
    def _empty_grain_is_null(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class StagedDatasetCard(pydantic.BaseModel):
    """On-disk row shape under `stage/<run_id>/datasets/*.jsonl`.

    Carries generation provenance for debugging; the load pass
    projects those fields away when MERGEing into `semantic.datasets`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    package_id: str
    summary: str
    grain: str | None
    measures: list[str]
    dimensions: list[str]
    date_range_start: date | None
    date_range_end: date | None
    embedding: list[float] | None
    generated_at: datetime
    generation_model: str
    generation_model_commit: str | None
    generation_run_id: str
    # Dry-run marker; absent for real runs (default False keeps the
    # field opt-in at the writer level).
    dry_run: bool = False


# ── Per-package outcomes ──


@dataclass(frozen=True)
class PackageGenerateOutcome:
    kind: Literal["generated", "skipped_already_staged", "failed"]
    error: str | None = None


# ── Run summaries (§11.3) ──


@dataclass(frozen=True)
class DatasetsExtractRunSummary:
    run_id: str
    dry_run: bool
    candidate_count: int
    packages_extracted: int
    packages_skipped_already_extracted: int
    packages_failed: int
    flush_files_written: int
    duration_ms: int


@dataclass(frozen=True)
class DatasetsGenerateRunSummary:
    run_id: str
    dry_run: bool
    input_row_count: int
    packages_generated: int
    packages_skipped_already_staged: int
    packages_failed: int
    flush_files_written: int
    duration_ms: int


@dataclass(frozen=True)
class DatasetsEmbedRunSummary:
    run_id: str
    dry_run: bool
    staged_files_seen: int
    rows_seen: int
    embeddings_written: int
    embeddings_skipped_already_embedded: int
    embeddings_failed: int
    duration_ms: int


@dataclass(frozen=True)
class DatasetsLoadRunSummary:
    run_id: str
    dry_run: bool
    coalesced_row_count: int
    embedding_null_count: int
    rows_staged: int
    rows_inserted: int
    rows_updated: int
    rows_unchanged: int
    duration_ms: int


# ── Internal counters used by the runners ──


@dataclass
class Counters:
    """Mutable counters threaded through the per-package loop.

    Frozen summary dataclasses are built from these at end-of-run.
    """

    generated: int = 0
    skipped: int = 0
    failed: int = 0
    extras: dict[str, int] = field(default_factory=dict)


# ── 4.5 input/output shapes ──


class ColumnInputs(pydantic.BaseModel):
    """One package's worth of inputs to the columns-generate pass.

    Materialised by `columns-extract` (laptop) and consumed by
    `columns-generate` (GPU). One JSONL line per package under
    `stage/<run_id>/column_inputs/`.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    package_id: str
    package_title: str | None = None
    package_subjects: tuple[str, ...] = ()
    package_summary: str | None = None
    representative_document_id: str
    column_names: tuple[str, ...]
    sample_values: dict[str, tuple[str, ...]]
    dropped_columns: tuple[str, ...] = ()
    overflow_column_count: int = 0
    extracted_at: datetime


class ColumnOutput(pydantic.BaseModel):
    """One row of the model's per-chunk JSON-array response.

    Constrained by `COLUMNS_GUIDED_JSON_SCHEMA` at generation time.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    column_name: str
    semantic_type: str | None = None
    description: str = pydantic.Field(min_length=20, max_length=600)
    sample_values: list[str] = pydantic.Field(default_factory=list, max_length=10)


class StagedColumnRow(pydantic.BaseModel):
    """On-disk row shape under `stage/<run_id>/columns/*.jsonl`.

    One row per `(package_id, column_name)`. Provenance fields are
    staging-only and projected away by the load pass.

    `generation_failed=True` marks a failure-marker line (an empty
    placeholder so the gap-fill check doesn't reprocess the package
    without operator intervention; see §7.4 + §8.3).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    package_id: str
    column_name: str
    semantic_type: str | None
    description: str
    sample_values: list[str]
    embedding: list[float] | None
    generated_at: datetime
    generation_model: str
    generation_model_commit: str | None
    generation_run_id: str
    generation_failed: bool = False
    failure_reason: str | None = None
    dry_run: bool = False


# ── 4.5 errors ──


class ColumnChunkInvariantError(RuntimeError):
    """Raised when one chunk's model output violates the per-chunk
    1:1 column-name mapping invariant. The chunk is retried once at
    a small temperature bump; two consecutive violations escalate to
    a per-package failure."""


class ColumnPackageInvariantError(RuntimeError):
    """Raised when the concatenated per-package output diverges from
    the input column-name sequence. Belt-and-braces against a chunk →
    concat ordering bug."""


# ── 4.5 run summaries (§11.3) ──


@dataclass(frozen=True)
class ColumnsExtractRunSummary:
    run_id: str
    dry_run: bool
    candidate_count: int
    packages_extracted: int
    packages_skipped_already_extracted: int
    packages_empty: int
    packages_summary_hit: int
    packages_summary_miss: int
    columns_dropped_by_allowlist: int
    flush_files_written: int
    duration_ms: int


@dataclass(frozen=True)
class ColumnsGenerateRunSummary:
    run_id: str
    dry_run: bool
    input_row_count: int
    packages_generated: int
    packages_skipped_already_staged: int
    packages_failed: int
    chunks_total: int
    chunks_retried: int
    columns_generated: int
    flush_files_written: int
    duration_ms: int


@dataclass(frozen=True)
class ColumnsEmbedRunSummary:
    run_id: str
    dry_run: bool
    staged_files_seen: int
    rows_seen: int
    embeddings_written: int
    embeddings_skipped_already_embedded: int
    embeddings_failed: int
    duration_ms: int


@dataclass(frozen=True)
class ColumnsLoadRunSummary:
    run_id: str
    dry_run: bool
    coalesced_row_count: int
    embedding_null_count: int
    failure_marker_count: int
    rows_staged: int
    rows_inserted: int
    rows_updated: int
    rows_unchanged: int
    duration_ms: int
