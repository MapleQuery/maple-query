"""`datasets-embed` and `columns-embed` orchestrators.

GPU-box-side. Reads each `stage/<run_id>/<artifact>/*.jsonl` in path-
ascending order, batch-embeds the source field of rows that don't
already carry an embedding, validates each vector, and atomically
rewrites the file with `embedding` populated.

`_embed_files` is parameterised on `artifact`, `row_type`, and a
`text_for_row` callable so the same buffer/flush/validate loop drives
both the `summary` (4.4) and `description` (4.5) embedding passes.
The two public entry points (`run_embed`, `run_columns_embed`) are
thin wrappers that hand the generic core their type-specific
callbacks.

The embed callable is injected so tests can replace it with a
deterministic fake.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import BaseModel

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import embed as embed_mod
from semantic_enrich.core import stage_io
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    ColumnsEmbedRunSummary,
    DatasetsEmbedRunSummary,
    EmbeddingModel,
    StagedColumnRow,
    StagedDatasetCard,
)

EmbedBatchFn = Callable[..., list[list[float]]]
LoadEmbeddingModelFn = Callable[..., EmbeddingModel]


@dataclass(frozen=True)
class EmbedRequest:
    run_id: str
    dry_run: bool
    batch_size: int | None  # None → settings.embedding_batch_size


@dataclass(frozen=True)
class ColumnsEmbedRequest:
    run_id: str
    dry_run: bool
    batch_size: int | None


# ── Datasets entry point (4.4, unchanged surface) ──


def preflight(*, settings: Settings, request: EmbedRequest) -> list[Path]:
    """Returns the sorted list of `stage/<run_id>/datasets/*.jsonl`
    paths. Raises if the dir is missing or empty."""
    return _preflight_files(
        settings=settings,
        run_id=request.run_id,
        artifact="datasets",
        log_name="semantic_enrich.embedding_pass",
        missing_event="datasets_dir_missing",
        missing_command_hint="datasets-generate",
    )


def run_embed(
    *,
    request: EmbedRequest,
    settings: Settings,
    load_embedding_model: LoadEmbeddingModelFn = embed_mod.load_embedding_model,
    embed_batch: EmbedBatchFn = embed_mod.embed_batch,
    logger: structlog.BoundLogger | None = None,
) -> DatasetsEmbedRunSummary:
    log = logger or get_logger("semantic_enrich.embedding_pass")
    started = time.monotonic()
    files = preflight(settings=settings, request=request)
    batch_size = request.batch_size or settings.embedding_batch_size

    log.info(
        "datasets_embed_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        batch_size=batch_size,
        staged_file_count=len(files),
    )

    model = _maybe_load_model(
        load_embedding_model=load_embedding_model,
        settings=settings,
        dry_run=request.dry_run,
        log=log,
    )

    try:
        stats = _embed_files(
            files=files,
            model=model,
            embed_batch=embed_batch,
            batch_size=batch_size,
            dry_run=request.dry_run,
            embedding_dim=settings.embedding_dim,
            row_type=StagedDatasetCard,
            text_for_row=_summary_text,
            log_bind_field="package_id",
            log_bind_value=lambda r: r.package_id,
            text_chars_field="summary_chars",
            log_prefix="datasets",
            run_id=request.run_id,
            log=log,
        )
    finally:
        if model is not None:
            del model
            _drop_cuda_cache()

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = DatasetsEmbedRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        staged_files_seen=len(files),
        rows_seen=stats.rows_seen,
        embeddings_written=stats.written,
        embeddings_skipped_already_embedded=stats.skipped,
        embeddings_failed=stats.failed,
        duration_ms=duration_ms,
    )
    _assert_datasets_invariant(summary, log)
    log.info(
        "datasets_embed_finish",
        run_id=request.run_id,
        embeddings_written=stats.written,
        embeddings_failed=stats.failed,
        duration_ms=duration_ms,
    )
    return summary


# ── Columns entry point (4.5) ──


def preflight_columns(
    *, settings: Settings, request: ColumnsEmbedRequest
) -> list[Path]:
    return _preflight_files(
        settings=settings,
        run_id=request.run_id,
        artifact="columns",
        log_name="semantic_enrich.embedding_pass",
        missing_event="columns_dir_missing",
        missing_command_hint="columns-generate",
    )


def run_columns_embed(
    *,
    request: ColumnsEmbedRequest,
    settings: Settings,
    load_embedding_model: LoadEmbeddingModelFn = embed_mod.load_embedding_model,
    embed_batch: EmbedBatchFn = embed_mod.embed_batch,
    logger: structlog.BoundLogger | None = None,
) -> ColumnsEmbedRunSummary:
    log = logger or get_logger("semantic_enrich.embedding_pass")
    started = time.monotonic()
    files = preflight_columns(settings=settings, request=request)
    batch_size = request.batch_size or settings.embedding_batch_size

    log.info(
        "columns_embed_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        batch_size=batch_size,
        staged_file_count=len(files),
    )

    model = _maybe_load_model(
        load_embedding_model=load_embedding_model,
        settings=settings,
        dry_run=request.dry_run,
        log=log,
    )

    try:
        stats = _embed_files(
            files=files,
            model=model,
            embed_batch=embed_batch,
            batch_size=batch_size,
            dry_run=request.dry_run,
            embedding_dim=settings.embedding_dim,
            row_type=StagedColumnRow,
            text_for_row=_description_text,
            log_bind_field="package_id_column",
            log_bind_value=lambda r: f"{r.package_id}:{r.column_name}",
            text_chars_field="description_chars",
            log_prefix="columns",
            run_id=request.run_id,
            log=log,
        )
    finally:
        if model is not None:
            del model
            _drop_cuda_cache()

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = ColumnsEmbedRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        staged_files_seen=len(files),
        rows_seen=stats.rows_seen,
        embeddings_written=stats.written,
        embeddings_skipped_already_embedded=stats.skipped,
        embeddings_failed=stats.failed,
        duration_ms=duration_ms,
    )
    _assert_columns_invariant(summary, log)
    log.info(
        "columns_embed_finish",
        run_id=request.run_id,
        embeddings_written=stats.written,
        embeddings_failed=stats.failed,
        duration_ms=duration_ms,
    )
    return summary


# ── Generic core ──


@dataclass
class _EmbedStats:
    rows_seen: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0


def _embed_files[R: BaseModel](
    *,
    files: list[Path],
    model: EmbeddingModel | None,
    embed_batch: EmbedBatchFn,
    batch_size: int,
    dry_run: bool,
    embedding_dim: int,
    row_type: type[R],
    text_for_row: Callable[[R], str | None],
    log_bind_field: str,
    log_bind_value: Callable[[R], str],
    text_chars_field: str,
    log_prefix: str,
    run_id: str,
    log: structlog.BoundLogger,
) -> _EmbedStats:
    """Generic file-by-file embed loop.

    `text_for_row` returns the text to embed, or `None` to skip the
    row entirely (failure markers, malformed). `None`-text rows are
    counted as skipped (already-embedded behaviour) so they round-
    trip through `skipped` rather than `failed` — they're not a
    pipeline-stage failure, they're rows the source field can't
    contribute to embedding.
    """
    stats = _EmbedStats()
    for path in files:
        rows: list[R] = _read_jsonl_rows(path, row_type=row_type)
        stats.rows_seen += len(rows)

        # Build the set of indices that need work. A None text_for_row
        # value means "don't embed" (failure marker), counted under
        # skipped to keep the run-invariant identity intact.
        indices_to_embed: list[int] = []
        non_embeddable = 0
        for i, r in enumerate(rows):
            embedding_attr = getattr(r, "embedding", None)
            if embedding_attr is not None:
                continue
            if text_for_row(r) is None:
                non_embeddable += 1
                continue
            indices_to_embed.append(i)

        already = len(rows) - len(indices_to_embed) - non_embeddable
        stats.skipped += already + non_embeddable

        if not indices_to_embed:
            continue

        for batch_index, batch_start in enumerate(
            range(0, len(indices_to_embed), batch_size)
        ):
            batch_idx = indices_to_embed[batch_start : batch_start + batch_size]
            batch_rows = [rows[i] for i in batch_idx]
            batch_ids = [log_bind_value(r) for r in batch_rows]
            batch_texts = [text_for_row(r) for r in batch_rows]
            # `text_for_row` returned a string for these rows by
            # construction (filtered above); type-narrow for mypy.
            assert all(t is not None for t in batch_texts)
            payload = [t for t in batch_texts if t is not None]

            if dry_run:
                for r, text in zip(batch_rows, payload, strict=True):
                    log.bind(
                        **{log_bind_field: log_bind_value(r)}
                    ).info(
                        f"would_have_embedded_{log_prefix}",
                        **{text_chars_field: len(text)},
                    )
                continue

            assert model is not None
            t0 = time.monotonic()
            try:
                vecs = embed_batch(
                    payload, model=model, batch_size=batch_size
                )
            except Exception as exc:
                log.error(
                    f"{log_prefix}_embedding_batch_failed",
                    run_id=run_id,
                    file=str(path),
                    batch_index=batch_index,
                    ids=batch_ids,
                    error=str(exc),
                )
                stats.failed += len(batch_rows)
                continue

            if len(vecs) != len(batch_rows):
                log.error(
                    f"{log_prefix}_embedding_batch_failed",
                    run_id=run_id,
                    file=str(path),
                    batch_index=batch_index,
                    ids=batch_ids,
                    error=(
                        f"length_mismatch: expected {len(batch_rows)} "
                        f"vectors, got {len(vecs)}"
                    ),
                )
                stats.failed += len(batch_rows)
                continue

            log.info(
                f"{log_prefix}_embedding_batch_done",
                run_id=run_id,
                file=str(path),
                batch_index=batch_index,
                batch_size=len(batch_rows),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

            for original_idx, row, vec in zip(
                batch_idx, batch_rows, vecs, strict=True
            ):
                reason = _validate_vector(vec, expected_dim=embedding_dim)
                if reason is not None:
                    log.bind(
                        **{log_bind_field: log_bind_value(row)}
                    ).warning(
                        f"{log_prefix}_embedding_validation_failed",
                        reason=reason,
                    )
                    stats.failed += 1
                    continue
                norm = math.sqrt(sum(c * c for c in vec))
                if abs(norm - 1.0) > 0.01:
                    log.bind(
                        **{log_bind_field: log_bind_value(row)}
                    ).warning(
                        f"{log_prefix}_embedding_norm_anomaly", norm=norm
                    )
                stats.written += 1
                rows[original_idx] = row.model_copy(
                    update={"embedding": vec}
                )

        if not dry_run:
            stage_io.atomic_rewrite_file(path=path, rows=list(rows))

    return stats


# ── Row-type-specific helpers ──


def _summary_text(row: StagedDatasetCard) -> str | None:
    """Datasets source-text accessor. Always returns the summary."""
    return row.summary


def _description_text(row: StagedColumnRow) -> str | None:
    """Columns source-text accessor.

    Returns `None` for failure-marker rows so the embed loop skips
    them and `columns-load` filters them out at coalesce time.
    """
    if row.generation_failed:
        return None
    return row.description


def _read_jsonl_rows[R: BaseModel](path: Path, *, row_type: type[R]) -> list[R]:
    """Read all rows of `row_type` from `path`, preserving order."""
    out: list[R] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(row_type.model_validate_json(line))
    return out


def _preflight_files(
    *,
    settings: Settings,
    run_id: str,
    artifact: stage_io.Artifact,
    log_name: str,
    missing_event: str,
    missing_command_hint: str,
) -> list[Path]:
    dir_ = stage_io.stage_path(
        staging_dir=settings.staging_dir, run_id=run_id, artifact=artifact
    )
    log = get_logger(log_name)
    if not dir_.is_dir():
        log.error(missing_event, run_id=run_id, path=str(dir_))
        raise RuntimeError(
            f"{missing_event}: {dir_} — run {missing_command_hint} first."
        )
    files = sorted(dir_.glob("*.jsonl"))
    if not files:
        log.error(missing_event, run_id=run_id, path=str(dir_))
        raise RuntimeError(
            f"{missing_event}: {dir_} contains no .jsonl files."
        )
    return files


def _maybe_load_model(
    *,
    load_embedding_model: LoadEmbeddingModelFn,
    settings: Settings,
    dry_run: bool,
    log: structlog.BoundLogger,
) -> EmbeddingModel | None:
    if dry_run:
        return None
    try:
        model = load_embedding_model(
            settings.embedding_model,
            device=settings.device,
            cache_dir=str(settings.hf_cache_dir)
            if settings.hf_cache_dir
            else None,
        )
        log.info(
            "embedding_model_loaded",
            repo=settings.embedding_model,
            device=settings.device,
            vram_mb=_vram_allocated_mb(),
        )
        return model
    except Exception as exc:
        log.error(
            "embedding_model_load_failed", error=str(exc), exc_info=True
        )
        raise


def _validate_vector(vec: list[float], *, expected_dim: int) -> str | None:
    """Return a reason string on failure, None on success."""
    if len(vec) != expected_dim:
        return "wrong_dim"
    for x in vec:
        if math.isnan(x):
            return "has_nan"
        if math.isinf(x):
            return "has_inf"
    return None


def _vram_allocated_mb() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _drop_cuda_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_datasets_invariant(
    summary: DatasetsEmbedRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.embeddings_written
        + summary.embeddings_skipped_already_embedded
        + summary.embeddings_failed
    )
    if total != summary.rows_seen:
        log.error(
            "run_invariant_violated",
            subcommand="datasets-embed",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"datasets-embed rows accounted-for mismatch: {summary}"
        )


def _assert_columns_invariant(
    summary: ColumnsEmbedRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.embeddings_written
        + summary.embeddings_skipped_already_embedded
        + summary.embeddings_failed
    )
    if total != summary.rows_seen:
        log.error(
            "run_invariant_violated",
            subcommand="columns-embed",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"columns-embed rows accounted-for mismatch: {summary}"
        )
