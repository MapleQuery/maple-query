"""`datasets-embed` orchestrator.

GPU-box-side. Reads each `stage/<run_id>/datasets/*.jsonl` in path-
ascending order, batch-embeds the `summary` field of rows that don't
already carry an embedding, validates each vector, and atomically
rewrites the file with `embedding` populated.

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

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import embed as embed_mod
from semantic_enrich.core import stage_io
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    DatasetsEmbedRunSummary,
    EmbeddingModel,
    StagedDatasetCard,
)

EmbedBatchFn = Callable[..., list[list[float]]]
LoadEmbeddingModelFn = Callable[..., EmbeddingModel]


@dataclass(frozen=True)
class EmbedRequest:
    run_id: str
    dry_run: bool
    batch_size: int | None  # None → settings.embedding_batch_size


def preflight(*, settings: Settings, request: EmbedRequest) -> list[Path]:
    """Returns the sorted list of `stage/<run_id>/datasets/*.jsonl`
    paths. Raises if the dir is missing or empty."""
    datasets_dir = stage_io.stage_path(
        staging_dir=settings.staging_dir,
        run_id=request.run_id,
        artifact="datasets",
    )
    log = get_logger("semantic_enrich.embedding_pass")
    if not datasets_dir.is_dir():
        log.error("datasets_dir_missing", run_id=request.run_id,
                  path=str(datasets_dir))
        raise RuntimeError(
            f"datasets_dir_missing: {datasets_dir} — run datasets-generate "
            "first."
        )
    files = sorted(datasets_dir.glob("*.jsonl"))
    if not files:
        log.error("datasets_dir_missing", run_id=request.run_id,
                  path=str(datasets_dir))
        raise RuntimeError(
            f"datasets_dir_missing: {datasets_dir} contains no .jsonl files."
        )
    return files


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

    model: EmbeddingModel | None = None
    if not request.dry_run:
        try:
            model = load_embedding_model(
                settings.embedding_model,
                device=settings.device,
                cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
            )
            log.info(
                "embedding_model_loaded",
                repo=settings.embedding_model,
                device=settings.device,
                vram_mb=_vram_allocated_mb(),
            )
        except Exception as exc:
            log.error("embedding_model_load_failed", error=str(exc), exc_info=True)
            raise

    rows_seen = 0
    written = 0
    skipped = 0
    failed = 0

    try:
        for path in files:
            rows = _read_jsonl_rows(path)
            rows_seen += len(rows)

            # Resume: only rows missing an embedding need work.
            indices_to_embed = [
                i for i, r in enumerate(rows) if r.embedding is None
            ]
            already = len(rows) - len(indices_to_embed)
            skipped += already

            if not indices_to_embed:
                continue

            for batch_index, batch_start in enumerate(
                range(0, len(indices_to_embed), batch_size)
            ):
                batch_idx = indices_to_embed[batch_start : batch_start + batch_size]
                batch_rows = [rows[i] for i in batch_idx]
                batch_ids = [r.package_id for r in batch_rows]

                if request.dry_run:
                    for r in batch_rows:
                        log.bind(package_id=r.package_id).info(
                            "would_have_embedded",
                            summary_chars=len(r.summary),
                        )
                    continue

                assert model is not None
                t0 = time.monotonic()
                try:
                    vecs = embed_batch(
                        [r.summary for r in batch_rows],
                        model=model,
                        batch_size=batch_size,
                    )
                except Exception as exc:
                    log.error(
                        "embedding_batch_failed",
                        run_id=request.run_id,
                        file=str(path),
                        batch_index=batch_index,
                        package_ids=batch_ids,
                        error=str(exc),
                    )
                    failed += len(batch_rows)
                    continue

                if len(vecs) != len(batch_rows):
                    log.error(
                        "embedding_batch_failed",
                        run_id=request.run_id,
                        file=str(path),
                        batch_index=batch_index,
                        package_ids=batch_ids,
                        error=(
                            f"length_mismatch: expected {len(batch_rows)} "
                            f"vectors, got {len(vecs)}"
                        ),
                    )
                    failed += len(batch_rows)
                    continue

                log.info(
                    "embedding_batch_done",
                    run_id=request.run_id,
                    file=str(path),
                    batch_index=batch_index,
                    batch_size=len(batch_rows),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )

                for original_idx, row, vec in zip(
                    batch_idx, batch_rows, vecs, strict=True
                ):
                    reason = _validate_vector(vec, expected_dim=settings.embedding_dim)
                    if reason is not None:
                        log.bind(package_id=row.package_id).warning(
                            "embedding_validation_failed", reason=reason
                        )
                        failed += 1
                        continue
                    # Norm check (warning-only — load anyway).
                    norm = math.sqrt(sum(c * c for c in vec))
                    if abs(norm - 1.0) > 0.01:
                        log.bind(package_id=row.package_id).warning(
                            "embedding_norm_anomaly", norm=norm
                        )
                    written += 1
                    rows[original_idx] = row.model_copy(update={"embedding": vec})

            if not request.dry_run:
                stage_io.atomic_rewrite_file(path=path, rows=list(rows))
    finally:
        if model is not None:
            del model
            _drop_cuda_cache()

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = DatasetsEmbedRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        staged_files_seen=len(files),
        rows_seen=rows_seen,
        embeddings_written=written,
        embeddings_skipped_already_embedded=skipped,
        embeddings_failed=failed,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "datasets_embed_finish",
        run_id=request.run_id,
        embeddings_written=written,
        embeddings_failed=failed,
        duration_ms=duration_ms,
    )
    return summary


def _read_jsonl_rows(path: Path) -> list[StagedDatasetCard]:
    """Read all StagedDatasetCard rows from `path`, preserving order."""
    out: list[StagedDatasetCard] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(StagedDatasetCard.model_validate_json(line))
    return out


def _validate_vector(
    vec: list[float], *, expected_dim: int
) -> str | None:
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


def _assert_invariant(
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
