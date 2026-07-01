"""`datasets-extract` orchestrator.

Laptop-side. Reads `raw.documents` + `raw.rows`, materialises one
`PackageInputs` per package into `stage/<run_id>/inputs/*.jsonl`. The
GPU box reads that JSONL after an rsync; it never touches BQ itself.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from google.api_core import exceptions as gax

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import package_grouper, sample_selector, stage_io
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    Counters,
    DatasetsExtractRunSummary,
    PackageInputs,
)


@dataclass(frozen=True)
class _ExtractOutcome:
    """Per-package result returned from a worker thread back to the
    main loop, which is the only thread that touches the StageWriter
    and the run-level counters."""

    kind: Literal["extracted", "failed"]
    package_id: str
    pkg: PackageInputs | None = None
    failure_reason: str | None = None
    failure_log_kwargs: dict[str, object] | None = None


@dataclass(frozen=True)
class ExtractRequest:
    run_id: str
    dry_run: bool
    limit_packages: int | None
    limit_package_ids: list[str] | None
    limit_orgs: list[str] | None


def preflight(*, settings: Settings, bq: BqClient) -> None:
    """Fail fast before the candidate query."""
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "datasets-extract; this subcommand reads from BigQuery."
        )
    log = get_logger("semantic_enrich.dataset_extract")
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc
    if not settings.staging_dir.parent.exists():
        raise RuntimeError(
            f"staging dir parent does not exist: {settings.staging_dir.parent}"
        )


def run_extract(
    *,
    request: ExtractRequest,
    settings: Settings,
    bq: BqClient,
    logger: structlog.BoundLogger | None = None,
) -> DatasetsExtractRunSummary:
    """End-to-end candidate query → per-package secondary queries →
    `stage/<run_id>/inputs/*.jsonl`."""
    log = logger or get_logger("semantic_enrich.dataset_extract")
    started = time.monotonic()

    log.info(
        "datasets_extract_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        limit_packages=request.limit_packages,
        limit_package_ids=request.limit_package_ids,
        limit_orgs=request.limit_orgs,
        staging_dir=str(settings.staging_dir),
    )

    already_extracted = stage_io.read_staged_package_ids(
        run_id=request.run_id,
        artifact="inputs",
        staging_dir=settings.staging_dir,
    )

    project_id = settings.gcp_project_id
    assert project_id is not None

    candidate_sql = package_grouper.build_candidate_sql(
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        documents_table=settings.bq_documents_table,
        with_limit=request.limit_packages is not None,
    )
    candidate_params = package_grouper.build_candidate_params(
        limit_orgs=request.limit_orgs,
        limit_package_ids=request.limit_package_ids,
        already_extracted=already_extracted,
        limit_packages=request.limit_packages,
    )

    cq_started = time.monotonic()
    candidate_rows = list(bq.query_rows(candidate_sql, params=candidate_params))
    cq_ms = int((time.monotonic() - cq_started) * 1000)
    log.info(
        "candidate_query_done",
        run_id=request.run_id,
        candidate_count=len(candidate_rows),
        duration_ms=cq_ms,
    )

    counters = Counters()
    counters.extras["skipped_already_extracted"] = len(already_extracted)

    def _on_flush(path: Path, seq: int, row_count: int) -> None:
        log.info(
            "inputs_flush_written",
            run_id=request.run_id,
            path=str(path),
            flush_seq=seq,
            row_count=row_count,
        )

    writer = stage_io.StageWriter(
        run_id=request.run_id,
        artifact="inputs",
        staging_dir=settings.staging_dir,
        flush_every=settings.flush_every_n_packages,
        on_flush=_on_flush,
    )

    workers = max(1, settings.extract_concurrency)
    log.info(
        "extract_pool_started",
        run_id=request.run_id,
        workers=workers,
    )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _extract_one_package,
                    raw_row=raw_row,
                    bq=bq,
                    project_id=project_id,
                    settings=settings,
                    log=log,
                )
                for raw_row in candidate_rows
            ]
            # Drain in submission order so the on-disk flush sequence
            # is deterministic across runs against the same candidate
            # set — the stage file containing package N is the same
            # across re-runs.
            for raw_row, future in zip(candidate_rows, futures, strict=True):
                try:
                    outcome = future.result()
                except Exception as exc:
                    pid = raw_row.get("package_id", "<unknown>") if isinstance(raw_row, dict) else "<unknown>"
                    log.bind(package_id=pid).exception(
                        "package_extract_failed", error=str(exc)
                    )
                    counters.failed += 1
                    continue

                if outcome.kind == "failed":
                    counters.failed += 1
                    continue

                assert outcome.pkg is not None
                writer.append(outcome.pkg)
                counters.generated += 1
                if request.dry_run:
                    log.bind(package_id=outcome.package_id).info(
                        "would_have_extracted",
                        resource_count=len(outcome.pkg.resources),
                        column_count=len(outcome.pkg.column_names),
                    )
    finally:
        writer.close()

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = DatasetsExtractRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        candidate_count=len(candidate_rows) + counters.extras["skipped_already_extracted"],
        packages_extracted=counters.generated,
        packages_skipped_already_extracted=counters.extras["skipped_already_extracted"],
        packages_failed=counters.failed,
        flush_files_written=writer.files_written,
        duration_ms=duration_ms,
    )

    _assert_invariant(summary, log)
    log.info("datasets_extract_finish", run_id=request.run_id, duration_ms=duration_ms,
             summary=summary.__dict__)
    return summary


def _extract_one_package(
    *,
    raw_row: dict[str, object],
    bq: BqClient,
    project_id: str,
    settings: Settings,
    log: structlog.BoundLogger,
) -> _ExtractOutcome:
    """Per-package work: decode candidate row, run column-union +
    sample-rows queries, assemble `PackageInputs`. Safe to call from
    a worker thread; the BQ client is thread-safe and structlog's
    bound loggers are immutable per-call.

    Returns an outcome instead of raising on data-shape failures
    (`package_id_missing`, `representative_doc_has_no_rows`) so the
    main thread can update counters without losing other packages
    in the same batch. Unexpected exceptions still propagate.
    """
    package_id, resources = package_grouper.decode_candidate_row(raw_row)
    plog = log.bind(package_id=package_id)

    if not resources:
        plog.error("package_id_missing", document_id=None)
        return _ExtractOutcome(
            kind="failed", package_id=package_id,
            failure_reason="no_resources",
        )

    rep = sample_selector.pick_representative(resources)
    if rep.row_count is None or rep.row_count == 0:
        plog.error(
            "representative_doc_has_no_rows",
            representative_document_id=rep.document_id,
        )
        return _ExtractOutcome(
            kind="failed", package_id=package_id,
            failure_reason="no_rows",
        )

    all_doc_ids = [r.document_id for r in resources]
    col_union = package_grouper.fetch_column_union(
        bq=bq,
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
        document_ids=all_doc_ids,
    )
    kept_cols, truncated_to = package_grouper.truncate_columns(
        names=col_union, cap=settings.sample_column_cap
    )
    if truncated_to is not None:
        plog.info("column_names_truncated_to",
                  kept=len(kept_cols), total=truncated_to)

    indices = sample_selector.derive_indices(
        document_id=rep.document_id,
        row_count=rep.row_count,
        k=settings.sample_rows_per_package,
    )
    sample_rows: list[dict[str, str | None]] = []
    for decoded in package_grouper.fetch_sample_rows(
        bq=bq,
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
        document_id=rep.document_id,
        indices=indices,
    ):
        sample_rows.append(
            {k: sample_selector.truncate_cell(v) for k, v in decoded.items()}
        )

    pkg = PackageInputs(
        package_id=package_id,
        resources=resources,
        column_names=kept_cols,
        column_names_truncated_to=truncated_to,
        representative_document_id=rep.document_id,
        sample_rows=tuple(sample_rows),
    )
    plog.info(
        "package_inputs_extracted",
        resource_count=len(resources),
        column_count=len(kept_cols),
        representative_document_id=rep.document_id,
        sample_row_count=len(sample_rows),
    )
    return _ExtractOutcome(kind="extracted", package_id=package_id, pkg=pkg)


def _assert_invariant(
    summary: DatasetsExtractRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.packages_extracted
        + summary.packages_skipped_already_extracted
        + summary.packages_failed
    )
    if total != summary.candidate_count:
        log.error("run_invariant_violated", subcommand="datasets-extract",
                  summary=summary.__dict__)
        raise RuntimeError(
            f"datasets-extract candidates accounted-for mismatch: {summary}"
        )
