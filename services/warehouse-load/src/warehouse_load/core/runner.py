"""End-to-end orchestration for the documents loader.

`run_documents_load` is the function the CLI calls. It wires the
runlog reader, filter, dedupe, bucket-existence intersection, and
MERGE together, emits structured log events for each stage, and
returns a `DocumentsRunSummary`.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from warehouse_load.clients.bq import BqClient
from warehouse_load.clients.gcs import GcsClient
from warehouse_load.core.documents_filter import (
    FilteredRow,
    dedupe_by_source_url,
    filter_rows,
    intersect_bucket,
)
from warehouse_load.core.documents_merge import dry_run, merge_documents
from warehouse_load.core.runlog_reader import iter_runlog_rows
from warehouse_load.core.schema_loader import load_schema
from warehouse_load.providers.logging import get_logger
from warehouse_load.types import DocumentsRunSummary, RawRunlogRow

# Refuse to MERGE when both: at least this many rows would be dropped
# as `blob_missing`, AND they make up at least this fraction of the
# post-dedupe set. Catches a misconfigured `WHLOAD_BUCKET_PREFIX` or
# a URI-format drift between ingest writes and the bucket listing,
# either of which would otherwise silently classify the whole corpus
# as zombies and ship zero rows. Small runs (<100 missing) never
# trip — the absolute floor protects unit-sized integration runs.
_MASS_BLOB_MISSING_MIN_COUNT = 100
_MASS_BLOB_MISSING_FRACTION = 0.5
_MASS_BLOB_MISSING_SAMPLE_SIZE = 3


@dataclass(frozen=True)
class RunRequest:
    """Per-invocation intent. Built by the CLI from flags + env."""

    local_dir: Path | None
    gcs_prefix: str | None
    since: datetime | None
    dry_run: bool
    limit_orgs: tuple[str, ...]
    bucket_prefix: str | None
    no_bucket_check: bool
    # Set to True only for the legitimate "we just cleaned the bucket"
    # workflow. Disables the mass-blob-missing guardrail.
    allow_mass_blob_missing: bool = False


def run_documents_load(
    *,
    request: RunRequest,
    bq: BqClient | None,
    gcs: GcsClient | None,
    project_id: str,
    dataset: str,
    table: str,
    schemas_dir: Path,
    run_id: str,
) -> DocumentsRunSummary:
    """Run one pass. `bq` may be None when `request.dry_run` is True."""
    log = get_logger("warehouse_load.runner")
    started = time.monotonic()

    log.info(
        "documents_load_start",
        run_id=run_id,
        dry_run=request.dry_run,
        runlog_local_dir=str(request.local_dir) if request.local_dir else None,
        runlog_gcs_prefix=request.gcs_prefix,
        since=request.since.isoformat() if request.since else None,
    )

    files_seen: set[str] = set()
    rows_seen = 0
    parse_errors = 0
    kept_after_filter: list[RawRunlogRow] = []
    filtered_not_csv = 0
    filtered_not_success = 0

    events = iter_runlog_rows(
        local_dir=request.local_dir,
        gcs_prefix=request.gcs_prefix,
        gcs_client=gcs,
        since=request.since,
    )

    for event in events:
        files_seen.add(event.source)
        if event.error is not None:
            parse_errors += 1
            log.warning(
                "runlog_parse_error",
                path=event.error.source,
                line_number=event.error.line_number,
                error=event.error.error,
            )
            continue
        assert event.row is not None
        rows_seen += 1
        if request.limit_orgs and event.row.organization_code not in request.limit_orgs:
            continue
        kept_after_filter.append(event.row)

    # `kept_after_filter` is the post-`--limit-orgs` set; filter+dedupe
    # operate on it below.
    filtered: list[RawRunlogRow] = []
    for filter_event in filter_rows(kept_after_filter):
        if isinstance(filter_event, FilteredRow):
            log.info(
                "row_filtered",
                source_url=filter_event.row.source_url,
                reason=filter_event.reason,
            )
            if filter_event.reason == "not_csv":
                filtered_not_csv += 1
            else:
                filtered_not_success += 1
        else:
            filtered.append(filter_event)

    deduped, dropped = dedupe_by_source_url(filtered)
    for drop in dropped:
        log.info(
            "row_deduped",
            source_url=drop.dropped.source_url,
            dropped_document_id=drop.dropped.document_id,
            kept_document_id=drop.kept.document_id,
            reason=drop.reason,
        )

    after_intersection, blob_missing, bucket_check_skipped = _intersect_against_bucket(
        rows=deduped,
        request=request,
        gcs=gcs,
        log=log,
        run_id=run_id,
    )
    filtered_blob_missing = len(blob_missing)

    _guard_mass_blob_missing(
        blob_missing=blob_missing,
        deduped_count=len(deduped),
        gcs=gcs,
        allow=request.allow_mass_blob_missing,
        bucket_check_skipped=bucket_check_skipped,
        log=log,
        run_id=run_id,
    )

    rows_kept = len(after_intersection)

    documents_inserted = 0
    documents_updated = 0
    documents_unchanged = 0

    schema = load_schema(schemas_dir / "raw_documents.json")

    if request.dry_run:
        result = dry_run(after_intersection)
        for payload_row in result.payload:
            log.info(
                "would_have_merged",
                document_id=payload_row["document_id"],
                source_url=payload_row["source_url"],
                action="insert_or_update",
            )
    else:
        if bq is None:
            raise ValueError("bq client required when dry_run is False")
        merge_result = merge_documents(
            bq=bq,
            rows=after_intersection,
            project_id=project_id,
            dataset=dataset,
            table=table,
            schema=schema,
            run_id_short=run_id[:8],
        )
        documents_inserted = merge_result.rows_inserted
        documents_updated = merge_result.rows_updated
        documents_unchanged = merge_result.rows_unchanged
        log.info(
            "documents_merge_done",
            run_id=run_id,
            rows_inserted=documents_inserted,
            rows_updated=documents_updated,
            rows_unchanged=documents_unchanged,
        )

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = DocumentsRunSummary(
        run_id=run_id,
        dry_run=request.dry_run,
        runlog_files_seen=len(files_seen),
        runlog_rows_seen=rows_seen,
        runlog_parse_errors=parse_errors,
        rows_filtered_not_csv=filtered_not_csv,
        rows_filtered_not_success=filtered_not_success,
        rows_filtered_blob_missing=filtered_blob_missing,
        rows_deduped=len(dropped),
        rows_kept=rows_kept,
        documents_inserted=documents_inserted,
        documents_updated=documents_updated,
        documents_unchanged=documents_unchanged,
        bucket_check_skipped=bucket_check_skipped,
        duration_ms=duration_ms,
    )
    log.info("documents_load_finish", run_id=run_id, summary=_summary_dict(summary))
    return summary


def _intersect_against_bucket(
    *,
    rows: list[RawRunlogRow],
    request: RunRequest,
    gcs: GcsClient | None,
    log: Any,
    run_id: str,
) -> tuple[list[RawRunlogRow], list[FilteredRow], bool]:
    """Apply the bucket-truth intersection. Returns (kept, dropped, skipped).

    Skipping happens when the operator passes `--no-bucket-check`, or
    when no bucket is configured in a dry-run (keeps the existing
    no-bucket dry-run path working for local development).

    Real runs without a reachable bucket fail loudly: the alternative —
    silently skipping — would silently pollute the warehouse on the
    next reachable run.
    """
    if request.no_bucket_check:
        log.warning("bucket_check_disabled", run_id=run_id, reason="no_bucket_check_flag")
        return rows, [], True

    if request.bucket_prefix is None or gcs is None:
        if request.dry_run:
            log.warning(
                "bucket_check_disabled",
                run_id=run_id,
                reason="no_bucket_configured_in_dry_run",
            )
            return rows, [], True
        raise ValueError(
            "bucket-intersection requires a configured bucket_prefix and gcs client; "
            "set WHLOAD_BUCKET_PREFIX or pass --no-bucket-check to opt out.",
        )

    log.info("bucket_check_started", run_id=run_id, prefix=request.bucket_prefix)
    started = time.monotonic()
    existing = frozenset(gcs.list_existing(request.bucket_prefix))
    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "bucket_check_done",
        run_id=run_id,
        prefix=request.bucket_prefix,
        existing_count=len(existing),
        elapsed_ms=elapsed_ms,
    )

    kept, dropped = intersect_bucket(rows, existing)
    for filtered_row in dropped:
        log.info(
            "row_filtered",
            run_id=run_id,
            reason=filtered_row.reason,
            source_url=filtered_row.row.source_url,
            document_id=filtered_row.row.document_id,
            gcs_uri=filtered_row.row.gcs_uri,
        )
    return kept, dropped, False


def _guard_mass_blob_missing(
    *,
    blob_missing: list[FilteredRow],
    deduped_count: int,
    gcs: GcsClient | None,
    allow: bool,
    bucket_check_skipped: bool,
    log: Any,
    run_id: str,
) -> None:
    """Refuse to MERGE when the bucket says most rows are zombies.

    Trips on `missing/deduped >= 0.5` AND `missing >= 100`. The absolute
    floor keeps unit-sized runs from refusing on a single drop; the
    fraction floor keeps a sparse miss-rate from refusing on
    legitimate corpus growth.

    Before refusing, sample a few of the "missing" URIs and call
    `blob_exists` on them. If any actually exist, that's the smoking
    gun for a URI-format drift (not a real bucket clean), and the
    error message names that case explicitly.
    """
    if bucket_check_skipped or allow:
        return
    missing_count = len(blob_missing)
    if missing_count < _MASS_BLOB_MISSING_MIN_COUNT:
        return
    fraction = missing_count / max(1, deduped_count)
    if fraction < _MASS_BLOB_MISSING_FRACTION:
        return

    if gcs is not None:
        # Seed on run_id so the sampled URIs in the log line are stable
        # across re-runs of the same run_id — easier to triage from logs.
        sample = random.Random(run_id).sample(
            blob_missing,
            k=min(_MASS_BLOB_MISSING_SAMPLE_SIZE, missing_count),
        )
        smoking_guns = [
            fr.row.gcs_uri
            for fr in sample
            if fr.row.gcs_uri is not None and gcs.blob_exists(fr.row.gcs_uri)
        ]
        if smoking_guns:
            log.error(
                "mass_blob_missing_format_drift",
                run_id=run_id,
                rows_missing=missing_count,
                rows_deduped=deduped_count,
                sampled_existing=smoking_guns,
            )
            raise RuntimeError(
                f"sampled 'missing' blobs actually exist on the bucket: "
                f"{smoking_guns}. URI format may have drifted between "
                "ingest writes and the bucket listing — investigate "
                "before re-running.",
            )

    log.error(
        "mass_blob_missing_refused",
        run_id=run_id,
        rows_missing=missing_count,
        rows_deduped=deduped_count,
        fraction=fraction,
    )
    raise RuntimeError(
        f"{missing_count}/{deduped_count} rows ({fraction:.0%}) would be "
        "dropped as blob_missing — refusing to MERGE. Pass "
        "--allow-mass-blob-missing if this is intentional (e.g. you just "
        "cleaned the bucket).",
    )


def _summary_dict(summary: DocumentsRunSummary) -> dict[str, Any]:
    """`asdict` would do the same; spelled out for stability under
    dataclass refactors."""
    return {
        "run_id": summary.run_id,
        "dry_run": summary.dry_run,
        "runlog_files_seen": summary.runlog_files_seen,
        "runlog_rows_seen": summary.runlog_rows_seen,
        "runlog_parse_errors": summary.runlog_parse_errors,
        "rows_filtered_not_csv": summary.rows_filtered_not_csv,
        "rows_filtered_not_success": summary.rows_filtered_not_success,
        "rows_filtered_blob_missing": summary.rows_filtered_blob_missing,
        "rows_deduped": summary.rows_deduped,
        "rows_kept": summary.rows_kept,
        "documents_inserted": summary.documents_inserted,
        "documents_updated": summary.documents_updated,
        "documents_unchanged": summary.documents_unchanged,
        "bucket_check_skipped": summary.bucket_check_skipped,
        "duration_ms": summary.duration_ms,
    }
