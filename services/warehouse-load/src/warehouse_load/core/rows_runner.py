"""End-to-end orchestration for the rows loader.

`run_rows_load` is the function the CLI calls. It wires the
candidate query, the per-doc worker pool, the batch flush, the
column-index refresh, and the post-run invariant check.

Concurrency model: `ThreadPoolExecutor(rows_concurrency)` runs the
per-doc pipeline (mark in-flight → download → sniff → detect →
stream → append to staging). After each parallel batch completes,
the orchestrator decides whether to flush staging into `raw.rows`
(time-or-size cutoff per PRD §8.3); after the final batch, flushes
unconditionally; then refreshes `raw.column_index` (unless
disabled).
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import dataclasses
import json
import re
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from google.cloud.bigquery import (
    ArrayQueryParameter,
    ScalarQueryParameter,
)

from warehouse_load.clients.bq import BqClient
from warehouse_load.clients.gcs_stream import (
    BlobMissingError,
    BytesCapExceededError,
    GcsStreamClient,
)
from warehouse_load.config.settings import Settings
from warehouse_load.core import column_index as column_index_mod
from warehouse_load.core import document_status, rows_merge
from warehouse_load.core.csv_sniff import sniff_csv
from warehouse_load.core.header_detect import detect_header
from warehouse_load.core.row_stream import (
    iter_lookahead_rows,
    needs_utf8_conversion,
    prepare_utf8_copy,
    stream_body_rows,
)
from warehouse_load.core.schema_loader import load_schema
from warehouse_load.providers.logging import get_logger
from warehouse_load.types import (
    DocumentRow,
    RowsRunRequest,
    RowsRunSummary,
    SniffResult,
)

# Validate identifiers we interpolate into the candidate query. The
# values come from Settings; defense in depth keeps a future code
# path that pulls them from less-trusted config from injecting SQL.
_BQ_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class DocumentLoadResult:
    """One per-doc outcome. Always set; never None on the disposition
    axis — the post-run invariant check (§8.7) depends on it.
    """

    document_id: str
    organization_code: str
    final_status: document_status.LoadStatus
    load_error: str | None
    preamble_rows: tuple[tuple[str, ...], ...] | None
    header_confidence: str | None
    row_count: int | None
    bytes_read: int
    duration_ms: int
    staging_jsonl_path: Path | None
    # True iff the doc was already `loaded` and we skipped it (no
    # --force). Recorded so the summary can split docs_loaded from
    # docs_skipped_already_loaded.
    skipped_already_loaded: bool = False


def run_rows_load(
    *,
    request: RowsRunRequest,
    bq: BqClient | None,
    gcs: GcsStreamClient | None,
    settings: Settings,
    run_id: str,
) -> RowsRunSummary:
    """Run one pass of the rows loader."""
    log = get_logger("warehouse_load.rows_runner")
    started = time.monotonic()

    project = settings.gcp_project_id
    dataset = settings.bq_dataset_raw
    documents_table = f"{project}.{dataset}.{settings.bq_documents_table}"
    rows_table = f"{project}.{dataset}.{settings.bq_rows_table}"
    staging_table = f"{project}.{dataset}.{settings.bq_rows_staging_table}"
    column_index_table = f"{project}.{dataset}.{settings.bq_column_index_table}"

    for ident in (
        settings.bq_dataset_raw,
        settings.bq_documents_table,
        settings.bq_rows_table,
        settings.bq_rows_staging_table,
        settings.bq_column_index_table,
    ):
        if not _BQ_IDENT_RE.fullmatch(ident):
            raise ValueError(f"invalid BQ identifier in settings: {ident!r}")

    if not request.dry_run:
        if bq is None:
            raise ValueError("bq client required when dry_run is False")
        _assert_staging_empty_or_die(
            bq=bq,
            staging_table=staging_table,
            log=log,
            project=project,
        )

    candidates = _load_candidates(
        bq=bq,
        documents_table=documents_table,
        request=request,
        log=log,
        dry_run=request.dry_run,
    )

    log.info(
        "rows_load_start",
        run_id=run_id,
        dry_run=request.dry_run,
        concurrency=request.concurrency or settings.rows_concurrency,
        candidate_count=len(candidates),
    )

    rows_schema = load_schema(settings.schemas_dir / "raw_rows.json")

    concurrency = request.concurrency or settings.rows_concurrency
    flush_rows_threshold = settings.rows_staging_flush_threshold
    flush_bytes_threshold = settings.rows_staging_flush_bytes_threshold
    flush_files_threshold = settings.rows_staging_flush_files_threshold

    pending_to_record: list[DocumentLoadResult] = []
    staging_rows_pending = 0
    staging_bytes_pending = 0
    staging_files_pending = 0
    rows_merged = 0
    accumulated_results: list[DocumentLoadResult] = []
    temp_files: list[Path] = []

    # Skip-loaded gate: if the operator selected an already-loaded
    # status without passing --force, every candidate is a skip.
    # This is the only path that produces skipped_already_loaded
    # outcomes — the §8.7 invariant relies on it being explicit.
    if request.status == "loaded" and not request.force:
        log.warning(
            "rows_skipped_without_force",
            run_id=run_id,
            count=len(candidates),
            hint="pass --force to replay already-loaded docs",
        )
        for doc in candidates:
            accumulated_results.append(_make_skipped_result(doc))
        return _finish(
            summary=_build_summary(
                run_id=run_id, request=request,
                candidate_count=len(candidates),
                results=accumulated_results, rows_merged=0,
                column_index_refreshed=False,
                duration_ms=int((time.monotonic() - started) * 1000),
            ),
            log=log, run_id=run_id,
        )

    try:
        for batch in _chunks(candidates, concurrency):
            results = _run_doc_batch(
                docs=batch,
                bq=bq,
                gcs=gcs,
                settings=settings,
                staging_table=staging_table,
                rows_schema=rows_schema,
                request=request,
                log=log,
            )
            for r in results:
                accumulated_results.append(r)
                if r.staging_jsonl_path is not None:
                    temp_files.append(r.staging_jsonl_path)
                if r.final_status == "loaded":
                    pending_to_record.append(r)
                    staging_rows_pending += r.row_count or 0
                    if r.staging_jsonl_path is not None:
                        staging_files_pending += 1
                        # `stat()` can race with cleanup on bizarre FS
                        # errors; treat missing as 0 bytes rather than
                        # crashing the run.
                        with contextlib.suppress(OSError):
                            staging_bytes_pending += (
                                r.staging_jsonl_path.stat().st_size
                            )
                else:
                    # Failures and skipped-loaded are already recorded
                    # in the per-doc worker. Nothing to defer.
                    pass

            if not request.dry_run and pending_to_record and (
                staging_rows_pending >= flush_rows_threshold
                or staging_bytes_pending >= flush_bytes_threshold
                or staging_files_pending >= flush_files_threshold
            ):
                rows_merged += _flush_and_record(
                    bq=bq,
                    log=log,
                    run_id=run_id,
                    staging_table=staging_table,
                    rows_table=rows_table,
                    documents_table=documents_table,
                    pending=pending_to_record,
                    staging_rows=staging_rows_pending,
                    rows_schema=rows_schema,
                )
                pending_to_record = []
                staging_rows_pending = 0
                staging_bytes_pending = 0
                staging_files_pending = 0

        # Final flush — anything left over after the last batch.
        if not request.dry_run and pending_to_record:
            rows_merged += _flush_and_record(
                bq=bq,
                log=log,
                run_id=run_id,
                staging_table=staging_table,
                rows_table=rows_table,
                documents_table=documents_table,
                pending=pending_to_record,
                staging_rows=staging_rows_pending,
                rows_schema=rows_schema,
            )
            pending_to_record = []
            staging_rows_pending = 0
            staging_bytes_pending = 0
            staging_files_pending = 0
    finally:
        for tf in temp_files:
            tf.unlink(missing_ok=True)

    column_index_refreshed = False
    if request.refresh_column_index and not request.dry_run:
        if bq is None:
            raise ValueError("bq client required to refresh column_index")
        column_index_mod.refresh_column_index(
            bq=bq,
            rows_table=rows_table,
            column_index_table=column_index_table,
            doc_ids_cap=settings.column_index_doc_ids_cap,
            log=log,
            run_id=run_id,
        )
        column_index_refreshed = True

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = _build_summary(
        run_id=run_id,
        request=request,
        candidate_count=len(candidates),
        results=accumulated_results,
        rows_merged=rows_merged,
        column_index_refreshed=column_index_refreshed,
        duration_ms=duration_ms,
    )
    return _finish(summary=summary, log=log, run_id=run_id)


# --------------------------------------------------------------------
# candidate selection
# --------------------------------------------------------------------


def _load_candidates(
    *,
    bq: BqClient | None,
    documents_table: str,
    request: RowsRunRequest,
    log: Any,
    dry_run: bool,
) -> list[DocumentRow]:
    """Run the candidate query against `raw.documents`. Returns rows
    projected to `DocumentRow`.

    `dry_run` without a bq client returns an empty list — useful for
    local plumbing tests where the operator wants to exercise the
    code path without standing up a project. The CLI requires a
    project for real runs.
    """
    if bq is None:
        if dry_run:
            log.warning("candidate_query_skipped", reason="dry_run_without_bq")
            return []
        raise ValueError("bq client required to load candidates")

    sql = f"""\
SELECT
  document_id,
  organization_code,
  source_url,
  gcs_uri,
  file_format,
  declared_format,
  checksum,
  resource_last_modified
FROM `{documents_table}`
WHERE file_format IN ('csv', 'tsv')
  AND ingestion_status = 'success'
  AND load_status = @status
  AND (
    @limit_orgs_empty
    OR organization_code IN UNNEST(@limit_orgs)
  )
  AND (
    @limit_documents_empty
    OR document_id IN UNNEST(@limit_documents)
  )
ORDER BY resource_last_modified DESC NULLS LAST
"""
    if request.force:
        # --force replays already-loaded docs. The candidate query
        # still uses @status as its filter; CLI defaults flip from
        # `pending` to whatever the operator wants to replay
        # (typically `loaded`). We log this so it's auditable.
        log.info("rows_force_replay_active", status=request.status)

    params: list[ScalarQueryParameter | ArrayQueryParameter] = [
        ScalarQueryParameter("status", "STRING", request.status),
        ScalarQueryParameter("limit_orgs_empty", "BOOL", not bool(request.limit_orgs)),
        ArrayQueryParameter("limit_orgs", "STRING", list(request.limit_orgs)),
        ScalarQueryParameter(
            "limit_documents_empty", "BOOL", not bool(request.limit_documents),
        ),
        ArrayQueryParameter("limit_documents", "STRING", list(request.limit_documents)),
    ]

    candidates: list[DocumentRow] = []
    for row in bq.query_rows(sql, params=params):
        candidates.append(
            DocumentRow(
                document_id=row["document_id"],
                organization_code=row["organization_code"],
                source_url=row["source_url"],
                gcs_uri=row.get("gcs_uri"),
                file_format=row["file_format"],
                declared_format=row.get("declared_format"),
                checksum=row.get("checksum"),
                resource_last_modified=row.get("resource_last_modified"),
            ),
        )
    return candidates


# --------------------------------------------------------------------
# pre-run + post-run invariants
# --------------------------------------------------------------------


def _assert_staging_empty_or_die(
    *,
    bq: BqClient,
    staging_table: str,
    log: Any,
    project: str,
) -> None:
    """PRD §8.0: refuse to start if `raw.rows_staging` is non-empty.

    Exit code 2 distinguishes this precondition failure from a generic
    error (1), so wrapper scripts can branch on it.
    """
    count = rows_merge.assert_staging_empty(bq=bq, staging_table=staging_table)
    if count == 0:
        return
    log.error(
        "staging_precondition_violated",
        staging_row_count=count,
        aborting=True,
        likely_cause=(
            "another runner is in flight, or a prior run crashed mid-batch"
        ),
        recovery_if_no_other_runner=(
            f"bq query --use_legacy_sql=false "
            f"'TRUNCATE TABLE `{staging_table}`'"
        ),
        check_for_other_runners=(
            "ps aux | grep 'warehouse-load rows'   # local; "
            "check Cloud Run job history once scheduled execution lands"
        ),
        project=project,
    )
    sys.exit(2)


def _assert_run_invariant(*, summary: RowsRunSummary, log: Any) -> None:
    """PRD §8.7 — every candidate doc must end up in exactly one
    disposition bucket. A delta means a worker raised an uncaught
    exception that didn't get mapped to a load_status — that's a
    code bug, surfaced loudly on the first run."""
    total_disposed = (
        summary.docs_loaded
        + summary.docs_blob_missing
        + summary.docs_parse_failed
        + summary.docs_skipped_already_loaded
    )
    if total_disposed == summary.candidate_count:
        return
    delta = summary.candidate_count - total_disposed
    log.error(
        "run_invariant_violated",
        candidate_count=summary.candidate_count,
        total_disposed=total_disposed,
        delta=delta,
        summary=dataclasses.asdict(summary),
    )
    raise RuntimeError(
        f"docs accounted-for mismatch: "
        f"candidate_count={summary.candidate_count} "
        f"vs disposed={total_disposed}. "
        f"{delta} docs are unaccounted for — this is a logic bug, "
        "not a data issue. Investigate the per-doc loop (workers may "
        "have raised an uncaught exception).",
    )


# --------------------------------------------------------------------
# batch processing
# --------------------------------------------------------------------


def _run_doc_batch(
    *,
    docs: list[DocumentRow],
    bq: BqClient | None,
    gcs: GcsStreamClient | None,
    settings: Settings,
    staging_table: str,
    rows_schema: list[bigquery.SchemaField],
    request: RowsRunRequest,
    log: Any,
) -> list[DocumentLoadResult]:
    """Run up to `concurrency` docs in parallel; return the results."""
    if not docs:
        return []

    concurrency = request.concurrency or settings.rows_concurrency
    timeout = settings.per_doc_timeout_seconds

    results: list[DocumentLoadResult] = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_doc = {
            pool.submit(
                _load_one_document,
                doc=doc,
                bq=bq,
                gcs=gcs,
                settings=settings,
                staging_table=staging_table,
                rows_schema=rows_schema,
                dry_run=request.dry_run,
                force=request.force,
                log=log,
            ): doc
            for doc in docs
        }
        for future in cf.as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                results.append(future.result(timeout=timeout))
            except cf.TimeoutError:
                log.error(
                    "doc_load_timeout",
                    document_id=doc.document_id,
                    timeout_seconds=timeout,
                )
                # Best-effort cancel; running threads can't be killed
                # cleanly in Python — accepted leak since the runner
                # is short-lived.
                future.cancel()
                _record_failure_outcome(
                    bq=bq,
                    settings=settings,
                    doc=doc,
                    load_error=f"timed_out_after_{timeout}s",
                    final_status="parse_failed",
                    dry_run=request.dry_run,
                )
                results.append(
                    _make_failure_result(
                        doc=doc,
                        final_status="parse_failed",
                        load_error=f"timed_out_after_{timeout}s",
                        bytes_read=0,
                        duration_ms=timeout * 1000,
                    ),
                )
            except Exception as exc:
                # An uncaught exception in the worker is itself a
                # bug; map to parse_failed so the §8.7 invariant
                # holds and the operator can investigate.
                log.exception(
                    "doc_load_uncaught_exception",
                    document_id=doc.document_id,
                    error=str(exc),
                )
                _record_failure_outcome(
                    bq=bq,
                    settings=settings,
                    doc=doc,
                    load_error=f"uncaught_exception: {exc}"[:1024],
                    final_status="parse_failed",
                    dry_run=request.dry_run,
                )
                results.append(
                    _make_failure_result(
                        doc=doc,
                        final_status="parse_failed",
                        load_error=f"uncaught_exception: {exc}"[:1024],
                        bytes_read=0,
                        duration_ms=0,
                    ),
                )
    return results


# --------------------------------------------------------------------
# per-doc worker
# --------------------------------------------------------------------


@dataclass
class _DocWorkPaths:
    """Per-doc scratch files. Lifecycle managed by the worker."""

    raw_blob: Path
    utf8_csv: Path | None = None
    staging_jsonl: Path | None = None
    extra: list[Path] = field(default_factory=list)


def _load_one_document(
    *,
    doc: DocumentRow,
    bq: BqClient | None,
    gcs: GcsStreamClient | None,
    settings: Settings,
    staging_table: str,
    rows_schema: list[bigquery.SchemaField],
    dry_run: bool,
    force: bool,
    log: Any,
) -> DocumentLoadResult:
    """One doc's worth of work. Returns a DocumentLoadResult.

    Steps:
      1. Skip-loaded check (unless --force).
      2. Mark in-flight on raw.documents.
      3. Download blob → temp file.
      4. Sniff delimiter + encoding.
      5. Optional utf-8 conversion (for non-utf-8 files).
      6. Lookahead → detect header.
      7. Stream body → JSONL temp file.
      8. Append JSONL to raw.rows_staging.
      9. (Failure paths only) record terminal outcome on raw.documents.

    Success outcomes are recorded by the batch flusher AFTER the
    MERGE lands. Failure outcomes are recorded here because nothing
    is appended to staging for them.
    """
    started = time.monotonic()
    documents_table = (
        f"{settings.gcp_project_id}.{settings.bq_dataset_raw}.{settings.bq_documents_table}"
    )

    log.info("doc_load_start", document_id=doc.document_id, gcs_uri=doc.gcs_uri,
             file_format=doc.file_format)

    if doc.gcs_uri is None:
        # Should never happen — 3.2's bucket intersection filters
        # gcs_uri=None at catalog load. Defensive.
        _record_failure_outcome(
            bq=bq, settings=settings, doc=doc,
            load_error="gcs_uri is null",
            final_status="parse_failed", dry_run=dry_run,
        )
        return _make_failure_result(
            doc=doc, final_status="parse_failed",
            load_error="gcs_uri is null", bytes_read=0,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if gcs is None:
        if not dry_run:
            raise ValueError("gcs client required when dry_run is False")
        # Dry-run without a gcs client → can't fetch bytes; report
        # parse_failed so the §8.7 invariant still holds.
        return _make_failure_result(
            doc=doc, final_status="parse_failed",
            load_error="dry_run_without_gcs_client", bytes_read=0,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # Step 2: mark in-flight (skipped in dry-run).
    if not dry_run:
        assert bq is not None
        document_status.mark_in_flight(
            bq=bq, documents_table=documents_table, document_id=doc.document_id,
        )

    # Steps 3-8 inside a try so per-doc temp files clean up on error.
    blob_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — lifecycle managed by finally
        prefix=f"wh_rows_{doc.document_id[:12]}_", suffix=".csv", delete=False,
    )
    paths = _DocWorkPaths(raw_blob=Path(blob_tmp.name))
    # Default to "this worker owns the JSONL"; set to False only when we
    # return a success result and hand the path off to the orchestrator
    # for post-flush cleanup. Anything else (parse_failed, max_rows
    # exceeded, exception) leaves us responsible for unlinking it here.
    handoff_staging_jsonl = False
    try:
        try:
            # `NamedTemporaryFile` returns a `_TemporaryFileWrapper`, not a
            # raw `BinaryIO`, but its `write`/`seek`/`truncate`/`flush` are
            # the protocol's interface. The cast keeps the strict-mypy
            # boundary on `download_blob_to_file` honest without leaking
            # the wrapper type to clients.
            bytes_read = gcs.download_blob_to_file(
                gcs_uri=doc.gcs_uri,
                sink=blob_tmp,  # type: ignore[arg-type]
                max_bytes=settings.max_bytes_per_doc,
            )
        finally:
            blob_tmp.close()

        # Step 4: sniff.
        with paths.raw_blob.open("rb") as f:
            head = f.read(settings.sniff_buffer_bytes)
        sniff = sniff_csv(head)
        log.info("csv_encoding_detected", document_id=doc.document_id,
                 encoding=sniff.encoding)
        log.info("csv_delimiter_detected", document_id=doc.document_id,
                 delimiter=sniff.delimiter)
        if _delimiter_disagrees(doc, sniff):
            log.warning(
                "csv_delimiter_disagrees_with_declared",
                document_id=doc.document_id,
                declared_format=doc.file_format,
                sniffed_delimiter=sniff.delimiter,
            )

        # Step 5-7: parse with one-shot latin-1 retry on
        # mid-stream UnicodeDecodeError.
        sniff_used, result = _parse_with_latin1_fallback(
            doc=doc, paths=paths, sniff=sniff, settings=settings, log=log,
        )

        if result.final_status != "loaded":
            # parse failed inside the streaming pass; record outcome
            # and return — no staging append needed.
            if not dry_run:
                assert bq is not None
                document_status.record_load_outcome(
                    bq=bq, documents_table=documents_table,
                    document_id=doc.document_id,
                    load_status=result.final_status,
                    load_error=result.load_error,
                    preamble_rows=None, header_confidence=None, row_count=None,
                )
            return dataclasses.replace(
                result,
                bytes_read=bytes_read,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Step 8: the JSONL is kept local; the orchestrator concatenates
        # all pending per-doc JSONLs and does ONE BQ load job per flush
        # (see `_flush_and_record`). Per-doc load jobs would otherwise
        # blow through BQ's 1500-load-job-per-table-per-day quota on a
        # corpus of more than ~1500 docs.

        log.info(
            "doc_load_success",
            document_id=doc.document_id,
            row_count=result.row_count,
            bytes_read=bytes_read,
            duration_ms=int((time.monotonic() - started) * 1000),
            encoding_used=sniff_used.encoding,
        )

        handoff_staging_jsonl = True
        return dataclasses.replace(
            result,
            bytes_read=bytes_read,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    except BlobMissingError as exc:
        log.warning("doc_load_failed", document_id=doc.document_id,
                    load_status="blob_missing", load_error=str(exc))
        _record_failure_outcome(
            bq=bq, settings=settings, doc=doc,
            load_error=str(exc), final_status="blob_missing", dry_run=dry_run,
        )
        return _make_failure_result(
            doc=doc, final_status="blob_missing", load_error=str(exc),
            bytes_read=0, duration_ms=int((time.monotonic() - started) * 1000),
        )
    except BytesCapExceededError as exc:
        log.warning("doc_load_failed", document_id=doc.document_id,
                    load_status="parse_failed", load_error=str(exc))
        _record_failure_outcome(
            bq=bq, settings=settings, doc=doc,
            load_error=str(exc), final_status="parse_failed", dry_run=dry_run,
        )
        return _make_failure_result(
            doc=doc, final_status="parse_failed", load_error=str(exc),
            bytes_read=0, duration_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        log.exception("doc_load_failed", document_id=doc.document_id, error=str(exc))
        load_error = f"{type(exc).__name__}: {exc}"[:1024]
        _record_failure_outcome(
            bq=bq, settings=settings, doc=doc,
            load_error=load_error, final_status="parse_failed", dry_run=dry_run,
        )
        return _make_failure_result(
            doc=doc, final_status="parse_failed", load_error=load_error,
            bytes_read=0, duration_ms=int((time.monotonic() - started) * 1000),
        )
    finally:
        # Raw blob and utf-8 copy are always per-worker; clean them.
        paths.raw_blob.unlink(missing_ok=True)
        if paths.utf8_csv is not None:
            paths.utf8_csv.unlink(missing_ok=True)
        for extra in paths.extra:
            extra.unlink(missing_ok=True)
        # The staging JSONL is handed off to the orchestrator ONLY on
        # the success path (see `handoff_staging_jsonl = True` above).
        # Failure paths (parse_failed, max_rows exceeded, exception)
        # would otherwise leak the partially-written JSONL into /tmp —
        # the failure result has staging_jsonl_path=None, so the
        # orchestrator's post-batch cleanup never sees it.
        if not handoff_staging_jsonl and paths.staging_jsonl is not None:
            paths.staging_jsonl.unlink(missing_ok=True)


def _parse_with_latin1_fallback(
    *,
    doc: DocumentRow,
    paths: _DocWorkPaths,
    sniff: SniffResult,
    settings: Settings,
    log: Any,
) -> tuple[SniffResult, DocumentLoadResult]:
    """Run header detect + stream; retry once with latin-1 on a
    mid-stream UnicodeDecodeError. PRD §5.2.1.

    Returns the (possibly fallback-adjusted) sniff and a DocumentLoadResult
    in either `loaded` or `parse_failed` state. bytes_read /
    duration_ms are stamped by the caller.
    """
    import polars as pl  # local import to keep the module-level surface tight

    try:
        return sniff, _parse_one_pass(doc=doc, paths=paths, sniff=sniff,
                                       settings=settings, log=log)
    except UnicodeDecodeError:
        # Python-level decode error during prepare_utf8_copy.
        pass
    except pl.exceptions.ComputeError as exc:
        # polars 1.x raises ComputeError, NOT UnicodeDecodeError, for
        # mid-stream encoding failures (`invalid utf-8 sequence ...`).
        # Re-raise anything that isn't encoding-related so the broader
        # parse_failed path still surfaces real bugs.
        if "invalid utf-8" not in str(exc).lower():
            raise

    log.warning(
        "csv_encoding_fallback_to_latin1",
        document_id=doc.document_id,
        sniffed=sniff.encoding,
    )
    # Reset paths for the retry: prior pass's utf-8 copy and
    # staging JSONL are stale.
    if paths.utf8_csv is not None:
        paths.utf8_csv.unlink(missing_ok=True)
        paths.utf8_csv = None
    if paths.staging_jsonl is not None:
        paths.staging_jsonl.unlink(missing_ok=True)
        paths.staging_jsonl = None
    fallback = dataclasses.replace(sniff, encoding="latin-1")
    return fallback, _parse_one_pass(doc=doc, paths=paths, sniff=fallback,
                                      settings=settings, log=log)


def _parse_one_pass(
    *,
    doc: DocumentRow,
    paths: _DocWorkPaths,
    sniff: SniffResult,
    settings: Settings,
    log: Any,
) -> DocumentLoadResult:
    """Single pass: convert to utf-8 if needed, detect header, stream
    body to a per-doc JSONL temp file.

    Returns `loaded` on success or `parse_failed` on rule violations
    (max_rows exceeded, malformed encoding the latin-1 retry can't
    recover from, etc.).
    """
    csv_path = paths.raw_blob
    if needs_utf8_conversion(sniff.encoding):
        utf8_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — lifecycle via paths
            prefix=f"wh_rows_utf8_{doc.document_id[:12]}_",
            suffix=".csv",
            delete=False,
        )
        utf8_tmp.close()
        paths.utf8_csv = Path(utf8_tmp.name)
        prepare_utf8_copy(
            source_path=paths.raw_blob,
            encoding=sniff.encoding,
            dest_path=paths.utf8_csv,
        )
        csv_path = paths.utf8_csv

    lookahead = list(
        iter_lookahead_rows(
            path=csv_path,
            sniff=sniff,
            max_rows=settings.body_min_run + settings.header_lookback,
        ),
    )
    header = detect_header(
        iter(lookahead),
        body_min_run=settings.body_min_run,
        header_lookback=settings.header_lookback,
        body_modal_match_ratio=settings.body_modal_match_ratio,
        header_max_cell_chars=settings.header_max_cell_chars,
    )
    log.info(
        "header_detect_done",
        document_id=doc.document_id,
        body_start_index=header.body_start_index,
        header_rows=[list(r) for r in header.header_rows],
        confidence=header.confidence,
        preamble_row_count=len(header.preamble_rows),
    )

    staging_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — lifecycle via paths
        prefix=f"wh_rows_jsonl_{doc.document_id[:12]}_",
        suffix=".jsonl",
        delete=False,
    )
    paths.staging_jsonl = Path(staging_tmp.name)

    row_count = 0
    nul_event_emitted = False
    loaded_at_iso = datetime.now(UTC).isoformat()
    try:
        with paths.staging_jsonl.open("w", encoding="utf-8") as out:
            for staging_row in stream_body_rows(
                path=csv_path,
                sniff=sniff,
                header=header,
            ):
                if staging_row.nul_stripped and not nul_event_emitted:
                    log.warning("csv_nul_stripped", document_id=doc.document_id)
                    nul_event_emitted = True
                row_count += 1
                if row_count > settings.max_rows_per_doc:
                    return _make_failure_result(
                        doc=doc, final_status="parse_failed",
                        load_error=f"exceeded max_rows_per_doc={settings.max_rows_per_doc}",
                        bytes_read=0, duration_ms=0,
                    )
                json.dump(
                    {
                        "document_id": doc.document_id,
                        "row_index": staging_row.row_index,
                        # `row` lands as a BQ JSON column — dump as a
                        # JSON string and BQ parses it on load.
                        "row": json.dumps(staging_row.row, ensure_ascii=False),
                        "loaded_at": loaded_at_iso,
                    },
                    out,
                    ensure_ascii=False,
                )
                out.write("\n")
    finally:
        staging_tmp.close()

    return DocumentLoadResult(
        document_id=doc.document_id,
        organization_code=doc.organization_code,
        final_status="loaded",
        load_error=None,
        preamble_rows=header.preamble_rows,
        header_confidence=header.confidence,
        row_count=row_count,
        bytes_read=0,  # stamped by caller
        duration_ms=0,  # stamped by caller
        staging_jsonl_path=paths.staging_jsonl,
    )


def _delimiter_disagrees(doc: DocumentRow, sniff: SniffResult) -> bool:
    """`'csv'` declared + tab sniffed → disagrees. `'tsv'` declared +
    comma sniffed → disagrees. Both are observable in the corpus per
    PRD §14 decision 8.
    """
    return (doc.file_format == "csv" and sniff.delimiter == "\t") or (
        doc.file_format == "tsv" and sniff.delimiter == ","
    )


def _record_failure_outcome(
    *,
    bq: BqClient | None,
    settings: Settings,
    doc: DocumentRow,
    load_error: str,
    final_status: document_status.LoadStatus,
    dry_run: bool,
) -> None:
    """Per-doc failure paths land their outcome immediately — no
    staging append, no batch flush. The success path defers recording
    until after the MERGE."""
    if dry_run or bq is None:
        return
    documents_table = (
        f"{settings.gcp_project_id}.{settings.bq_dataset_raw}.{settings.bq_documents_table}"
    )
    document_status.record_load_outcome(
        bq=bq, documents_table=documents_table,
        document_id=doc.document_id,
        load_status=final_status, load_error=load_error,
        preamble_rows=None, header_confidence=None, row_count=None,
    )


def _make_failure_result(
    *,
    doc: DocumentRow,
    final_status: document_status.LoadStatus,
    load_error: str,
    bytes_read: int,
    duration_ms: int,
) -> DocumentLoadResult:
    return DocumentLoadResult(
        document_id=doc.document_id,
        organization_code=doc.organization_code,
        final_status=final_status,
        load_error=load_error,
        preamble_rows=None,
        header_confidence=None,
        row_count=None,
        bytes_read=bytes_read,
        duration_ms=duration_ms,
        staging_jsonl_path=None,
    )


def _make_skipped_result(doc: DocumentRow) -> DocumentLoadResult:
    """Already-loaded doc that the operator chose not to replay."""
    return DocumentLoadResult(
        document_id=doc.document_id,
        organization_code=doc.organization_code,
        final_status="loaded",  # the doc is loaded — we're just not re-touching it
        load_error=None,
        preamble_rows=None,
        header_confidence=None,
        row_count=None,
        bytes_read=0,
        duration_ms=0,
        staging_jsonl_path=None,
        skipped_already_loaded=True,
    )


def _finish(*, summary: RowsRunSummary, log: Any, run_id: str) -> RowsRunSummary:
    """Common post-run path: invariant check + structured log."""
    _assert_run_invariant(summary=summary, log=log)
    log.info("rows_load_finish", run_id=run_id, summary=dataclasses.asdict(summary))
    return summary


# --------------------------------------------------------------------
# batch flush + outcome recording
# --------------------------------------------------------------------


def _flush_and_record(
    *,
    bq: BqClient | None,
    log: Any,
    run_id: str,
    staging_table: str,
    rows_table: str,
    documents_table: str,
    pending: list[DocumentLoadResult],
    staging_rows: int,
    rows_schema: list[bigquery.SchemaField],
) -> int:
    """Concatenate all pending per-doc JSONLs, append them to staging
    in a SINGLE BQ load job, MERGE staging into raw.rows, then record
    `load_status='loaded'` for each doc in the batch.

    Coalescing the per-doc appends into one load job per flush keeps
    the run under BQ's 1500-load-job-per-table-per-day quota: at
    `flush_files_threshold=32`, 14K docs become ~440 load jobs.

    The MERGE and the per-doc UPDATEs are NOT atomic together — a
    crash between them leaves rows in `raw.rows` but the doc still
    marked `pending`. The next run picks the doc back up, the MERGE's
    WHEN MATCHED THEN DELETE branch fires (idempotent), and the
    UPDATE re-runs. At-least-once retry, as documented in PRD §8.5.
    """
    if bq is None:
        return 0
    started = time.monotonic()

    # Step 1: concat + ONE append. The JSONL format is line-delimited,
    # so concatenation is byte-level — no parse/serialise round-trip.
    jsonl_paths = [
        r.staging_jsonl_path for r in pending if r.staging_jsonl_path is not None
    ]
    if jsonl_paths:
        concat_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — lifecycle via try/finally
            prefix="wh_rows_concat_", suffix=".jsonl", delete=False,
        )
        concat_path = Path(concat_tmp.name)
        try:
            with concat_path.open("wb") as dst:
                for src_path in jsonl_paths:
                    with src_path.open("rb") as src:
                        # 1 MiB chunks: bounded memory regardless of
                        # individual JSONL size.
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
            concat_tmp.close()
            bq.append_jsonl_file(
                jsonl_path=concat_path,
                destination=staging_table,
                schema=rows_schema,
            )
        finally:
            concat_path.unlink(missing_ok=True)

    # Step 2: MERGE staging → raw.rows.
    rows_merge.flush_batch(
        bq=bq, staging_table=staging_table, target_table=rows_table,
    )
    log.info(
        "batch_merge_done",
        run_id=run_id,
        batch_size=len(pending),
        rows_merged=staging_rows,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    # Per-doc UPDATEs are post-MERGE bookkeeping: the rows already
    # landed in raw.rows. A transient network blip on any one UPDATE
    # would otherwise kill the whole orchestrator (Google SDK retries
    # for 600s before raising). Swallow per-doc UPDATE failures so the
    # rest of the run continues — the affected doc stays `pending`
    # and the next run re-processes it idempotently (MERGE WHEN
    # MATCHED THEN DELETE handles the replay). At-least-once, as
    # documented in PRD §8.5.
    record_failures = 0
    for result in pending:
        try:
            document_status.record_load_outcome(
                bq=bq, documents_table=documents_table,
                document_id=result.document_id,
                load_status="loaded", load_error=None,
                preamble_rows=result.preamble_rows,
                header_confidence=result.header_confidence,
                row_count=result.row_count,
            )
        except Exception as exc:
            record_failures += 1
            log.warning(
                "record_load_outcome_failed_will_retry_next_run",
                run_id=run_id,
                document_id=result.document_id,
                error=str(exc)[:512],
            )
    if record_failures:
        log.warning(
            "record_load_outcome_batch_partial",
            run_id=run_id,
            failed=record_failures,
            total=len(pending),
        )
    return staging_rows


# --------------------------------------------------------------------
# summary
# --------------------------------------------------------------------


def _build_summary(
    *,
    run_id: str,
    request: RowsRunRequest,
    candidate_count: int,
    results: list[DocumentLoadResult],
    rows_merged: int,
    column_index_refreshed: bool,
    duration_ms: int,
) -> RowsRunSummary:
    # Skipped docs carry `final_status='loaded'` (the doc IS loaded —
    # we just didn't re-touch it), so they're excluded from
    # `docs_loaded` here to keep the §8.7 invariant disjoint:
    #   candidate_count == loaded + blob_missing + parse_failed + skipped
    docs_loaded = sum(
        1
        for r in results
        if r.final_status == "loaded" and not r.skipped_already_loaded
    )
    docs_blob_missing = sum(1 for r in results if r.final_status == "blob_missing")
    docs_parse_failed = sum(1 for r in results if r.final_status == "parse_failed")
    docs_header_low_confidence = sum(
        1
        for r in results
        if (
            r.final_status == "loaded"
            and not r.skipped_already_loaded
            and r.header_confidence == "low"
        )
    )
    docs_skipped_already_loaded = sum(1 for r in results if r.skipped_already_loaded)
    return RowsRunSummary(
        run_id=run_id,
        dry_run=request.dry_run,
        candidate_count=candidate_count,
        docs_loaded=docs_loaded,
        docs_blob_missing=docs_blob_missing,
        docs_parse_failed=docs_parse_failed,
        docs_header_low_confidence=docs_header_low_confidence,
        docs_skipped_already_loaded=docs_skipped_already_loaded,
        rows_merged=rows_merged,
        column_index_refreshed=column_index_refreshed,
        duration_ms=duration_ms,
    )


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------


def _chunks(seq: list[DocumentRow], size: int) -> Iterator[list[DocumentRow]]:
    """Yield fixed-size sub-lists. Last chunk may be shorter."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
