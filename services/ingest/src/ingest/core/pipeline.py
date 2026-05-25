"""Pipeline orchestrator.

Serial execution, filter-driven via `RunRequest`, writes bytes to GCS
and one JSONL record per resource to the run log. No BigQuery, no
watermark, no scheduler — a follow-up task loads the JSONL into BQ and
adds the metadata-/conditional-GET-based dedup layers.

Re-run idempotency comes from `gcs.upload()`'s write-time existence
check (`if_generation_match=0` + md5 compare) — same bytes go to the
same path, GCS returns `IdempotentSkip`, no duplicate write.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from ingest.clients.ckan import CkanClient, Dataset, Resource
from ingest.clients.gcs import (
    GcsClient,
    IdempotentSkip,
    PathCollision,
    Uploaded,
)
from ingest.clients.http import Downloaded, HttpClient
from ingest.config.settings import Settings
from ingest.config.sources import OrganizationConfig, SourceConfig, SourcesConfig
from ingest.core.dedup import compute_checksum, compute_document_id
from ingest.core.format_sniff import sniff_format
from ingest.core.language import Language, filter_resources_by_pairing
from ingest.core.path_builder import build_quarantine_path, build_raw_path
from ingest.core.quarantine import decide as quarantine_decide
from ingest.core.runlog import RunLogWriter
from ingest.providers.logging import get_logger
from ingest.types import DocumentRow

log = get_logger(__name__)


@dataclass(frozen=True)
class RunRequest:
    subject: str
    formats: tuple[str, ...] = ()
    limit_orgs: tuple[str, ...] = ()
    dry_run: bool = False
    since: datetime | None = None


@dataclass
class RunSummary:
    run_id: str
    request: RunRequest
    datasets_seen: int = 0
    resources_seen: int = 0
    success: int = 0
    quarantined: int = 0
    failed: int = 0
    skipped_by_pairing: int = 0
    skipped_by_gcs_dedup: int = 0  # GCS IdempotentSkip on re-upload
    duration_ms: int = 0


def run(
    *,
    settings: Settings,
    sources: SourcesConfig,
    request: RunRequest,
    ckans: dict[str, CkanClient],
    http: HttpClient,
    gcs: GcsClient,
    runlog: RunLogWriter,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunSummary:
    """Run one ingest pass. Caller owns lifecycle of all clients + the run log."""
    start = time.monotonic()
    summary = RunSummary(run_id=settings.run_id, request=request)

    log.info(
        "pipeline_start",
        run_id=settings.run_id,
        subject=request.subject,
        formats=list(request.formats),
        limit_orgs=list(request.limit_orgs),
        dry_run=request.dry_run,
        sources_count=len(sources),
        runlog_path=str(runlog.path),
    )

    ingest_date = clock().date()

    for source_cfg in sources:
        ckan = ckans.get(source_cfg.source)
        if ckan is None:
            log.warning("source_skipped_no_client", source=source_cfg.source)
            continue

        for org in source_cfg.organizations:
            if request.limit_orgs and org.code not in request.limit_orgs:
                continue
            _process_org(
                source_cfg=source_cfg,
                org=org,
                settings=settings,
                request=request,
                ckan=ckan,
                http=http,
                gcs=gcs,
                runlog=runlog,
                ingest_date=ingest_date,
                clock=clock,
                summary=summary,
            )

    summary.duration_ms = int((time.monotonic() - start) * 1000)
    log.info(
        "pipeline_finish",
        run_id=settings.run_id,
        duration_ms=summary.duration_ms,
        datasets_seen=summary.datasets_seen,
        resources_seen=summary.resources_seen,
        success=summary.success,
        quarantined=summary.quarantined,
        failed=summary.failed,
        skipped_by_pairing=summary.skipped_by_pairing,
        skipped_by_gcs_dedup=summary.skipped_by_gcs_dedup,
    )
    return summary


def _process_org(
    *,
    source_cfg: SourceConfig,
    org: OrganizationConfig,
    settings: Settings,
    request: RunRequest,
    ckan: CkanClient,
    http: HttpClient,
    gcs: GcsClient,
    runlog: RunLogWriter,
    ingest_date: Any,
    clock: Callable[[], datetime],
    summary: RunSummary,
) -> None:
    org_start = time.monotonic()
    since = request.since  # No watermark yet — --since is the only cursor.

    log.info(
        "org_start",
        country=source_cfg.country,
        source=source_cfg.source,
        organization=org.code,
        since=since.isoformat() if since else None,
    )

    counts = _OrgCounts()

    try:
        for dataset in ckan.search(
            subject=request.subject,
            formats=list(request.formats) or None,
            organization=org.code,
            since=since,
            page_size=source_cfg.page_size,
        ):
            counts.datasets += 1
            kept = filter_resources_by_pairing(dataset)
            counts.skipped_pairing += len(dataset.resources) - len(kept)

            for resource, lang in kept:
                if request.formats and not _resource_matches_formats(
                    resource, request.formats
                ):
                    continue
                counts.resources += 1

                row, outcome = _process_resource(
                    source_cfg=source_cfg,
                    org=org,
                    dataset=dataset,
                    resource=resource,
                    lang=lang,
                    settings=settings,
                    request=request,
                    http=http,
                    gcs=gcs,
                    ingest_date=ingest_date,
                    clock=clock,
                )

                if outcome == "skipped_gcs_dedup":
                    counts.skipped_gcs_dedup += 1
                    continue

                if row is None:
                    counts.failed += 1
                    continue

                if row.ingestion_status == "success":
                    counts.success += 1
                elif row.ingestion_status == "quarantined":
                    counts.quarantined += 1
                else:
                    counts.failed += 1

                if not request.dry_run:
                    runlog.write_row(row)

    except Exception as exc:
        log.error(
            "org_aborted",
            country=source_cfg.country,
            source=source_cfg.source,
            organization=org.code,
            error=str(exc),
            exc_info=True,
        )

    summary.datasets_seen += counts.datasets
    summary.resources_seen += counts.resources
    summary.success += counts.success
    summary.quarantined += counts.quarantined
    summary.failed += counts.failed
    summary.skipped_by_pairing += counts.skipped_pairing
    summary.skipped_by_gcs_dedup += counts.skipped_gcs_dedup

    log.info(
        "org_finish",
        country=source_cfg.country,
        source=source_cfg.source,
        organization=org.code,
        datasets_seen=counts.datasets,
        resources_seen=counts.resources,
        success=counts.success,
        quarantined=counts.quarantined,
        failed=counts.failed,
        skipped_by_pairing=counts.skipped_pairing,
        skipped_by_gcs_dedup=counts.skipped_gcs_dedup,
        duration_ms=int((time.monotonic() - org_start) * 1000),
    )


@dataclass
class _OrgCounts:
    datasets: int = 0
    resources: int = 0
    success: int = 0
    quarantined: int = 0
    failed: int = 0
    skipped_pairing: int = 0
    skipped_gcs_dedup: int = 0


def _process_resource(
    *,
    source_cfg: SourceConfig,
    org: OrganizationConfig,
    dataset: Dataset,
    resource: Resource,
    lang: Language,
    settings: Settings,
    request: RunRequest,
    http: HttpClient,
    gcs: GcsClient,
    ingest_date: Any,
    clock: Callable[[], datetime],
) -> tuple[DocumentRow | None, str]:
    """Process one resource. Returns (row_or_none, outcome).

    Outcomes: "row", "skipped_gcs_dedup". `row.ingestion_status`
    distinguishes success / quarantined / failed.
    """
    download: Downloaded | None = None
    download_error: Exception | None = None
    try:
        download = http.download(resource.url)
        # We don't send conditional headers yet (no etag store), so
        # http.download() never returns NotModified in normal operation.
    except Exception as exc:
        download_error = exc
        log.warning("download_failed", url=resource.url, error=str(exc))

    decision = quarantine_decide(
        download=download,
        download_error=download_error,
    )

    if decision.quarantine:
        return _build_quarantine_row(
            source_cfg=source_cfg,
            org=org,
            dataset=dataset,
            resource=resource,
            lang=lang,
            settings=settings,
            request=request,
            gcs=gcs,
            ingest_date=ingest_date,
            decision_reason=decision.reason or "download_failed",
            decision_note=decision.note,
            body=download.body if download else b"",
            clock=clock,
        ), "row"

    assert download is not None
    body = download.body
    checksum = compute_checksum(body)

    sniff = sniff_format(
        body=body, declared_format=resource.format_declared, url=resource.url
    )
    if sniff.mismatch_with_declared:
        log.info(
            "format_mismatch",
            url=resource.url,
            declared=resource.format_declared,
            sniffed=sniff.fmt,
        )

    document_id = compute_document_id(source_url=resource.url, checksum=checksum)
    path = build_raw_path(
        country=source_cfg.country,
        source=source_cfg.source,
        organization=org.code,
        ingest_date=ingest_date,
        fmt=sniff.fmt,
        document_id=document_id,
        resource_url=resource.url,
    )

    gcs_uri = f"gs://{settings.gcs_bucket}/{path}"

    if request.dry_run:
        log.info(
            "would_have_resource_success",
            document_id=document_id,
            gcs_uri=gcs_uri,
            bytes=len(body),
            fmt=sniff.fmt,
            declared_format=resource.format_declared,
            language=lang,
        )
    else:
        upload = gcs.upload(
            object_name=path,
            body=body,
            content_type=_header(download.headers, "Content-Type"),
        )
        if isinstance(upload, PathCollision):
            return _build_quarantine_row(
                source_cfg=source_cfg,
                org=org,
                dataset=dataset,
                resource=resource,
                lang=lang,
                settings=settings,
                request=request,
                gcs=gcs,
                ingest_date=ingest_date,
                decision_reason="path_collision",
                decision_note=(
                    f"existing md5 {upload.existing_md5_b64} "
                    f"!= attempted {upload.attempted_md5_b64}"
                ),
                body=body,
                clock=clock,
            ), "row"
        if isinstance(upload, IdempotentSkip):
            log.info("gcs_idempotent_skip", document_id=document_id, gcs_uri=gcs_uri)
            return None, "skipped_gcs_dedup"
        assert isinstance(upload, Uploaded)
        gcs_uri = upload.gcs_uri
        log.info(
            "resource_success",
            document_id=document_id,
            gcs_uri=gcs_uri,
            bytes=len(body),
            fmt=sniff.fmt,
            declared_format=resource.format_declared,
            language=lang,
        )

    return (
        DocumentRow(
            country_code=source_cfg.country,
            source_code=source_cfg.source,
            organization_code=org.code,
            document_id=document_id,
            source_url=resource.url,
            gcs_uri=gcs_uri,
            checksum=checksum,
            etag=_header(download.headers, "ETag"),
            http_last_modified=_parse_http_date(_header(download.headers, "Last-Modified")),
            resource_last_modified=resource.last_modified,
            file_format=sniff.fmt,
            declared_format=resource.format_declared,
            language=lang,
            title=dataset.title,
            document_type=None,
            subjects=list(dataset.subjects),
            published_date=None,
            metadata_modified=dataset.metadata_modified,
            ingested_at=clock(),
            ingestion_status="success",
            quarantine_reason=None,
            run_id=settings.run_id,
        ),
        "row",
    )


def _build_quarantine_row(
    *,
    source_cfg: SourceConfig,
    org: OrganizationConfig,
    dataset: Dataset,
    resource: Resource,
    lang: Language,
    settings: Settings,
    request: RunRequest,
    gcs: GcsClient,
    ingest_date: Any,
    decision_reason: str,
    decision_note: str,
    body: bytes,
    clock: Callable[[], datetime],
) -> DocumentRow:
    placeholder_checksum = compute_checksum(body) if body else "0" * 64
    document_id = compute_document_id(source_url=resource.url, checksum=placeholder_checksum)
    path = build_quarantine_path(
        country=source_cfg.country,
        source=source_cfg.source,
        ingest_date=ingest_date,
        reason=decision_reason,  # type: ignore[arg-type]
        document_id=document_id,
        resource_url=resource.url,
    )
    gcs_uri = f"gs://{settings.gcs_bucket}/{path}"

    if request.dry_run:
        log.info(
            "would_have_resource_quarantine",
            document_id=document_id,
            reason=decision_reason,
            note=decision_note,
            bytes=len(body),
            gcs_uri=gcs_uri,
        )
    elif body:
        upload = gcs.upload(object_name=path, body=body)
        if isinstance(upload, (Uploaded, IdempotentSkip)):
            gcs_uri = upload.gcs_uri
        log.info(
            "resource_quarantine",
            document_id=document_id,
            reason=decision_reason,
            note=decision_note,
            bytes=len(body),
            gcs_uri=gcs_uri,
        )
    else:
        log.info(
            "resource_failed",
            document_id=document_id,
            reason=decision_reason,
            note=decision_note,
        )

    ingestion_status = "quarantined" if body else "failed"

    return DocumentRow(
        country_code=source_cfg.country,
        source_code=source_cfg.source,
        organization_code=org.code,
        document_id=document_id,
        source_url=resource.url,
        gcs_uri=gcs_uri if body else None,
        checksum=placeholder_checksum if body else None,
        etag=None,
        http_last_modified=None,
        resource_last_modified=resource.last_modified,
        file_format=resource.format_declared.lower() if resource.format_declared else "unknown",
        declared_format=resource.format_declared,
        language=lang,
        title=dataset.title,
        document_type=None,
        subjects=list(dataset.subjects),
        published_date=None,
        metadata_modified=dataset.metadata_modified,
        ingested_at=clock(),
        ingestion_status=ingestion_status,
        quarantine_reason=decision_reason,
        run_id=settings.run_id,
    )


def _resource_matches_formats(resource: Resource, formats: tuple[str, ...]) -> bool:
    declared = (resource.format_declared or "").lower()
    return declared in formats


def _header(headers: Any, name: str) -> str | None:
    try:
        return headers[name]
    except (KeyError, TypeError):
        pass
    if isinstance(headers, dict):
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v
    return None


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
