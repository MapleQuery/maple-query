"""`datasets-backfill-representative` orchestrator.

Laptop-side, picker-only. Re-runs the representative picker for every
package already present in `semantic.datasets` and MERGEs the chosen
`representative_document_id` — plus the `title` derived from it —
back onto the table. No GPU, no LLM.

Two uses:
- Populate the column for rows enriched before the picker persisted
  its choice.
- After a picker behaviour change, report (via `--dry-run` +
  `representative_pick_changed` events) which packages would swap
  representatives — those are the packages whose `semantic.columns`
  rows describe the wrong file and need a scoped re-enrichment run.

The MERGE touches `representative_document_id` only. `generated_at`
is deliberately left alone: it is the always-newer-wins clock for the
enrichment load path, and a picker-only backfill does not regenerate
the summary/embedding it stamps.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import structlog
from google.api_core import exceptions as gax
from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import package_grouper, sample_selector
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import RepresentativeBackfillRunSummary


@dataclass(frozen=True)
class RepresentativeBackfillRequest:
    run_id: str
    dry_run: bool
    limit_package_ids: list[str] | None


@dataclass(frozen=True)
class _Pick:
    package_id: str
    representative_document_id: str
    title: str | None
    previous_document_id: str | None
    dictionary_candidates: int


def run_backfill(
    *,
    request: RepresentativeBackfillRequest,
    settings: Settings,
    bq: BqClient,
    logger: structlog.BoundLogger | None = None,
) -> RepresentativeBackfillRunSummary:
    log = logger or get_logger("semantic_enrich.representative_backfill")
    started = time.monotonic()

    project_id = _require_project_id(settings)
    target = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_datasets_table}"
    )
    _preflight(bq=bq, log=log)

    previous_by_pkg = _read_current_representatives(
        bq=bq, target=target, limit_package_ids=request.limit_package_ids
    )
    log.info(
        "representative_backfill_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        target=target,
        packages_in_target=len(previous_by_pkg),
    )
    if not previous_by_pkg:
        raise RuntimeError(
            f"datasets-backfill-representative: no matching rows in "
            f"`{target}`; nothing to backfill."
        )

    candidate_rows = list(
        bq.query_rows(
            package_grouper.build_candidate_sql(
                project_id=project_id,
                dataset_raw=settings.bq_dataset_raw,
                documents_table=settings.bq_documents_table,
                with_limit=False,
            ),
            params=package_grouper.build_candidate_params(
                limit_orgs=None,
                limit_package_ids=sorted(previous_by_pkg),
                already_extracted=[],
                limit_packages=None,
            ),
        )
    )

    workers = max(1, settings.extract_concurrency)
    picks: list[_Pick] = []
    no_resources = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _pick_one_package,
                raw_row=raw_row,
                bq=bq,
                project_id=project_id,
                settings=settings,
                previous_by_pkg=previous_by_pkg,
                log=log,
            )
            for raw_row in candidate_rows
        ]
        for future in futures:
            pick = future.result()
            if pick is None:
                no_resources += 1
            else:
                picks.append(pick)

    # Packages in semantic.datasets with no loaded raw.documents rows
    # (should not happen; counted so the invariant stays visible).
    no_resources += len(previous_by_pkg) - len(candidate_rows)

    changed = [
        p for p in picks
        if p.representative_document_id != p.previous_document_id
    ]
    for p in changed:
        log.info(
            "representative_pick_changed",
            package_id=p.package_id,
            previous_document_id=p.previous_document_id,
            new_document_id=p.representative_document_id,
            dictionary_candidates=p.dictionary_candidates,
        )

    rows_merged = 0
    if request.dry_run:
        for p in picks:
            log.bind(package_id=p.package_id).info(
                "would_have_backfilled",
                representative_document_id=p.representative_document_id,
            )
    elif picks:
        rows_merged = _stage_and_merge(
            bq=bq,
            settings=settings,
            target=target,
            run_id=request.run_id,
            picks=picks,
            log=log,
        )

    summary = RepresentativeBackfillRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        packages_in_target=len(previous_by_pkg),
        packages_picked=len(picks),
        packages_no_resources=no_resources,
        picks_changed=len(changed),
        picks_unchanged=len(picks) - len(changed),
        rows_merged=rows_merged,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    _assert_invariant(summary, log)
    log.info(
        "representative_backfill_finish",
        run_id=request.run_id,
        duration_ms=summary.duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _pick_one_package(
    *,
    raw_row: dict[str, object],
    bq: BqClient,
    project_id: str,
    settings: Settings,
    previous_by_pkg: dict[str, str | None],
    log: structlog.BoundLogger,
) -> _Pick | None:
    """Worker-thread body: per-doc headers → picker → `_Pick`."""
    package_id, resources = package_grouper.decode_candidate_row(raw_row)
    plog = log.bind(package_id=package_id)
    if not resources:
        plog.warning("package_has_no_resources")
        return None

    columns_by_doc = package_grouper.fetch_doc_columns(
        bq=bq,
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
        document_ids=[r.document_id for r in resources],
    )
    rep = sample_selector.pick_representative(
        resources, columns_by_doc=columns_by_doc
    )
    dictionary_candidates = sum(
        1 for r in resources
        if sample_selector.looks_like_dictionary(
            columns_by_doc.get(r.document_id, [])
        )
    )
    plog.info(
        "representative_picked",
        resource_count=len(resources),
        dictionary_candidates=dictionary_candidates,
        chosen_document_id=rep.document_id,
        chosen_row_count=rep.row_count,
    )
    # Title mirrors dataset_generator._resolve_title: representative's
    # title, else first non-null resource title, else None (the UI
    # falls back to package_id; never invent one).
    title = rep.title or next((r.title for r in resources if r.title), None)
    return _Pick(
        package_id=package_id,
        representative_document_id=rep.document_id,
        title=title,
        previous_document_id=previous_by_pkg.get(package_id),
        dictionary_candidates=dictionary_candidates,
    )


def _require_project_id(settings: Settings) -> str:
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "datasets-backfill-representative; this subcommand reads/writes "
            "BigQuery."
        )
    return settings.gcp_project_id


def _preflight(*, bq: BqClient, log: structlog.BoundLogger) -> None:
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc


def _read_current_representatives(
    *,
    bq: BqClient,
    target: str,
    limit_package_ids: list[str] | None,
) -> dict[str, str | None]:
    """Map `package_id -> representative_document_id` (None when not
    yet backfilled) for the rows in scope."""
    sql = f"""
SELECT package_id, representative_document_id
FROM `{target}`
WHERE (@limit_package_ids IS NULL
       OR package_id IN UNNEST(@limit_package_ids))
ORDER BY package_id;
""".strip()
    params = [
        bigquery.ArrayQueryParameter(
            "limit_package_ids", "STRING", list(limit_package_ids or [])
        )
    ]
    out: dict[str, str | None] = {}
    for row in bq.query_rows(sql, params=params):
        pid = row.get("package_id")
        rep = row.get("representative_document_id")
        if isinstance(pid, str):
            out[pid] = rep if isinstance(rep, str) else None
    return out


def _stage_and_merge(
    *,
    bq: BqClient,
    settings: Settings,
    target: str,
    run_id: str,
    picks: list[_Pick],
    log: structlog.BoundLogger,
) -> int:
    project_id = settings.gcp_project_id
    assert project_id is not None
    run_id_short = run_id.replace("-", "")[:12]
    staging_table_id = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"_representative_backfill_{run_id_short}"
    )
    schema = [
        bigquery.SchemaField("package_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField(
            "representative_document_id", "STRING", mode="REQUIRED"
        ),
        bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
    ]

    payload_path = (
        settings.staging_dir / run_id / "_representative_backfill_payload.jsonl"
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(path=payload_path, picks=picks)

    bq.create_staging_table(
        table_id=staging_table_id,
        schema=schema,
        expires_in=timedelta(hours=24),
    )
    rows_staged = 0
    try:
        rows_staged = bq.append_jsonl_file(
            jsonl_path=payload_path,
            destination=staging_table_id,
            schema=schema,
        )
        log.info(
            "representative_backfill_staging_loaded",
            run_id=run_id,
            staging_table=staging_table_id,
            rows_staged=rows_staged,
        )
        merge_sql = f"""
MERGE INTO `{target}` t
USING `{staging_table_id}` s
  ON t.package_id = s.package_id
WHEN MATCHED THEN UPDATE SET
  representative_document_id = s.representative_document_id,
  title = s.title
""".strip()
        m_started = time.monotonic()
        bq.execute(merge_sql)
        log.info(
            "representative_backfill_merge_done",
            run_id=run_id,
            rows_merged=rows_staged,
            duration_ms=int((time.monotonic() - m_started) * 1000),
        )
    finally:
        bq.delete_table(staging_table_id, not_found_ok=True)
    return rows_staged


def _write_payload(*, path: Path, picks: list[_Pick]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for p in picks:
            f.write(
                json.dumps(
                    {
                        "package_id": p.package_id,
                        "representative_document_id": (
                            p.representative_document_id
                        ),
                        "title": p.title,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")
    tmp.replace(path)


def _assert_invariant(
    summary: RepresentativeBackfillRunSummary, log: structlog.BoundLogger
) -> None:
    if (
        summary.packages_picked + summary.packages_no_resources
        != summary.packages_in_target
    ):
        log.error(
            "run_invariant_violated",
            subcommand="datasets-backfill-representative",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            "datasets-backfill-representative packages accounted-for "
            f"mismatch: {summary}"
        )
    if summary.picks_changed + summary.picks_unchanged != summary.packages_picked:
        log.error(
            "run_invariant_violated",
            subcommand="datasets-backfill-representative",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            "datasets-backfill-representative picks accounted-for "
            f"mismatch: {summary}"
        )
