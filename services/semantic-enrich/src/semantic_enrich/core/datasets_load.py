"""`datasets-load` orchestrator.

Laptop-side. Coalesces the per-flush JSONL for a run into one
`bq load` into a session-scoped staging table, then MERGEs into
`semantic.datasets` keyed by `package_id`.

The MERGE's `s.generated_at > t.generated_at` guard makes the
operation always-newer-wins and naturally idempotent: re-running
against the same stage is a no-op.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from google.api_core import exceptions as gax
from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import schema_loader, stage_io
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import DatasetsLoadRunSummary, StagedDatasetCard

# Columns the MERGE writes to (the target schema), in stable order.
# Provenance fields (`generation_model`, `generation_model_commit`,
# `generation_run_id`) are staging-only and projected away.
TARGET_COLUMNS: tuple[str, ...] = (
    "package_id",
    "summary",
    "grain",
    "measures",
    "dimensions",
    "date_range_start",
    "date_range_end",
    "embedding",
    "generated_at",
)


@dataclass(frozen=True)
class LoadRequest:
    run_id: str
    dry_run: bool


def preflight(*, settings: Settings, bq: BqClient) -> list[bigquery.SchemaField]:
    """Validate config + load + drift-guard the target schema."""
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "datasets-load; this subcommand writes to BigQuery."
        )
    log = get_logger("semantic_enrich.datasets_load")
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc

    schema_path = settings.schemas_dir / "semantic_datasets.json"
    schema = schema_loader.load_schema(schema_path)
    schema_loader.assert_datasets_schema(schema)
    return schema


def run_load(
    *,
    request: LoadRequest,
    settings: Settings,
    bq: BqClient,
    logger: structlog.BoundLogger | None = None,
) -> DatasetsLoadRunSummary:
    log = logger or get_logger("semantic_enrich.datasets_load")
    started = time.monotonic()

    schema = preflight(settings=settings, bq=bq)

    # Coalesce.
    coalesced_rows: list[StagedDatasetCard] = [
        row
        for _, _, row in stage_io.iter_staged_rows(
            run_id=request.run_id,
            artifact="datasets",
            staging_dir=settings.staging_dir,
            row_type=StagedDatasetCard,
        )
    ]
    embedding_null = sum(1 for r in coalesced_rows if r.embedding is None)
    loadable_rows = [r for r in coalesced_rows if r.embedding is not None]

    log.info(
        "datasets_load_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        coalesced_row_count=len(coalesced_rows),
        embedding_null_count=embedding_null,
    )

    # Reject obviously-broken staged rows (e.g. dry-run placeholders
    # accidentally piped to a real load) before they reach BQ.
    for r in loadable_rows:
        if len(r.summary) < 50:
            raise RuntimeError(
                f"datasets-load: row for package_id={r.package_id!r} has "
                f"summary length {len(r.summary)} < 50 chars; this looks "
                "like a dry-run placeholder. Refusing to load."
            )

    project_id = settings.gcp_project_id
    assert project_id is not None

    target_table_fq = (
        f"{project_id}.{settings.bq_dataset_semantic}.{settings.bq_datasets_table}"
    )

    if request.dry_run:
        return _dry_run_summary(
            request=request,
            bq=bq,
            target_table_fq=target_table_fq,
            coalesced=coalesced_rows,
            loadable=loadable_rows,
            embedding_null=embedding_null,
            log=log,
            started=started,
        )

    # Write coalesced load payload (target-schema only) to a tempfile.
    payload_path = settings.staging_dir / request.run_id / "_load_payload.jsonl"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_load_payload(path=payload_path, rows=loadable_rows)

    run_id_short = request.run_id.replace("-", "")[:12]
    staging_table_id = (
        f"{project_id}.{settings.bq_dataset_semantic}"
        f"._datasets_staging_{run_id_short}"
    )

    bq.create_staging_table(
        table_id=staging_table_id,
        schema=schema,
        expires_in=timedelta(hours=1),
    )
    try:
        rows_staged = bq.append_jsonl_file(
            jsonl_path=payload_path,
            destination=staging_table_id,
            schema=schema,
        )
        log.info(
            "datasets_staging_loaded",
            run_id=request.run_id,
            staging_table=staging_table_id,
            rows_staged=rows_staged,
        )

        # Snapshot per-package counts before the MERGE so we can report
        # inserts/updates/unchanged.
        before = _read_generated_at_by_package(
            bq=bq,
            target=target_table_fq,
            package_ids=[r.package_id for r in loadable_rows],
        )
        inserted = 0
        updated = 0
        unchanged = 0
        for r in loadable_rows:
            prior = before.get(r.package_id)
            if prior is None:
                inserted += 1
            elif r.generated_at > prior:
                updated += 1
            else:
                unchanged += 1

        merge_sql = _build_merge_sql(
            target=target_table_fq, staging=staging_table_id
        )
        m_started = time.monotonic()
        bq.execute(merge_sql)
        m_ms = int((time.monotonic() - m_started) * 1000)
        log.info(
            "datasets_merge_done",
            run_id=request.run_id,
            rows_inserted=inserted,
            rows_updated=updated,
            rows_unchanged=unchanged,
            duration_ms=m_ms,
        )
    finally:
        bq.delete_table(staging_table_id, not_found_ok=True)
        # Keep `_load_payload.jsonl` for post-mortem; archived with
        # the stage dir per parent §14 cleanup.

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = DatasetsLoadRunSummary(
        run_id=request.run_id,
        dry_run=False,
        coalesced_row_count=len(coalesced_rows),
        embedding_null_count=embedding_null,
        rows_staged=rows_staged,
        rows_inserted=inserted,
        rows_updated=updated,
        rows_unchanged=unchanged,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "datasets_load_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _build_merge_sql(*, target: str, staging: str) -> str:
    return f"""
MERGE INTO `{target}` t
USING `{staging}` s
  ON t.package_id = s.package_id

WHEN MATCHED AND s.generated_at > t.generated_at
THEN UPDATE SET
  summary           = s.summary,
  grain             = s.grain,
  measures          = s.measures,
  dimensions        = s.dimensions,
  date_range_start  = s.date_range_start,
  date_range_end    = s.date_range_end,
  embedding         = s.embedding,
  generated_at      = s.generated_at

WHEN NOT MATCHED THEN INSERT (
  package_id, summary, grain, measures, dimensions,
  date_range_start, date_range_end, embedding, generated_at
)
VALUES (
  s.package_id, s.summary, s.grain, s.measures, s.dimensions,
  s.date_range_start, s.date_range_end, s.embedding, s.generated_at
);
""".strip()


def _write_load_payload(*, path: Path, rows: list[StagedDatasetCard]) -> None:
    """Write target-schema-only JSONL for `bq load`."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "package_id": r.package_id,
                "summary": r.summary,
                "grain": r.grain,
                "measures": r.measures,
                "dimensions": r.dimensions,
                "date_range_start": r.date_range_start.isoformat()
                if r.date_range_start is not None
                else None,
                "date_range_end": r.date_range_end.isoformat()
                if r.date_range_end is not None
                else None,
                "embedding": r.embedding,
                "generated_at": r.generated_at.isoformat(),
            }
            import json as _json

            f.write(_json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def _read_generated_at_by_package(
    *, bq: BqClient, target: str, package_ids: list[str]
) -> dict[str, datetime]:
    """Map `package_id -> generated_at` for the subset already in the
    target table. Missing keys mean "not present" (insert path)."""
    if not package_ids:
        return {}
    sql = f"""
SELECT package_id, generated_at
FROM `{target}`
WHERE package_id IN UNNEST(@package_ids);
""".strip()
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [bigquery.ArrayQueryParameter("package_ids", "STRING", package_ids)]
    out: dict[str, datetime] = {}
    for row in bq.query_rows(sql, params=params):
        ga = row["generated_at"]
        if isinstance(ga, datetime):
            out[row["package_id"]] = ga
        else:
            # bigquery returns datetime by default; defend against str.
            out[row["package_id"]] = datetime.fromisoformat(str(ga))
    return out


def _dry_run_summary(
    *,
    request: LoadRequest,
    bq: BqClient,
    target_table_fq: str,
    coalesced: list[StagedDatasetCard],
    loadable: list[StagedDatasetCard],
    embedding_null: int,
    log: structlog.BoundLogger,
    started: float,
) -> DatasetsLoadRunSummary:
    """Compute would-be MERGE outcomes against the live target without
    touching the staging table."""
    before = _read_generated_at_by_package(
        bq=bq,
        target=target_table_fq,
        package_ids=[r.package_id for r in loadable],
    )
    inserted = updated = unchanged = 0
    for r in loadable:
        prior = before.get(r.package_id)
        if prior is None:
            action = "insert"
            inserted += 1
        elif r.generated_at > prior:
            action = "update"
            updated += 1
        else:
            action = "unchanged"
            unchanged += 1
        log.bind(package_id=r.package_id).info(
            "would_have_loaded", action=action
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = DatasetsLoadRunSummary(
        run_id=request.run_id,
        dry_run=True,
        coalesced_row_count=len(coalesced),
        embedding_null_count=embedding_null,
        rows_staged=len(loadable),
        rows_inserted=inserted,
        rows_updated=updated,
        rows_unchanged=unchanged,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "datasets_load_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _assert_invariant(
    summary: DatasetsLoadRunSummary, log: structlog.BoundLogger
) -> None:
    if (
        summary.rows_inserted + summary.rows_updated + summary.rows_unchanged
        != summary.rows_staged
    ):
        log.error(
            "run_invariant_violated",
            subcommand="datasets-load",
            check="rows_accounted_for",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"datasets-load rows accounted-for mismatch: {summary}"
        )
    if summary.rows_staged + summary.embedding_null_count != summary.coalesced_row_count:
        log.error(
            "run_invariant_violated",
            subcommand="datasets-load",
            check="coalesce",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"datasets-load coalesce mismatch: {summary}"
        )
