"""`columns-load` orchestrator.

Laptop-side. Coalesces the per-flush JSONL for a run into one
`bq load` into a session-scoped staging table, then MERGEs into
`semantic.columns` keyed by `(package_id, column_name)`.

The MERGE's `s.generated_at > t.generated_at` guard makes the
operation always-newer-wins and naturally idempotent: re-running
against the same stage is a no-op.
"""
from __future__ import annotations

import json
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
from semantic_enrich.types import ColumnsLoadRunSummary, StagedColumnRow

# Columns the MERGE writes to (target schema), in stable order.
# Provenance fields are staging-only and projected away.
TARGET_COLUMNS: tuple[str, ...] = (
    "package_id",
    "column_name",
    "semantic_type",
    "description",
    "sample_values",
    "embedding",
    "generated_at",
)


@dataclass(frozen=True)
class ColumnsLoadRequest:
    run_id: str
    dry_run: bool


def preflight(*, settings: Settings, bq: BqClient) -> list[bigquery.SchemaField]:
    """Validate config + load + drift-guard the target schema."""
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "columns-load; this subcommand writes to BigQuery."
        )
    log = get_logger("semantic_enrich.columns_load")
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc

    schema_path = settings.schemas_dir / "semantic_columns.json"
    schema = schema_loader.load_schema(schema_path)
    schema_loader.assert_columns_schema(schema)
    return schema


def run_load(
    *,
    request: ColumnsLoadRequest,
    settings: Settings,
    bq: BqClient,
    logger: structlog.BoundLogger | None = None,
) -> ColumnsLoadRunSummary:
    log = logger or get_logger("semantic_enrich.columns_load")
    started = time.monotonic()

    schema = preflight(settings=settings, bq=bq)

    # Coalesce: read everything under stage/<run_id>/columns/*.jsonl.
    coalesced_rows: list[StagedColumnRow] = [
        row
        for _, _, row in stage_io.iter_staged_rows(
            run_id=request.run_id,
            artifact="columns",
            staging_dir=settings.staging_dir,
            row_type=StagedColumnRow,
        )
    ]

    failure_marker_count = sum(1 for r in coalesced_rows if r.generation_failed)
    embedding_null = sum(
        1
        for r in coalesced_rows
        if not r.generation_failed and r.embedding is None
    )
    loadable_rows = [
        r
        for r in coalesced_rows
        if not r.generation_failed and r.embedding is not None
    ]

    log.info(
        "columns_load_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        coalesced_row_count=len(coalesced_rows),
        embedding_null_count=embedding_null,
        failure_marker_count=failure_marker_count,
    )

    # Pre-load validation: reject obviously-broken staged rows
    # (e.g. dry-run placeholders accidentally piped to a real load,
    # truncated descriptions, wrong-dim embeddings).
    if not request.dry_run:
        _preload_validate(
            rows=loadable_rows,
            embedding_dim=settings.embedding_dim,
            run_id=request.run_id,
            log=log,
        )

    project_id = settings.gcp_project_id
    assert project_id is not None

    target_table_fq = (
        f"{project_id}.{settings.bq_dataset_semantic}.{settings.bq_columns_table}"
    )

    if request.dry_run:
        return _dry_run_summary(
            request=request,
            bq=bq,
            target_table_fq=target_table_fq,
            coalesced=coalesced_rows,
            loadable=loadable_rows,
            embedding_null=embedding_null,
            failure_marker_count=failure_marker_count,
            log=log,
            started=started,
        )

    payload_path = settings.staging_dir / request.run_id / "_columns_load_payload.jsonl"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_load_payload(path=payload_path, rows=loadable_rows)

    run_id_short = request.run_id.replace("-", "")[:12]
    staging_table_id = (
        f"{project_id}.{settings.bq_dataset_semantic}"
        f"._columns_staging_{run_id_short}"
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
            "columns_staging_loaded",
            run_id=request.run_id,
            staging_table=staging_table_id,
            rows_staged=rows_staged,
        )

        before = _read_generated_at_by_key(
            bq=bq,
            target=target_table_fq,
            keys=[(r.package_id, r.column_name) for r in loadable_rows],
        )
        inserted = updated = unchanged = 0
        for r in loadable_rows:
            prior = before.get((r.package_id, r.column_name))
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
            "columns_merge_done",
            run_id=request.run_id,
            rows_inserted=inserted,
            rows_updated=updated,
            rows_unchanged=unchanged,
            duration_ms=m_ms,
        )
    finally:
        bq.delete_table(staging_table_id, not_found_ok=True)

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = ColumnsLoadRunSummary(
        run_id=request.run_id,
        dry_run=False,
        coalesced_row_count=len(coalesced_rows),
        embedding_null_count=embedding_null,
        failure_marker_count=failure_marker_count,
        rows_staged=rows_staged,
        rows_inserted=inserted,
        rows_updated=updated,
        rows_unchanged=unchanged,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "columns_load_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _build_merge_sql(*, target: str, staging: str) -> str:
    return f"""
MERGE INTO `{target}` t
USING `{staging}` s
  ON t.package_id  = s.package_id
 AND t.column_name = s.column_name

WHEN MATCHED AND s.generated_at > t.generated_at
THEN UPDATE SET
  semantic_type = s.semantic_type,
  description   = s.description,
  sample_values = s.sample_values,
  embedding     = s.embedding,
  generated_at  = s.generated_at

WHEN NOT MATCHED THEN INSERT (
  package_id, column_name, semantic_type, description,
  sample_values, embedding, generated_at
) VALUES (
  s.package_id, s.column_name, s.semantic_type, s.description,
  s.sample_values, s.embedding, s.generated_at
);
""".strip()


def _preload_validate(
    *,
    rows: list[StagedColumnRow],
    embedding_dim: int,
    run_id: str,
    log: structlog.BoundLogger,
) -> None:
    """Defence in depth before any `bq load`.

    Failures abort the load with `columns_preload_validation_failed`.
    The staging table is not touched; the operator fixes the staged
    JSONL (most likely by deleting it and re-running `columns-embed`)
    and retries.
    """
    for r in rows:
        if not r.column_name:
            _abort(log, run_id, r, "empty_column_name")
        if r.description is None or len(r.description) < 20:
            _abort(log, run_id, r, "description_too_short")
        if len(r.description) > 600:
            _abort(log, run_id, r, "description_too_long")
        if r.embedding is None or len(r.embedding) != embedding_dim:
            _abort(log, run_id, r, "wrong_embedding_dim")


def _abort(
    log: structlog.BoundLogger,
    run_id: str,
    row: StagedColumnRow,
    reason: str,
) -> None:
    log.error(
        "columns_preload_validation_failed",
        run_id=run_id,
        package_id=row.package_id,
        column_name=row.column_name,
        reason=reason,
        aborting=True,
    )
    raise RuntimeError(
        f"columns-load: row for package_id={row.package_id!r}, "
        f"column_name={row.column_name!r} failed pre-load validation: "
        f"{reason}. Refusing to load."
    )


def _write_load_payload(*, path: Path, rows: list[StagedColumnRow]) -> None:
    """Write target-schema-only JSONL for `bq load`."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "package_id": r.package_id,
                "column_name": r.column_name,
                "semantic_type": r.semantic_type,
                "description": r.description,
                "sample_values": r.sample_values,
                "embedding": r.embedding,
                "generated_at": r.generated_at.isoformat(),
            }
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def _read_generated_at_by_key(
    *,
    bq: BqClient,
    target: str,
    keys: list[tuple[str, str]],
) -> dict[tuple[str, str], datetime]:
    """Map `(package_id, column_name) -> generated_at` for the subset
    already in the target table.

    BQ doesn't have a struct-array parameter binding that translates
    to a 2-column IN clause directly. The pragmatic shape is two
    parallel arrays joined by index, then matched against the target.
    For the canonical full backfill of ~150K keys, this is one query
    rather than 150K.
    """
    if not keys:
        return {}
    pids = [k[0] for k in keys]
    cnames = [k[1] for k in keys]
    sql = f"""
WITH key_pairs AS (
  SELECT pid AS package_id, cname AS column_name
  FROM UNNEST(@package_ids) AS pid WITH OFFSET AS i
  JOIN UNNEST(@column_names) AS cname WITH OFFSET AS j ON i = j
)
SELECT t.package_id, t.column_name, t.generated_at
FROM `{target}` t
JOIN key_pairs k
  ON t.package_id = k.package_id
 AND t.column_name = k.column_name;
""".strip()
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("package_ids", "STRING", pids),
        bigquery.ArrayQueryParameter("column_names", "STRING", cnames),
    ]
    out: dict[tuple[str, str], datetime] = {}
    for row in bq.query_rows(sql, params=params):
        ga = row["generated_at"]
        if not isinstance(ga, datetime):
            ga = datetime.fromisoformat(str(ga))
        out[(str(row["package_id"]), str(row["column_name"]))] = ga
    return out


def _dry_run_summary(
    *,
    request: ColumnsLoadRequest,
    bq: BqClient,
    target_table_fq: str,
    coalesced: list[StagedColumnRow],
    loadable: list[StagedColumnRow],
    embedding_null: int,
    failure_marker_count: int,
    log: structlog.BoundLogger,
    started: float,
) -> ColumnsLoadRunSummary:
    """Compute would-be MERGE outcomes against the live target without
    touching the staging table."""
    before = _read_generated_at_by_key(
        bq=bq,
        target=target_table_fq,
        keys=[(r.package_id, r.column_name) for r in loadable],
    )
    inserted = updated = unchanged = 0
    for r in loadable:
        prior = before.get((r.package_id, r.column_name))
        if prior is None:
            action = "insert"
            inserted += 1
        elif r.generated_at > prior:
            action = "update"
            updated += 1
        else:
            action = "unchanged"
            unchanged += 1
        log.bind(
            package_id=r.package_id, column_name=r.column_name
        ).info("would_have_loaded", action=action)

    duration_ms = int((time.monotonic() - started) * 1000)
    summary = ColumnsLoadRunSummary(
        run_id=request.run_id,
        dry_run=True,
        coalesced_row_count=len(coalesced),
        embedding_null_count=embedding_null,
        failure_marker_count=failure_marker_count,
        rows_staged=len(loadable),
        rows_inserted=inserted,
        rows_updated=updated,
        rows_unchanged=unchanged,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "columns_load_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _assert_invariant(
    summary: ColumnsLoadRunSummary, log: structlog.BoundLogger
) -> None:
    if (
        summary.rows_inserted + summary.rows_updated + summary.rows_unchanged
        != summary.rows_staged
    ):
        log.error(
            "run_invariant_violated",
            subcommand="columns-load",
            check="rows_accounted_for",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"columns-load rows accounted-for mismatch: {summary}"
        )
    if (
        summary.rows_staged
        + summary.embedding_null_count
        + summary.failure_marker_count
        != summary.coalesced_row_count
    ):
        log.error(
            "run_invariant_violated",
            subcommand="columns-load",
            check="coalesce",
            summary=summary.__dict__,
        )
        raise RuntimeError(f"columns-load coalesce mismatch: {summary}")
