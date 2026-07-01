"""`datasets-reembed` and `columns-reembed` orchestrators.

One-off passes that overwrite `semantic.datasets.embedding` and
`semantic.columns.embedding` with OpenAI text-embedding-3-small vectors.
Source text (summary / description) stays exactly where it is; only
`embedding` + `generated_at` move.

Shape (both subcommands):
  1. Preflight: OPENAI_API_KEY present, one-vector ping matches
     `openai_embedding_dim`, target row count > 0, BQ auth ok.
  2. Read source text from BQ into memory.
  3. Batch through `OpenAIClient.embed`.
  4. Stage `(join_keys..., embedding)` rows into a temporary BQ table
     (auto-expires in 24h) and MERGE into the target updating
     `embedding` + `generated_at` only.

No `stage/<run_id>/*.jsonl` on disk — the source data is already
durable in BQ. A crash before the MERGE leaves the target untouched;
the staging table auto-expires. MERGE is a single BQ statement,
all-or-nothing at the row level.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import structlog
from google.api_core import exceptions as gax
from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.openai_embed import embed_texts_in_batches
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    ColumnsReembedRunSummary,
    DatasetsReembedRunSummary,
)

# ── Requests ──


@dataclass(frozen=True)
class DatasetsReembedRequest:
    run_id: str
    dry_run: bool
    batch_size: int | None  # None → settings.openai_embedding_batch_size


@dataclass(frozen=True)
class ColumnsReembedRequest:
    run_id: str
    dry_run: bool
    batch_size: int | None


# ── Datasets entry point ──


def run_datasets_reembed(
    *,
    request: DatasetsReembedRequest,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    logger: structlog.BoundLogger | None = None,
) -> DatasetsReembedRunSummary:
    log = logger or get_logger("semantic_enrich.reembed")
    started = time.monotonic()

    project_id = _require_project_id(settings)
    target = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_datasets_table}"
    )
    _preflight(
        settings=settings,
        bq=bq,
        openai_client=openai_client,
        target=target,
        log=log,
        subcommand="datasets-reembed",
        ping_openai=not request.dry_run,
    )

    read_sql = f"SELECT package_id, summary FROM `{target}` ORDER BY package_id"
    rows: list[tuple[str, str]] = []
    for row in bq.query_rows(read_sql):
        pid = row["package_id"]
        summary = row["summary"]
        if not isinstance(pid, str) or not isinstance(summary, str):
            raise RuntimeError(
                f"datasets-reembed: unexpected row shape from `{target}`: "
                f"{row!r}"
            )
        rows.append((pid, summary))

    log.info(
        "datasets_reembed_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        rows_read=len(rows),
        target=target,
    )

    batch_size = request.batch_size or settings.openai_embedding_batch_size

    if request.dry_run:
        for pid, _ in rows:
            log.info("would_have_reembedded", package_id=pid)
        summary = DatasetsReembedRunSummary(
            run_id=request.run_id,
            dry_run=True,
            rows_read=len(rows),
            rows_embedded=0,
            rows_failed=0,
            rows_merged=0,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        log.info(
            "datasets_reembed_finish",
            run_id=request.run_id,
            duration_ms=summary.duration_ms,
            summary=summary.__dict__,
        )
        return summary

    results = embed_texts_in_batches(
        client=openai_client,
        texts=[summary_text for _, summary_text in rows],
        batch_size=batch_size,
        expected_dim=settings.openai_embedding_dim,
        log=log,
        log_event_prefix="datasets_reembed",
    )
    embedded: list[tuple[str, list[float]]] = []
    failed = 0
    for (pid, _), result in zip(rows, results, strict=True):
        if result.vector is None:
            failed += 1
            log.error(
                "datasets_reembed_vector_invalid",
                package_id=pid,
                reason=result.failure_reason,
            )
            continue
        embedded.append((pid, result.vector))

    if not embedded:
        raise RuntimeError(
            f"datasets-reembed: {failed} vectors invalid, 0 usable; refusing "
            "to run an empty MERGE."
        )

    rows_merged = _stage_and_merge(
        bq=bq,
        settings=settings,
        target=target,
        run_id=request.run_id,
        embedded=[{"package_id": pid, "embedding": vec} for pid, vec in embedded],
        artifact="datasets_reembed",
        merge_sql_builder=_build_datasets_merge_sql,
        schema=_datasets_staging_schema(),
        log=log,
    )

    summary = DatasetsReembedRunSummary(
        run_id=request.run_id,
        dry_run=False,
        rows_read=len(rows),
        rows_embedded=len(embedded),
        rows_failed=failed,
        rows_merged=rows_merged,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    log.info(
        "datasets_reembed_finish",
        run_id=request.run_id,
        duration_ms=summary.duration_ms,
        summary=summary.__dict__,
    )
    return summary


# ── Columns entry point ──


def run_columns_reembed(
    *,
    request: ColumnsReembedRequest,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    logger: structlog.BoundLogger | None = None,
) -> ColumnsReembedRunSummary:
    log = logger or get_logger("semantic_enrich.reembed")
    started = time.monotonic()

    project_id = _require_project_id(settings)
    target = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_columns_table}"
    )
    _preflight(
        settings=settings,
        bq=bq,
        openai_client=openai_client,
        target=target,
        log=log,
        subcommand="columns-reembed",
        ping_openai=not request.dry_run,
    )

    read_sql = (
        f"SELECT package_id, column_name, description FROM `{target}` "
        "ORDER BY package_id, column_name"
    )
    rows: list[tuple[str, str, str]] = []
    for row in bq.query_rows(read_sql):
        pid = row["package_id"]
        col = row["column_name"]
        desc = row["description"]
        if (
            not isinstance(pid, str)
            or not isinstance(col, str)
            or not isinstance(desc, str)
        ):
            raise RuntimeError(
                f"columns-reembed: unexpected row shape from `{target}`: "
                f"{row!r}"
            )
        rows.append((pid, col, desc))

    log.info(
        "columns_reembed_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        rows_read=len(rows),
        target=target,
    )

    batch_size = request.batch_size or settings.openai_embedding_batch_size

    if request.dry_run:
        for pid, col, _ in rows:
            log.info("would_have_reembedded", package_id=pid, column_name=col)
        summary = ColumnsReembedRunSummary(
            run_id=request.run_id,
            dry_run=True,
            rows_read=len(rows),
            rows_embedded=0,
            rows_failed=0,
            rows_merged=0,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        log.info(
            "columns_reembed_finish",
            run_id=request.run_id,
            duration_ms=summary.duration_ms,
            summary=summary.__dict__,
        )
        return summary

    results = embed_texts_in_batches(
        client=openai_client,
        texts=[desc for _, _, desc in rows],
        batch_size=batch_size,
        expected_dim=settings.openai_embedding_dim,
        log=log,
        log_event_prefix="columns_reembed",
    )
    embedded: list[tuple[str, str, list[float]]] = []
    failed = 0
    for (pid, col, _), result in zip(rows, results, strict=True):
        if result.vector is None:
            failed += 1
            log.error(
                "columns_reembed_vector_invalid",
                package_id=pid,
                column_name=col,
                reason=result.failure_reason,
            )
            continue
        embedded.append((pid, col, result.vector))

    if not embedded:
        raise RuntimeError(
            f"columns-reembed: {failed} vectors invalid, 0 usable; refusing "
            "to run an empty MERGE."
        )

    rows_merged = _stage_and_merge(
        bq=bq,
        settings=settings,
        target=target,
        run_id=request.run_id,
        embedded=[
            {"package_id": pid, "column_name": col, "embedding": vec}
            for pid, col, vec in embedded
        ],
        artifact="columns_reembed",
        merge_sql_builder=_build_columns_merge_sql,
        schema=_columns_staging_schema(),
        log=log,
    )

    summary = ColumnsReembedRunSummary(
        run_id=request.run_id,
        dry_run=False,
        rows_read=len(rows),
        rows_embedded=len(embedded),
        rows_failed=failed,
        rows_merged=rows_merged,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    log.info(
        "columns_reembed_finish",
        run_id=request.run_id,
        duration_ms=summary.duration_ms,
        summary=summary.__dict__,
    )
    return summary


# ── Preflight + shared MERGE plumbing ──


def _require_project_id(settings: Settings) -> str:
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "*-reembed; this subcommand reads/writes BigQuery."
        )
    return settings.gcp_project_id


def _preflight(
    *,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    target: str,
    log: structlog.BoundLogger,
    subcommand: str,
    ping_openai: bool,
) -> None:
    if settings.openai_api_key is None:
        raise RuntimeError(
            f"{subcommand}: WHENRICH_OPENAI_API_KEY (or OPENAI_API_KEY) must "
            "be set."
        )
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc

    if ping_openai:
        ping = openai_client.embed(["ping"])
        if len(ping) != 1 or len(ping[0]) != settings.openai_embedding_dim:
            raise RuntimeError(
                f"{subcommand}: openai preflight returned unexpected shape "
                f"(vectors={len(ping)}, "
                f"dim={len(ping[0]) if ping else 'n/a'}, "
                f"expected_dim={settings.openai_embedding_dim}). "
                "Check WHENRICH_OPENAI_EMBEDDING_MODEL / "
                "WHENRICH_OPENAI_EMBEDDING_DIM."
            )

    count_sql = f"SELECT COUNT(*) AS n FROM `{target}`"
    rows = list(bq.query_rows(count_sql))
    n = int(rows[0]["n"]) if rows else 0
    if n == 0:
        raise RuntimeError(
            f"{subcommand}: `{target}` is empty; nothing to reembed."
        )
    log.info(
        "reembed_preflight_ok",
        subcommand=subcommand,
        target=target,
        target_row_count=n,
        openai_pinged=ping_openai,
        openai_embedding_dim=settings.openai_embedding_dim,
    )


def _stage_and_merge(
    *,
    bq: BqClient,
    settings: Settings,
    target: str,
    run_id: str,
    embedded: list[dict[str, object]],
    artifact: str,
    merge_sql_builder: Callable[[str, str], str],
    schema: list[bigquery.SchemaField],
    log: structlog.BoundLogger,
) -> int:
    project_id = settings.gcp_project_id
    assert project_id is not None
    run_id_short = run_id.replace("-", "")[:12]
    staging_table_id = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"_{artifact}_{run_id_short}"
    )

    payload_path = settings.staging_dir / run_id / f"_{artifact}_payload.jsonl"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(path=payload_path, rows=embedded)

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
            f"{artifact}_staging_loaded",
            run_id=run_id,
            staging_table=staging_table_id,
            rows_staged=rows_staged,
        )
        merge_sql = merge_sql_builder(target, staging_table_id)
        m_started = time.monotonic()
        bq.execute(merge_sql)
        log.info(
            f"{artifact}_merge_done",
            run_id=run_id,
            rows_merged=rows_staged,
            duration_ms=int((time.monotonic() - m_started) * 1000),
        )
    finally:
        bq.delete_table(staging_table_id, not_found_ok=True)
    return rows_staged


def _write_payload(*, path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def _datasets_staging_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("package_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]


def _columns_staging_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("package_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("column_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]


def _build_datasets_merge_sql(target: str, staging: str) -> str:
    return f"""
MERGE INTO `{target}` t
USING `{staging}` s
  ON t.package_id = s.package_id
WHEN MATCHED THEN UPDATE SET
  embedding = s.embedding,
  generated_at = CURRENT_TIMESTAMP()
""".strip()


def _build_columns_merge_sql(target: str, staging: str) -> str:
    return f"""
MERGE INTO `{target}` t
USING `{staging}` s
  ON t.package_id  = s.package_id
 AND t.column_name = s.column_name
WHEN MATCHED THEN UPDATE SET
  embedding = s.embedding,
  generated_at = CURRENT_TIMESTAMP()
""".strip()
