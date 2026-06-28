"""Typer CLI for the warehouse loader.

```
uv run warehouse-load documents [--dry-run] [--since ISO]
                                [--limit-orgs CODE ...] [--no-bucket-check]
uv run warehouse-load rows      [--dry-run] [--limit-orgs CODE ...]
                                [--limit-documents ID ...] [--status STATUS]
                                [--force] [--concurrency N]
                                [--refresh-column-index]
uv run warehouse-load column-index [--dry-run]
```

Thin shim: builds `Settings` from env, overrides with flags, calls
into `core/`, prints the summary as JSON. No business logic here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from warehouse_load.clients.bq import RealBqClient
from warehouse_load.clients.gcs import RealGcsClient
from warehouse_load.clients.gcs_stream import RealGcsStreamClient
from warehouse_load.config.settings import Settings
from warehouse_load.core import column_index as column_index_mod
from warehouse_load.core.rows_runner import run_rows_load
from warehouse_load.core.runner import RunRequest, run_documents_load
from warehouse_load.providers.logging import configure_logging, get_logger
from warehouse_load.types import RowsRunRequest

app = typer.Typer(name="warehouse-load", help="MapleQuery warehouse loader CLI.")


@app.callback()
def _root() -> None:
    """Forces Typer to keep `documents` as an explicit subcommand
    (otherwise single-command apps get flattened), leaving room for
    a future sibling like `rows` without breaking the CLI surface."""


@app.command()
def documents(
    runlog_local_dir: Path | None = typer.Option(
        None, "--runlog-local-dir",
        help="Override WHLOAD_RUNLOG_LOCAL_DIR.",
    ),
    runlog_gcs_prefix: str | None = typer.Option(
        None, "--runlog-gcs-prefix",
        help="Override WHLOAD_RUNLOG_GCS_PREFIX (gs:// URI).",
    ),
    since: str | None = typer.Option(
        None, "--since",
        help="Only consider rows with ingested_at >= this ISO 8601 timestamp.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Dry-run skips BQ writes; emits would_have_merged events.",
    ),
    limit_orgs: list[str] | None = typer.Option(
        None, "--limit-orgs",
        help="Repeatable; restrict to these organization_code values.",
    ),
    no_bucket_check: bool = typer.Option(
        False, "--no-bucket-check",
        help=(
            "Skip the GCS bucket-intersection step. Logs a loud warning. "
            "Use only when the bucket is unreachable and you understand "
            "that zombie rows may land in raw.documents."
        ),
    ),
    allow_mass_blob_missing: bool = typer.Option(
        False, "--allow-mass-blob-missing",
        help=(
            "Disable the mass-blob-missing guardrail. Use only when you "
            "intentionally cleaned the bucket and expect most rows to "
            "drop as zombies."
        ),
    ),
) -> None:
    """Load raw.documents from ingest runlog JSONL files."""
    configure_logging()
    log = get_logger("warehouse_load.entrypoint")

    settings = Settings()  # type: ignore[call-arg]

    local_dir = runlog_local_dir if runlog_local_dir is not None else settings.runlog_local_dir
    gcs_prefix = (
        runlog_gcs_prefix if runlog_gcs_prefix is not None else settings.runlog_gcs_prefix
    )
    if local_dir is None and gcs_prefix is None:
        raise typer.BadParameter(
            "no runlog source: set --runlog-local-dir or --runlog-gcs-prefix "
            "(or WHLOAD_RUNLOG_LOCAL_DIR / WHLOAD_RUNLOG_GCS_PREFIX).",
        )

    parsed_since = _parse_since(since)

    bucket_prefix = settings.bucket_prefix if not no_bucket_check else None

    request = RunRequest(
        local_dir=local_dir,
        gcs_prefix=gcs_prefix,
        since=parsed_since,
        dry_run=dry_run,
        limit_orgs=tuple(limit_orgs or []),
        bucket_prefix=bucket_prefix,
        no_bucket_check=no_bucket_check,
        allow_mass_blob_missing=allow_mass_blob_missing,
    )

    log.info(
        "cli_invoked",
        dry_run=dry_run,
        runlog_local_dir=str(local_dir) if local_dir else None,
        runlog_gcs_prefix=gcs_prefix,
        since=parsed_since.isoformat() if parsed_since else None,
        limit_orgs=request.limit_orgs,
        bucket_prefix=bucket_prefix,
        no_bucket_check=no_bucket_check,
        allow_mass_blob_missing=allow_mass_blob_missing,
        gcp_project=settings.gcp_project_id,
    )

    bq = None if dry_run else RealBqClient.for_project(settings.gcp_project_id)
    # The same GcsClient instance services both runlog reads and the
    # bucket-existence check, so build it whenever either is needed.
    gcs = (
        RealGcsClient.for_project(settings.gcp_project_id)
        if gcs_prefix or bucket_prefix
        else None
    )

    summary = run_documents_load(
        request=request,
        bq=bq,
        gcs=gcs,
        project_id=settings.gcp_project_id,
        dataset=settings.bq_dataset_raw,
        table=settings.bq_documents_table,
        schemas_dir=settings.schemas_dir,
        run_id=settings.run_id,
    )

    typer.echo(json.dumps(asdict(summary), indent=2, default=str))


@app.command()
def rows(
    limit_orgs: list[str] | None = typer.Option(
        None, "--limit-orgs",
        help="Repeatable; restrict to these organization_code values.",
    ),
    limit_documents: list[str] | None = typer.Option(
        None, "--limit-documents",
        help="Repeatable; restrict to these document_id values.",
    ),
    status: str = typer.Option(
        "pending", "--status",
        help=(
            "Candidate-query filter on raw.documents.load_status. "
            "'pending' (default) picks up never-loaded and reset-on-refresh "
            "docs; 'parse_failed' / 'blob_missing' retry failures; 'loaded' "
            "is only meaningful with --force."
        ),
    ),
    force: bool = typer.Option(
        False, "--force/--no-force",
        help="With --force, already-loaded docs are reprocessed end-to-end.",
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency",
        help="Override WHLOAD_ROWS_CONCURRENCY (parallel docs per batch).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help=(
            "Dry-run runs the full sniff+detect+stream pipeline but skips "
            "all BQ writes. Useful for plumbing tests against the candidate "
            "set."
        ),
    ),
    refresh_column_index: bool = typer.Option(
        True, "--refresh-column-index/--no-refresh-column-index",
        help="After the rows run, refresh raw.column_index from raw.rows.",
    ),
) -> None:
    """Load CSV bodies into raw.rows. Reads candidates from raw.documents."""
    configure_logging()
    log = get_logger("warehouse_load.entrypoint")

    settings = Settings()  # type: ignore[call-arg]

    request = RowsRunRequest(
        limit_orgs=tuple(limit_orgs or []),
        limit_documents=tuple(limit_documents or []),
        status=status,
        force=force,
        concurrency=concurrency,
        dry_run=dry_run,
        refresh_column_index=refresh_column_index,
    )

    log.info(
        "cli_invoked",
        command="rows",
        dry_run=dry_run,
        status=status,
        force=force,
        concurrency=concurrency or settings.rows_concurrency,
        refresh_column_index=refresh_column_index,
        limit_orgs=request.limit_orgs,
        limit_documents=request.limit_documents,
        gcp_project=settings.gcp_project_id,
    )

    bq = None if dry_run else RealBqClient.for_project(settings.gcp_project_id)
    gcs = (
        None
        if dry_run
        else RealGcsStreamClient.for_project(settings.gcp_project_id)
    )

    summary = run_rows_load(
        request=request,
        bq=bq,
        gcs=gcs,
        settings=settings,
        run_id=settings.run_id,
    )

    typer.echo(json.dumps(asdict(summary), indent=2, default=str))


@app.command("column-index")
def column_index(
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Dry-run skips the BQ DDL. Logs the SQL that would have run.",
    ),
) -> None:
    """Standalone refresh of raw.column_index from raw.rows. Idempotent."""
    configure_logging()
    log = get_logger("warehouse_load.entrypoint")

    settings = Settings()  # type: ignore[call-arg]

    log.info(
        "cli_invoked",
        command="column-index",
        dry_run=dry_run,
        gcp_project=settings.gcp_project_id,
    )

    if dry_run:
        log.info("column_index_dry_run", run_id=settings.run_id)
        return

    bq = RealBqClient.for_project(settings.gcp_project_id)
    rows_table = (
        f"{settings.gcp_project_id}.{settings.bq_dataset_raw}.{settings.bq_rows_table}"
    )
    column_index_table = (
        f"{settings.gcp_project_id}.{settings.bq_dataset_raw}."
        f"{settings.bq_column_index_table}"
    )
    result = column_index_mod.refresh_column_index(
        bq=bq,
        rows_table=rows_table,
        column_index_table=column_index_table,
        doc_ids_cap=settings.column_index_doc_ids_cap,
        log=log,
        run_id=settings.run_id,
    )
    typer.echo(json.dumps(asdict(result), indent=2, default=str))


def _parse_since(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"--since must be ISO 8601, got {value!r}: {exc}") from exc
    # Naive inputs (e.g. "2026-06-01" or "2026-06-01T00:00:00") would
    # raise TypeError when compared against tz-aware `ingested_at`.
    # Anchor to UTC to match how the runlog rows are stored.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
