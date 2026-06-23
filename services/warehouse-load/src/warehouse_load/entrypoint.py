"""Typer CLI for the warehouse loader.

```
uv run warehouse-load documents [--dry-run] [--since ISO] [--limit-orgs CODE ...]
```

Thin shim: builds `Settings` from env, overrides with flags, calls
`core.runner.run_documents_load`, prints the summary as JSON. No
business logic here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from warehouse_load.clients.bq import RealBqClient
from warehouse_load.clients.gcs import RealGcsClient
from warehouse_load.config.settings import Settings
from warehouse_load.core.runner import RunRequest, run_documents_load
from warehouse_load.providers.logging import configure_logging, get_logger

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

    request = RunRequest(
        local_dir=local_dir,
        gcs_prefix=gcs_prefix,
        since=parsed_since,
        dry_run=dry_run,
        limit_orgs=tuple(limit_orgs or []),
    )

    log.info(
        "cli_invoked",
        dry_run=dry_run,
        runlog_local_dir=str(local_dir) if local_dir else None,
        runlog_gcs_prefix=gcs_prefix,
        since=parsed_since.isoformat() if parsed_since else None,
        limit_orgs=request.limit_orgs,
        gcp_project=settings.gcp_project_id,
    )

    bq = None if dry_run else RealBqClient.for_project(settings.gcp_project_id)
    gcs = RealGcsClient.for_project(settings.gcp_project_id) if gcs_prefix else None

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
