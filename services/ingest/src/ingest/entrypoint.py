"""Typer CLI for local backfill runs.

`uv run ingest -s <subject> -f csv -f xlsx --limit-orgs fin --dry-run`

Writes bytes to GCS and one per-resource record to
`runlog/<run_id>.jsonl` (override the directory with `INGEST_RUNLOG_DIR`).
A follow-up task loads the JSONL into BigQuery's `raw.documents` table.

GCP clients use application-default credentials (run `gcloud auth
application-default login` once).
"""
from __future__ import annotations

from datetime import datetime

import typer
from google.cloud import storage as gcs_sdk

from ingest.clients.ckan import CkanClient
from ingest.clients.gcs import GcsClient
from ingest.clients.http import HttpClient
from ingest.config.settings import Settings
from ingest.config.sources import load_sources
from ingest.core.pipeline import RunRequest, run
from ingest.core.runlog import RunLogWriter, default_runlog_path
from ingest.providers.logging import configure_logging, get_logger

app = typer.Typer(name="ingest", help="MapleQuery local-backfill ingestion CLI.")

VERSION = "0.1.0"


@app.command()
def main(
    subject: str = typer.Option(
        ..., "-s", "--subject",
        help="CKAN subject filter, e.g. government_and_politics.",
    ),
    formats: list[str] | None = typer.Option(
        None, "-f", "--format",
        help="Repeatable; e.g. -f csv -f xlsx. Omit to ingest all formats.",
    ),
    limit_orgs: list[str] | None = typer.Option(
        None, "--limit-orgs",
        help="Repeatable; restrict to these org codes.",
    ),
    since: str | None = typer.Option(
        None, "--since",
        help="ISO 8601 cursor for CKAN metadata_modified filter.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skip GCS uploads and run-log writes; emit would_have_* log events.",
    ),
) -> None:
    """Run one ingest pass."""
    configure_logging()
    log = get_logger("ingest.entrypoint")

    settings = Settings()  # type: ignore[call-arg]
    sources = load_sources(settings.sources_config_path)

    parsed_since = datetime.fromisoformat(since) if since else None
    request = RunRequest(
        subject=subject,
        formats=tuple(f.lower() for f in (formats or [])),
        limit_orgs=tuple(limit_orgs or []),
        dry_run=dry_run,
        since=parsed_since,
    )

    runlog_path = default_runlog_path(
        run_id=settings.run_id, override_dir=settings.runlog_dir
    )

    log.info(
        "cli_invoked",
        subject=subject,
        formats=request.formats,
        limit_orgs=request.limit_orgs,
        dry_run=dry_run,
        since=parsed_since.isoformat() if parsed_since else None,
        gcp_project=settings.gcp_project_id,
        runlog_path=str(runlog_path),
    )

    user_agent = f"maplequery-ingest/{VERSION}"

    with HttpClient(
        user_agent=user_agent,
        request_timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    ) as http, RunLogWriter(path=runlog_path) as runlog:
        ckans = {
            src.source: CkanClient(
                http=http,
                api_base=str(src.api_base),
                inter_request_delay_seconds=settings.inter_request_delay_seconds,
            )
            for src in sources
        }
        gcs = GcsClient(
            client=gcs_sdk.Client(project=settings.gcp_project_id),
            bucket=settings.gcs_bucket,
        )

        summary = run(
            settings=settings,
            sources=sources,
            request=request,
            ckans=ckans,
            http=http,
            gcs=gcs,
            runlog=runlog,
        )

    typer.echo(
        f"Done in {summary.duration_ms / 1000:.1f}s — "
        f"{summary.success} success, "
        f"{summary.quarantined} quarantined, "
        f"{summary.failed} failed, "
        f"{summary.skipped_by_gcs_dedup} gcs-dedup-skipped, "
        f"{summary.skipped_by_pairing} pairing-skipped. "
        f"Run log: {runlog_path}"
    )


if __name__ == "__main__":
    app()
