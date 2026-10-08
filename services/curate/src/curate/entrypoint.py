"""curate CLI: build the curated tables.

curate people                 # people, person_names, person_terms
curate contributions          # person_contributions (after people)
curate people --dry-run --out /tmp/curate   # JSONL, no BigQuery
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from curate.clients.bq import RealBqClient
from curate.clients.gcs import RealGcsClient
from curate.clients.openparliament import RealOpenParliamentClient
from curate.config.settings import Settings
from curate.core.runner import Deps, Plan, run_contributions, run_people
from curate.providers.logging import configure_logging, get_logger

app = typer.Typer(name="curate", help="MapleQuery Normalize (M3): build curated.* tables.")


def _deps(settings: Settings, *, dry_run: bool, need_op: bool, need_gcs: bool) -> Deps:
    log = get_logger("curate")
    if not dry_run and not settings.gcp_project_id:
        raise typer.BadParameter("CURATE_GCP_PROJECT_ID (or GCP_PROJECT_ID) is required unless --dry-run")
    return Deps(
        settings=settings,
        log=log,
        op=RealOpenParliamentClient(
            base_url=settings.openparliament_base,
            user_agent=settings.openparliament_user_agent,
            timeout_s=settings.request_timeout_seconds,
            detail_rps=settings.detail_requests_per_second,
        )
        if need_op
        else None,
        bq=None if dry_run else RealBqClient(project_id=settings.gcp_project_id),
        gcs=RealGcsClient(project_id=settings.gcp_project_id, bucket=settings.gcs_bucket)
        if need_gcs
        else None,
    )


@app.command()
def people(
    dry_run: bool = typer.Option(False, "--dry-run", help="Write JSONL to --out instead of BigQuery."),
    out: Path | None = typer.Option(None, "--out", help="Directory for --dry-run output."),
    allow_shrink: bool = typer.Option(False, "--allow-shrink", help="Override the >50% shrink guardrail."),
    max_detail: int | None = typer.Option(None, "--max-detail", help="Cap per-person detail fetches."),
) -> None:
    """Build people, person_names and person_terms from openparliament.ca."""
    configure_logging()
    settings = Settings()
    deps = _deps(settings, dry_run=dry_run, need_op=True, need_gcs=False)
    summary = run_people(
        deps, Plan(dry_run=dry_run, out_dir=out, allow_shrink=allow_shrink), max_detail=max_detail
    )
    typer.echo(json.dumps({"run_id": settings.run_id, **summary}))


@app.command()
def contributions(
    dry_run: bool = typer.Option(False, "--dry-run", help="Write JSONL to --out instead of BigQuery."),
    out: Path | None = typer.Option(None, "--out", help="Directory for --dry-run output."),
    source_file: Path | None = typer.Option(None, "--source-file", help="Local archive instead of GCS."),
    people_dir: Path | None = typer.Option(
        None, "--people-from", help="Read people from a `curate people --dry-run` directory."
    ),
    allow_shrink: bool = typer.Option(False, "--allow-shrink", help="Override the >50% shrink guardrail."),
) -> None:
    """Link Elections Canada contributions to people (person_contributions)."""
    configure_logging()
    settings = Settings()
    deps = _deps(settings, dry_run=dry_run, need_op=False, need_gcs=source_file is None)
    summary = run_contributions(
        deps,
        Plan(dry_run=dry_run, out_dir=out, allow_shrink=allow_shrink),
        source_file=source_file,
        people_dir=people_dir,
    )
    typer.echo(json.dumps({"run_id": settings.run_id, **summary}))
