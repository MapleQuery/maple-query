"""Typer CLI.

Laptop (uv) ──────────────────────────────────────────────────────
  uv run semantic-enrich datasets-extract [--run-id ID]
                                          [--limit-packages N]
                                          [--limit-package-ids ID...]
                                          [--limit-orgs CODE...]
                                          [--dry-run] [--staging-dir PATH]
  uv run semantic-enrich datasets-load    [--run-id ID] [--dry-run]
                                          [--staging-dir PATH]

GPU box (conda env active — no uv) ──────────────────────────────
  semantic-enrich smoke-test       [--write-lock]
  semantic-enrich datasets-generate [--run-id ID] [--dry-run]
                                    [--staging-dir PATH]
  semantic-enrich datasets-embed    [--run-id ID] [--batch-size N]
                                    [--dry-run] [--staging-dir PATH]

Thin shim: builds `Settings` from env, overrides with flags, builds a
`*Request`, calls `core.run_*(...)`, prints the summary as JSON. No
business logic.

Exit codes (datasets-* subcommands):
  - 0  success.
  - 2  invariant violated mid-run (a `RuntimeError` from the runner).
  - 3  precondition failed (bad config, BQ auth, missing inputs dir,
       model load error, dry-run-placeholder fed to load).
  - 1  unexpected internal error.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from semantic_enrich.clients.bq import RealBqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.dataset_extract import ExtractRequest, run_extract
from semantic_enrich.core.dataset_generator import GenerateRequest, run_generate
from semantic_enrich.core.datasets_load import LoadRequest, run_load
from semantic_enrich.core.embedding_pass import EmbedRequest, run_embed
from semantic_enrich.core.smoke import run_smoke_test, write_models_lock
from semantic_enrich.providers.logging import configure_logging, get_logger

app = typer.Typer(name="semantic-enrich", help="MapleQuery semantic enrichment runtime CLI.")


@app.callback()
def _root() -> None:
    """Keeps each subcommand explicit so the CLI surface is stable."""


# ──────────────────────────────────────────────────────────────────
# smoke-test (4.3 — unchanged)
# ──────────────────────────────────────────────────────────────────


@app.command("smoke-test")
def smoke_test(
    write_lock: bool = typer.Option(
        False,
        "--write-lock/--no-write-lock",
        help=(
            "On success, write MODELS.lock with resolved package "
            "versions and HF commit SHAs."
        ),
    ),
    lock_path: Path = typer.Option(
        Path("MODELS.lock"),
        "--lock-path",
        help="Path to MODELS.lock. Default writes to the CWD.",
    ),
) -> None:
    """Round-trip both models; assert shape; optionally lock versions."""
    configure_logging()
    log = get_logger("semantic_enrich.entrypoint")

    settings = Settings()

    log.info(
        "smoke_test_invoked",
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        device=settings.device,
        hf_cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
        write_lock=write_lock,
    )

    try:
        result = run_smoke_test(settings=settings)
    except Exception as exc:
        log.error("smoke_internal_error", error=str(exc), exc_info=True)
        typer.echo(
            json.dumps({"ok": False, "internal_error": str(exc)}, indent=2),
            err=True,
        )
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(asdict(result), indent=2, default=str))

    if not result.ok:
        log.error(
            "smoke_precondition_failed",
            reason=result.precondition_failure,
            duration_ms=result.duration_ms,
        )
        raise typer.Exit(2)

    log.info(
        "smoke_test_passed",
        duration_ms=result.duration_ms,
        embedding_dim=result.embedding_dim,
        embedding_norm=result.embedding_norm,
    )

    if write_lock:
        payload = write_models_lock(
            lock_path,
            generation_repo=settings.generation_model,
            embedding_repo=settings.embedding_model,
        )
        log.info("models_lock_written", path=str(lock_path), payload=payload)


# ──────────────────────────────────────────────────────────────────
# datasets-extract  (laptop)
# ──────────────────────────────────────────────────────────────────


@app.command("datasets-extract")
def datasets_extract(
    run_id: str | None = typer.Option(
        None, "--run-id", help="Override WHENRICH_RUN_ID. Reuse to resume."
    ),
    limit_packages: int | None = typer.Option(
        None, "--limit-packages", help="Cap candidates to N for smoke runs."
    ),
    limit_package_ids: list[str] | None = typer.Option(
        None,
        "--limit-package-ids",
        help="Repeatable; restrict to these package_ids.",
    ),
    limit_orgs: list[str] | None = typer.Option(
        None, "--limit-orgs", help="Repeatable; restrict to these org codes."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Emits would_have_extracted events alongside the JSONL.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Laptop-side. Read raw.documents + raw.rows; write
    stage/<run_id>/inputs/*.jsonl."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="datasets-extract")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    request = ExtractRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        limit_packages=limit_packages,
        limit_package_ids=limit_package_ids,
        limit_orgs=limit_orgs,
    )
    _dispatch(lambda: run_extract(request=request, settings=settings, bq=bq))


# ──────────────────────────────────────────────────────────────────
# datasets-generate  (GPU box)
# ──────────────────────────────────────────────────────────────────


@app.command("datasets-generate")
def datasets_generate(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "datasets-extract run.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips model loads; writes placeholder JSONL.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """GPU-side. Read stage/<run_id>/inputs/*.jsonl; write
    stage/<run_id>/datasets/*.jsonl."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    request = GenerateRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
    )
    _dispatch(lambda: run_generate(request=request, settings=settings))


# ──────────────────────────────────────────────────────────────────
# datasets-embed  (GPU box)
# ──────────────────────────────────────────────────────────────────


@app.command("datasets-embed")
def datasets_embed(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "datasets-generate run.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override WHENRICH_EMBEDDING_BATCH_SIZE.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips embed calls; emits would_have_embedded events.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """GPU-side. Augment stage/<run_id>/datasets/*.jsonl with embeddings."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    request = EmbedRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        batch_size=batch_size,
    )
    _dispatch(lambda: run_embed(request=request, settings=settings))


# ──────────────────────────────────────────────────────────────────
# datasets-load  (laptop)
# ──────────────────────────────────────────────────────────────────


@app.command("datasets-load")
def datasets_load(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "datasets-embed run.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips the MERGE; emits would_have_loaded per row.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Laptop-side. Coalesce + MERGE into semantic.datasets."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="datasets-load")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    request = LoadRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
    )
    _dispatch(lambda: run_load(request=request, settings=settings, bq=bq))


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────


def _build_settings(
    *, run_id: str | None, staging_dir: Path | None
) -> Settings:
    """Settings with per-invocation overrides applied.

    pydantic-settings re-reads env on each `Settings()` call; we apply
    the CLI overrides via `model_copy(update=...)` so the same Settings
    is observable end-to-end."""
    settings = Settings()
    overrides: dict[str, Any] = {}
    if run_id is not None:
        overrides["run_id"] = run_id
    if staging_dir is not None:
        overrides["staging_dir"] = staging_dir
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings


def _dispatch(runner: Callable[[], Any]) -> None:
    """Run a subcommand body; translate exceptions into exit codes."""
    log = get_logger("semantic_enrich.entrypoint")
    try:
        summary = runner()
    except RuntimeError as exc:
        # Preconditions (auth, missing inputs, bad project) raise
        # `RuntimeError`; invariant violations raise `RuntimeError`
        # with a `mismatch` substring. Exit 2 for an invariant, 3 for
        # a precondition.
        msg = str(exc)
        if "mismatch" in msg:
            log.error("run_invariant_violated", error=msg)
            raise typer.Exit(2) from exc
        log.error("precondition_failed", error=msg)
        raise typer.Exit(3) from exc
    except Exception as exc:
        log.error("internal_error", error=str(exc), exc_info=True)
        typer.echo(json.dumps({"ok": False, "internal_error": str(exc)}),
                   err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(asdict(summary), indent=2, default=str))
