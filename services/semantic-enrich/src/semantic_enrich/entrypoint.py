"""Typer CLI.

Laptop (uv) ──────────────────────────────────────────────────────
  uv run semantic-enrich datasets-extract  [--run-id ID]
                                           [--limit-packages N]
                                           [--limit-package-ids ID...]
                                           [--limit-orgs CODE...]
                                           [--dry-run] [--staging-dir PATH]
  uv run semantic-enrich datasets-embed    [--run-id ID] [--batch-size N]
                                           [--dry-run] [--staging-dir PATH]
  uv run semantic-enrich datasets-load     [--run-id ID] [--dry-run]
                                           [--staging-dir PATH]
  uv run semantic-enrich datasets-reembed  [--run-id ID] [--batch-size N]
                                           [--dry-run]
  uv run semantic-enrich columns-reembed   [--run-id ID] [--batch-size N]
                                           [--dry-run]

GPU box (conda env active — no uv) ──────────────────────────────
  semantic-enrich smoke-test        [--write-lock]
  semantic-enrich datasets-generate [--run-id ID] [--dry-run]
                                    [--staging-dir PATH]

Post-4.7, `datasets-embed` and `columns-embed` call OpenAI
text-embedding-3-small — no GPU needed. They can run on either box.

Thin shim: builds `Settings` from env, overrides with flags, builds a
`*Request`, calls `core.run_*(...)`, prints the summary as JSON. No
business logic.

Exit codes (datasets-* / columns-* subcommands):
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
from semantic_enrich.clients.openai import OpenAIClient, RealOpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_generator import (
    ColumnsGenerateRequest,
)
from semantic_enrich.core.column_generator import (
    run_generate as run_columns_generate,
)
from semantic_enrich.core.column_inputs import (
    ColumnsExtractRequest,
)
from semantic_enrich.core.column_inputs import (
    run_extract as run_columns_extract,
)
from semantic_enrich.core.columns_load import (
    ColumnsLoadRequest,
)
from semantic_enrich.core.columns_load import (
    run_load as run_columns_load,
)
from semantic_enrich.core.dataset_extract import ExtractRequest, run_extract
from semantic_enrich.core.dataset_generator import GenerateRequest, run_generate
from semantic_enrich.core.datasets_load import LoadRequest, run_load
from semantic_enrich.core.embedding_pass import (
    ColumnsEmbedRequest,
    EmbedRequest,
    run_columns_embed,
    run_embed,
)
from semantic_enrich.core.eval_runner import (
    EvalRequest,
    PreconditionError,
    run_eval,
)
from semantic_enrich.core.reembed import (
    ColumnsReembedRequest,
    DatasetsReembedRequest,
    run_columns_reembed,
    run_datasets_reembed,
)
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
        help="Override WHENRICH_OPENAI_EMBEDDING_BATCH_SIZE.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips embed calls; emits would_have_embedded events.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Augment stage/<run_id>/datasets/*.jsonl with OpenAI embeddings."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    client = _build_openai_client(settings=settings, log=log, dry_run=dry_run)
    request = EmbedRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        batch_size=batch_size,
    )
    _dispatch(
        lambda: run_embed(
            request=request, settings=settings, openai_client=client
        )
    )


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
# columns-extract  (laptop)
# ──────────────────────────────────────────────────────────────────


@app.command("columns-extract")
def columns_extract(
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
        help="Emits would_have_extracted_columns events alongside the JSONL.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Laptop-side. Read raw.documents + raw.rows + semantic.datasets;
    write stage/<run_id>/column_inputs/*.jsonl."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="columns-extract")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    request = ColumnsExtractRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        limit_packages=limit_packages,
        limit_package_ids=limit_package_ids,
        limit_orgs=limit_orgs,
    )
    _dispatch(lambda: run_columns_extract(request=request, settings=settings, bq=bq))


# ──────────────────────────────────────────────────────────────────
# columns-generate  (GPU box)
# ──────────────────────────────────────────────────────────────────


@app.command("columns-generate")
def columns_generate(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "columns-extract run.",
    ),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Override WHENRICH_COLUMN_CHUNK_SIZE."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips model loads; writes placeholder JSONL.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """GPU-side. Read stage/<run_id>/column_inputs/*.jsonl; chunk; generate;
    write stage/<run_id>/columns/*.jsonl."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    request = ColumnsGenerateRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        chunk_size=chunk_size,
    )
    _dispatch(lambda: run_columns_generate(request=request, settings=settings))


# ──────────────────────────────────────────────────────────────────
# columns-embed  (GPU box)
# ──────────────────────────────────────────────────────────────────


@app.command("columns-embed")
def columns_embed(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "columns-generate run.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override WHENRICH_OPENAI_EMBEDDING_BATCH_SIZE.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips embed calls; emits would_have_embedded_columns events.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Augment stage/<run_id>/columns/*.jsonl with OpenAI embeddings."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    client = _build_openai_client(settings=settings, log=log, dry_run=dry_run)
    request = ColumnsEmbedRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        batch_size=batch_size,
    )
    _dispatch(
        lambda: run_columns_embed(
            request=request, settings=settings, openai_client=client
        )
    )


# ──────────────────────────────────────────────────────────────────
# datasets-reembed / columns-reembed  (4.7 — laptop)
# ──────────────────────────────────────────────────────────────────


@app.command("datasets-reembed")
def datasets_reembed(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Names the staging table.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override WHENRICH_OPENAI_EMBEDDING_BATCH_SIZE.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips OpenAI calls and the MERGE; emits one "
             "would_have_reembedded event per row.",
    ),
) -> None:
    """Overwrite `semantic.datasets.embedding` with OpenAI vectors."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=None)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="datasets-reembed")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    # Reembed preflights against OpenAI even in --dry-run (§8+§9) so
    # we always require a real key here.
    client = _build_openai_client(settings=settings, log=log, dry_run=False)
    request = DatasetsReembedRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        batch_size=batch_size,
    )
    _dispatch(
        lambda: run_datasets_reembed(
            request=request,
            settings=settings,
            bq=bq,
            openai_client=client,
        )
    )


@app.command("columns-reembed")
def columns_reembed(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Names the staging table.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override WHENRICH_OPENAI_EMBEDDING_BATCH_SIZE.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips OpenAI calls and the MERGE; emits one "
             "would_have_reembedded event per row.",
    ),
) -> None:
    """Overwrite `semantic.columns.embedding` with OpenAI vectors."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=None)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="columns-reembed")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    # Reembed preflights against OpenAI even in --dry-run (§8+§9) so
    # we always require a real key here.
    client = _build_openai_client(settings=settings, log=log, dry_run=False)
    request = ColumnsReembedRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
        batch_size=batch_size,
    )
    _dispatch(
        lambda: run_columns_reembed(
            request=request,
            settings=settings,
            bq=bq,
            openai_client=client,
        )
    )


# ──────────────────────────────────────────────────────────────────
# columns-load  (laptop)
# ──────────────────────────────────────────────────────────────────


@app.command("columns-load")
def columns_load(
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="REQUIRED unless WHENRICH_RUN_ID is set. Must match a prior "
             "columns-embed run.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Skips the MERGE; emits would_have_loaded per row.",
    ),
    staging_dir: Path | None = typer.Option(
        None, "--staging-dir", help="Override WHENRICH_STAGING_DIR."
    ),
) -> None:
    """Laptop-side. Coalesce + MERGE into semantic.columns."""
    configure_logging()
    settings = _build_settings(run_id=run_id, staging_dir=staging_dir)
    log = get_logger("semantic_enrich.entrypoint")
    if not settings.gcp_project_id:
        log.error("missing_project_id", subcommand="columns-load")
        raise typer.Exit(3)
    try:
        bq = RealBqClient.for_project(settings.gcp_project_id)
    except Exception as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise typer.Exit(3) from exc
    request = ColumnsLoadRequest(
        run_id=settings.run_id,
        dry_run=dry_run or settings.dry_run,
    )
    _dispatch(lambda: run_columns_load(request=request, settings=settings, bq=bq))


# ──────────────────────────────────────────────────────────────────
# eval  (laptop, 4.6 retrieval-validation harness)
# ──────────────────────────────────────────────────────────────────


@app.command("eval")
def eval_run(
    questions: Path | None = typer.Option(
        None,
        "--questions",
        help="Path to questions YAML. Default services/semantic-enrich/eval/questions.yaml.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Run only the first N questions."
    ),
    question_ids: list[str] | None = typer.Option(
        None,
        "--question-ids",
        help="Repeatable. Combined with --limit, --limit wins.",
    ),
    k_packages: int | None = typer.Option(
        None,
        "--k-packages",
        help="Override top-k for packages. Env WHENRICH_EVAL_K_PACKAGES.",
    ),
    k_columns: int | None = typer.Option(
        None,
        "--k-columns",
        help="Override top-k for columns. Env WHENRICH_EVAL_K_COLUMNS.",
    ),
    max_bytes_billed: int | None = typer.Option(
        None,
        "--max-bytes-billed",
        help=(
            "Cost cap. Default 50 GB. Env WHENRICH_EVAL_MAX_BYTES_BILLED."
        ),
    ),
    no_execute: bool = typer.Option(
        False,
        "--no-execute/--execute",
        help="Stop after SQL gen + dry-run.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Report path override. Default eval_reports_dir/<run_id>.json.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Harness self-test — skips OpenAI + BQ, exercises fixture / template / report writers.",
    ),
) -> None:
    """Run the 4.6 retrieval-validation harness against the fixture."""
    configure_logging()
    log = get_logger("semantic_enrich.entrypoint")

    overrides: dict[str, Any] = {}
    if questions is not None:
        overrides["eval_questions_path"] = questions
    if k_packages is not None:
        overrides["eval_k_packages"] = k_packages
    if k_columns is not None:
        overrides["eval_k_columns"] = k_columns
    settings = Settings()
    if overrides:
        settings = settings.model_copy(update=overrides)

    if not dry_run:
        if not settings.gcp_project_id:
            log.error("missing_project_id", subcommand="eval")
            raise typer.Exit(3)
        try:
            bq: Any = RealBqClient.for_project(settings.gcp_project_id)
        except Exception as exc:
            log.error("bq_auth_failed", error=str(exc))
            raise typer.Exit(3) from exc
        openai_client = _build_openai_client(
            settings=settings, log=log, dry_run=False
        )
    else:
        # Dry-run touches neither client; the runner short-circuits
        # before either is dereferenced. Pass stubs at the Protocol
        # boundary so the type stays honest.
        bq = _StubBqClient()
        openai_client = _StubOpenAIClient()

    request = EvalRequest(
        run_id=settings.eval_run_id,
        dry_run=dry_run,
        no_execute=no_execute,
        limit=limit,
        question_ids=tuple(question_ids) if question_ids else None,
        max_bytes_billed_override=max_bytes_billed,
        output_override=output,
    )

    try:
        summary = run_eval(
            request=request,
            settings=settings,
            bq=bq,
            openai_client=openai_client,
        )
    except PreconditionError as exc:
        log.error(
            "preconditions_failed",
            run_id=request.run_id,
            reason=str(exc),
        )
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "preconditions_failed": str(exc),
                    "run_id": request.run_id,
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(3) from exc
    except RuntimeError as exc:
        log.error("eval_run_failed", error=str(exc))
        typer.echo(
            json.dumps({"ok": False, "error": str(exc)}, indent=2), err=True
        )
        raise typer.Exit(2) from exc
    except Exception as exc:
        log.error("eval_internal_error", error=str(exc), exc_info=True)
        typer.echo(
            json.dumps({"ok": False, "internal_error": str(exc)}, indent=2),
            err=True,
        )
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(asdict(summary), indent=2, default=str))


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


def _build_openai_client(
    *, settings: Settings, log: Any, dry_run: bool
) -> OpenAIClient:
    """Build the shared OpenAI client from Settings.

    Dry-run still requires a key for `datasets-embed` / `columns-embed`
    only if the runner will call it; the runner short-circuits on
    dry-run before touching the client, so we return a stub that
    raises if called. This keeps the "no key + dry-run" path usable
    for local previewing.
    """
    if settings.openai_api_key is None:
        if dry_run:
            return _StubOpenAIClient()
        log.error(
            "missing_openai_api_key",
            hint="set WHENRICH_OPENAI_API_KEY or OPENAI_API_KEY",
        )
        raise typer.Exit(3)
    return RealOpenAIClient.for_settings(
        api_key=settings.openai_api_key.get_secret_value(),
        embedding_model=settings.openai_embedding_model,
        request_timeout_s=settings.openai_request_timeout_s,
        max_retries=settings.openai_max_retries,
    )


class _StubOpenAIClient:
    """Placeholder used for `--dry-run` when no OpenAI key is present.

    The stage embed runners skip the client in dry-run; the reembed
    runners preflight-ping it even in dry-run per the reembed PRD, so
    this stub only survives the stage-embed dry-run path and the 4.6
    `eval --dry-run` harness self-test.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "openai client not configured; set WHENRICH_OPENAI_API_KEY. "
            "This stub is only viable for --dry-run previews."
        )

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        raise RuntimeError(
            "openai client not configured; set WHENRICH_OPENAI_API_KEY. "
            "This stub is only viable for --dry-run previews."
        )


class _StubBqClient:
    """Placeholder used for `eval --dry-run`. The eval runner short-
    circuits before touching BQ in dry-run; any dereference here is a
    logic bug in the runner and blows up loudly."""

    def _refuse(self, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError(
            "bq client not configured; this stub only survives the eval "
            "--dry-run harness self-test."
        )

    execute = _refuse
    execute_with_params = _refuse
    query_rows = _refuse
    append_jsonl_file = _refuse
    create_staging_table = _refuse
    delete_table = _refuse
    dry_run_bytes = _refuse
    run_bounded_query = _refuse


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
