"""Typer CLI.

```
uv run semantic-enrich smoke-test [--write-lock]
```

One subcommand. Replaces the abandoned 4.3's HTTP wait loop +
`validate_round_trip.py` + 15-step runbook.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.smoke import run_smoke_test, write_models_lock
from semantic_enrich.providers.logging import configure_logging, get_logger

app = typer.Typer(name="semantic-enrich", help="MapleQuery semantic enrichment runtime CLI.")


@app.callback()
def _root() -> None:
    """Keeps `smoke-test` as an explicit subcommand so future siblings
    (e.g. `benchmark`) can land without breaking the CLI surface."""


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
    """Round-trip both models; assert shape; optionally lock versions.

    Exit codes:
      - 0  both passed.
      - 2  precondition failed (model load, schema violation, dim/norm drift).
      - 1  unexpected internal error.
    """
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
