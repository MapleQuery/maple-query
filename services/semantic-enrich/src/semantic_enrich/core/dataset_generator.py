"""`datasets-generate` orchestrator.

GPU-box-side. Reads `stage/<run_id>/inputs/*.jsonl`, generates one
`DatasetCard` per package via `generate_json`, writes
`stage/<run_id>/datasets/*.jsonl` with `embedding=None`. Never touches
BQ.

The generation + tokenizer callables are injected so tests can replace
them with deterministic fakes (cleaner than monkeypatching torch).
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic
import structlog

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import dataset_prompt, stage_io
from semantic_enrich.core import generate as gen_mod
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    Counters,
    DatasetCard,
    DatasetsGenerateRunSummary,
    GenerationModel,
    PackageInputs,
    StagedDatasetCard,
)

GenerateJsonFn = Callable[..., dict[str, Any]]
LoadGenerationModelFn = Callable[..., GenerationModel]


@dataclass(frozen=True)
class GenerateRequest:
    run_id: str
    dry_run: bool


def _load_models_lock(staging_dir: Path) -> dict[str, Any] | None:
    """Locate MODELS.lock by walking up from the staging dir.

    The lock lives at `services/semantic-enrich/MODELS.lock`. The
    staging dir is `services/semantic-enrich/stage/`, so one parent
    walk reaches it; broader walks fall back gracefully when invoked
    from a non-standard layout.
    """
    for parent in [staging_dir, *staging_dir.parents]:
        candidate = parent / "MODELS.lock"
        if candidate.is_file():
            try:
                payload: dict[str, Any] = json.loads(candidate.read_text())
            except json.JSONDecodeError:
                return None
            return payload
    return None


def preflight(*, settings: Settings, request: GenerateRequest) -> int:
    """Returns the number of input rows under
    `stage/<run_id>/inputs/`. Raises if the dir is missing or empty."""
    inputs_dir = stage_io.stage_path(
        staging_dir=settings.staging_dir,
        run_id=request.run_id,
        artifact="inputs",
    )
    log = get_logger("semantic_enrich.dataset_generator")
    if not inputs_dir.is_dir():
        log.error("inputs_dir_missing", run_id=request.run_id, path=str(inputs_dir))
        raise RuntimeError(
            f"inputs_dir_missing: {inputs_dir} — run datasets-extract first "
            "(and rsync the stage dir to this box)."
        )
    files = list(inputs_dir.glob("*.jsonl"))
    if not files:
        log.error("inputs_dir_missing", run_id=request.run_id, path=str(inputs_dir))
        raise RuntimeError(
            f"inputs_dir_missing: {inputs_dir} contains no .jsonl files."
        )
    row_count = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            row_count += sum(1 for line in f if line.strip())
    return row_count


def run_generate(
    *,
    request: GenerateRequest,
    settings: Settings,
    load_generation_model: LoadGenerationModelFn = gen_mod.load_generation_model,
    generate_json: GenerateJsonFn = gen_mod.generate_json,
    logger: structlog.BoundLogger | None = None,
) -> DatasetsGenerateRunSummary:
    """Main entry. Reads inputs, generates cards, writes JSONL."""
    log = logger or get_logger("semantic_enrich.dataset_generator")
    started = time.monotonic()

    input_row_count = preflight(settings=settings, request=request)
    inputs_dir = stage_io.stage_path(
        staging_dir=settings.staging_dir,
        run_id=request.run_id,
        artifact="inputs",
    )
    input_files = sorted(inputs_dir.glob("*.jsonl"))

    log.info(
        "datasets_generate_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        input_file_count=len(input_files),
        input_row_count=input_row_count,
    )

    staged_ids = stage_io.read_staged_package_ids(
        run_id=request.run_id,
        artifact="datasets",
        staging_dir=settings.staging_dir,
    )

    models_lock = _load_models_lock(settings.staging_dir)
    gen_commit: str | None = None
    if models_lock is not None:
        try:
            gen_commit = models_lock["generation"]["commit"]
        except (KeyError, TypeError):
            gen_commit = None

    # Model load — skipped in dry-run (no torch needed for placeholders).
    model: GenerationModel | None = None
    tokenizer: Any = None
    if not request.dry_run:
        try:
            model = load_generation_model(
                settings.generation_model,
                dtype=settings.generation_dtype,
                device=settings.device,
                cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
            )
            tokenizer = gen_mod.get_tokenizer(model)
            log.info(
                "generation_model_loaded",
                repo=settings.generation_model,
                dtype=settings.generation_dtype,
                device=settings.device,
                vram_mb=_vram_allocated_mb(),
            )
        except Exception as exc:
            log.error("generation_model_load_failed", error=str(exc), exc_info=True)
            raise

    def _on_flush(path: Path, seq: int, row_count: int) -> None:
        log.info(
            "datasets_flush_written",
            run_id=request.run_id,
            path=str(path),
            flush_seq=seq,
            row_count=row_count,
        )

    writer = stage_io.StageWriter(
        run_id=request.run_id,
        artifact="datasets",
        staging_dir=settings.staging_dir,
        flush_every=settings.flush_every_n_packages,
        on_flush=_on_flush,
    )

    counters = Counters()

    try:
        for _, _, pkg in stage_io.iter_staged_rows(
            run_id=request.run_id,
            artifact="inputs",
            staging_dir=settings.staging_dir,
            row_type=PackageInputs,
        ):
            plog = log.bind(package_id=pkg.package_id)

            if pkg.package_id in staged_ids:
                plog.info("package_skipped_already_staged")
                counters.skipped += 1
                continue

            plog.info(
                "package_inputs_built",
                resource_count=len(pkg.resources),
                column_count=len(pkg.column_names),
                representative_document_id=pkg.representative_document_id,
                sample_row_count=len(pkg.sample_rows),
            )

            user_msg = dataset_prompt.render_user_message(pkg)

            if request.dry_run:
                plog.info(
                    "would_have_generated",
                    prompt_token_estimate=dataset_prompt.estimate_tokens(user_msg),
                )
                writer.append(_dry_run_placeholder(pkg=pkg, settings=settings,
                                                  gen_commit=gen_commit))
                counters.generated += 1
                continue

            assert model is not None and tokenizer is not None

            rendered = _build_chat_prompt(tokenizer=tokenizer, user_msg=user_msg)
            t0 = time.monotonic()
            try:
                # outlines 1.x's `python_types_to_terms` rejects raw
                # JSON-schema dicts; only pydantic classes (or its own
                # DSL types) are accepted. `DatasetCard` carries the
                # same constraints as `DATASET_CARD_GUIDED_JSON`.
                raw = generate_json(
                    rendered,
                    DatasetCard,
                    model=model,
                    max_tokens=settings.generation_max_tokens,
                    temperature=settings.generation_temperature,
                )
            except Exception as exc:
                plog.exception("package_generation_failed", error=str(exc))
                counters.failed += 1
                continue

            try:
                card = DatasetCard.model_validate(raw)
            except pydantic.ValidationError as exc:
                plog.error(
                    "package_generation_failed",
                    error=f"validation_error: {exc.errors()}",
                )
                counters.failed += 1
                continue

            staged = StagedDatasetCard(
                package_id=card.package_id,
                summary=card.summary,
                grain=card.grain,
                measures=card.measures,
                dimensions=card.dimensions,
                date_range_start=card.date_range_start,
                date_range_end=card.date_range_end,
                embedding=None,
                generated_at=datetime.now(UTC),
                generation_model=settings.generation_model,
                generation_model_commit=gen_commit,
                generation_run_id=request.run_id,
                representative_document_id=pkg.representative_document_id,
                dry_run=False,
            )
            writer.append(staged)
            counters.generated += 1
            plog.info(
                "package_generation_done",
                outcome="generated",
                duration_ms=int((time.monotonic() - t0) * 1000),
                prompt_chars=len(rendered),
                response_chars=len(card.summary),
            )
    finally:
        writer.close()
        if model is not None:
            del model
            _drop_cuda_cache()

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = DatasetsGenerateRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        input_row_count=input_row_count,
        packages_generated=counters.generated,
        packages_skipped_already_staged=counters.skipped,
        packages_failed=counters.failed,
        flush_files_written=writer.files_written,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "datasets_generate_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


def _build_chat_prompt(*, tokenizer: Any, user_msg: str) -> str:
    """Wrap the system + user messages through Qwen's chat template."""
    messages = [
        {"role": "system", "content": dataset_prompt.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    rendered: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def _dry_run_placeholder(
    *, pkg: PackageInputs, settings: Settings, gen_commit: str | None
) -> StagedDatasetCard:
    """Build a placeholder StagedDatasetCard for dry-run.

    `summary` is deliberately under 50 chars so the load pass's
    re-validation (per §10.3) rejects it loudly if a dry-run JSONL is
    ever accidentally fed to `datasets-load`.
    """
    return StagedDatasetCard(
        package_id=pkg.package_id,
        summary="DRY_RUN_PLACEHOLDER",
        grain=None,
        measures=[],
        dimensions=[],
        date_range_start=None,
        date_range_end=None,
        embedding=None,
        generated_at=datetime.now(UTC),
        generation_model=settings.generation_model,
        generation_model_commit=gen_commit,
        generation_run_id=settings.run_id,
        representative_document_id=pkg.representative_document_id,
        dry_run=True,
    )


def _vram_allocated_mb() -> int | None:
    """Returns MiB allocated on the active CUDA device, or None on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _drop_cuda_cache() -> None:
    """Free CUDA memory after the model reference is dropped."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_invariant(
    summary: DatasetsGenerateRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.packages_generated
        + summary.packages_skipped_already_staged
        + summary.packages_failed
    )
    if total != summary.input_row_count:
        log.error(
            "run_invariant_violated",
            subcommand="datasets-generate",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"datasets-generate rows accounted-for mismatch: {summary}"
        )
