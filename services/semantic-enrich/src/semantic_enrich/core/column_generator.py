"""`columns-generate` orchestrator.

GPU-box-side. Reads `stage/<run_id>/column_inputs/*.jsonl`, chunks
each package's column list into ≤`column_chunk_size` batches, issues
one `generate_json_list` call per chunk, validates the per-chunk 1:1
column-name mapping invariant, concatenates all chunks for a package,
validates the per-package invariant, and writes one
`StagedColumnRow` per `(package_id, column_name)` into
`stage/<run_id>/columns/*.jsonl`. Never touches BQ.

The generation + tokenizer callables are injected so tests can
substitute deterministic fakes (no torch import needed in CI).
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic
import structlog

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import column_prompt, stage_io
from semantic_enrich.core import generate as gen_mod
from semantic_enrich.core.schemas import COLUMNS_GUIDED_JSON_SCHEMA
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    ColumnChunkInvariantError,
    ColumnInputs,
    ColumnOutput,
    ColumnPackageInvariantError,
    ColumnsGenerateRunSummary,
    Counters,
    GenerationModel,
    StagedColumnRow,
)

GenerateJsonListFn = Callable[..., list[dict[str, Any]]]
LoadGenerationModelFn = Callable[..., GenerationModel]


@dataclass(frozen=True)
class ColumnsGenerateRequest:
    run_id: str
    dry_run: bool
    chunk_size: int | None  # None → settings.column_chunk_size


@dataclass(frozen=True)
class _ColumnChunk:
    package_id: str
    chunk_index: int
    chunk_count: int
    column_names: list[str]
    sample_values: dict[str, list[str]]


def _load_models_lock(staging_dir: Path) -> dict[str, Any] | None:
    """Locate MODELS.lock by walking up from the staging dir. Mirrors
    `dataset_generator._load_models_lock`."""
    for parent in [staging_dir, *staging_dir.parents]:
        candidate = parent / "MODELS.lock"
        if candidate.is_file():
            try:
                payload: dict[str, Any] = json.loads(candidate.read_text())
            except json.JSONDecodeError:
                return None
            return payload
    return None


def preflight(*, settings: Settings, request: ColumnsGenerateRequest) -> int:
    """Returns the number of input rows under
    `stage/<run_id>/column_inputs/`. Raises if missing or empty."""
    inputs_dir = stage_io.stage_path(
        staging_dir=settings.staging_dir,
        run_id=request.run_id,
        artifact="column_inputs",
    )
    log = get_logger("semantic_enrich.column_generator")
    if not inputs_dir.is_dir():
        log.error(
            "column_inputs_dir_missing",
            run_id=request.run_id,
            path=str(inputs_dir),
        )
        raise RuntimeError(
            f"column_inputs_dir_missing: {inputs_dir} — run columns-extract "
            "first (and rsync the stage dir to this box)."
        )
    files = list(inputs_dir.glob("*.jsonl"))
    if not files:
        log.error(
            "column_inputs_dir_missing",
            run_id=request.run_id,
            path=str(inputs_dir),
        )
        raise RuntimeError(
            f"column_inputs_dir_missing: {inputs_dir} contains no .jsonl files."
        )
    row_count = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            row_count += sum(1 for line in f if line.strip())
    return row_count


def chunk_columns(
    *, inputs: ColumnInputs, chunk_size: int
) -> Iterator[_ColumnChunk]:
    """Slice `inputs.column_names` into chunks of ≤`chunk_size`.

    Order is preserved across chunks; concatenation of all chunks'
    `column_names` equals `inputs.column_names`. The §8 invariant
    rests on this.
    """
    names = list(inputs.column_names)
    if not names:
        return
    chunk_count = math.ceil(len(names) / chunk_size)
    for i in range(chunk_count):
        slice_start = i * chunk_size
        slice_end = slice_start + chunk_size
        chunk_names = names[slice_start:slice_end]
        yield _ColumnChunk(
            package_id=inputs.package_id,
            chunk_index=i,
            chunk_count=chunk_count,
            column_names=chunk_names,
            sample_values={
                n: list(inputs.sample_values.get(n, ())) for n in chunk_names
            },
        )


def validate_chunk_output(
    chunk: _ColumnChunk, response: list[dict[str, Any]]
) -> list[ColumnOutput]:
    """Per-chunk 1:1 column-name mapping invariant (§8.1)."""
    if len(response) != len(chunk.column_names):
        raise ColumnChunkInvariantError(
            f"chunk {chunk.chunk_index} of package {chunk.package_id}: "
            f"requested {len(chunk.column_names)} columns, "
            f"model returned {len(response)} entries"
        )
    out: list[ColumnOutput] = []
    for i, (expected, entry) in enumerate(
        zip(chunk.column_names, response, strict=True)
    ):
        actual = entry.get("column_name")
        if actual != expected:
            raise ColumnChunkInvariantError(
                f"chunk {chunk.chunk_index} of package {chunk.package_id}: "
                f"position {i}: expected column_name={expected!r}, "
                f"model returned {actual!r}"
            )
        try:
            out.append(ColumnOutput.model_validate(entry))
        except pydantic.ValidationError as exc:
            raise ColumnChunkInvariantError(
                f"chunk {chunk.chunk_index} of package {chunk.package_id}: "
                f"position {i}: pydantic validation error: {exc.errors()}"
            ) from exc
    return out


def validate_package_output(
    *, inputs: ColumnInputs, package_outputs: list[ColumnOutput]
) -> None:
    """Per-package belt-and-braces invariant (§8.2)."""
    if len(package_outputs) != len(inputs.column_names):
        raise ColumnPackageInvariantError(
            f"package {inputs.package_id}: concatenated outputs have "
            f"{len(package_outputs)} entries, expected "
            f"{len(inputs.column_names)}"
        )
    output_names = [o.column_name for o in package_outputs]
    for i, (a, b) in enumerate(zip(inputs.column_names, output_names, strict=True)):
        if a != b:
            raise ColumnPackageInvariantError(
                f"package {inputs.package_id}: position {i}: "
                f"expected {a!r}, got {b!r}"
            )


def run_generate(
    *,
    request: ColumnsGenerateRequest,
    settings: Settings,
    load_generation_model: LoadGenerationModelFn = gen_mod.load_generation_model,
    generate_json_list: GenerateJsonListFn = gen_mod.generate_json_list,
    logger: structlog.BoundLogger | None = None,
) -> ColumnsGenerateRunSummary:
    """Main entry. Reads column_inputs, generates per-chunk JSON, writes
    `stage/<run_id>/columns/*.jsonl`."""
    log = logger or get_logger("semantic_enrich.column_generator")
    started = time.monotonic()

    input_row_count = preflight(settings=settings, request=request)
    chunk_size = request.chunk_size or settings.column_chunk_size
    if chunk_size < 1:
        raise RuntimeError(
            f"chunk_size must be >= 1; got {chunk_size}"
        )

    log.info(
        "columns_generate_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        chunk_size=chunk_size,
        input_row_count=input_row_count,
    )

    staged_ids = _read_staged_package_ids_for_columns(
        run_id=request.run_id, staging_dir=settings.staging_dir
    )

    models_lock = _load_models_lock(settings.staging_dir)
    gen_commit: str | None = None
    if models_lock is not None:
        try:
            gen_commit = models_lock["generation"]["commit"]
        except (KeyError, TypeError):
            gen_commit = None

    model: GenerationModel | None = None
    tokenizer: Any = None
    if not request.dry_run:
        try:
            model = load_generation_model(
                settings.generation_model,
                dtype=settings.generation_dtype,
                device=settings.device,
                cache_dir=str(settings.hf_cache_dir)
                if settings.hf_cache_dir
                else None,
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
            log.error(
                "generation_model_load_failed", error=str(exc), exc_info=True
            )
            raise

    def _on_flush(path: Path, seq: int, row_count: int) -> None:
        log.info(
            "columns_flush_written",
            run_id=request.run_id,
            path=str(path),
            flush_seq=seq,
            row_count=row_count,
        )

    writer = stage_io.StageWriter(
        run_id=request.run_id,
        artifact="columns",
        # Flush by package, not by row — `flush_every_n_packages` is
        # the package cadence; columns ship one row per column, so a
        # 100-column package contributes ~100 rows. Use 50x the
        # package cadence so the row-buffer flushes at a similar
        # disk frequency to the datasets pass.
        staging_dir=settings.staging_dir,
        flush_every=max(1, settings.flush_every_n_packages * 50),
        on_flush=_on_flush,
    )

    counters = Counters()
    chunks_total = 0
    chunks_retried = 0
    columns_generated = 0

    try:
        for _, _, inputs in stage_io.iter_staged_rows(
            run_id=request.run_id,
            artifact="column_inputs",
            staging_dir=settings.staging_dir,
            row_type=ColumnInputs,
        ):
            plog = log.bind(package_id=inputs.package_id)

            if inputs.package_id in staged_ids:
                plog.info(
                    "package_columns_skipped_already_staged",
                    staged_at=None,
                )
                counters.skipped += 1
                continue

            if not inputs.column_names:
                # Empty packages should be filtered upstream by
                # columns-extract; defensively pass them through as
                # generated with zero rows.
                plog.warning("package_columns_inputs_empty_at_generate")
                counters.generated += 1
                continue

            chunk_list = list(
                chunk_columns(inputs=inputs, chunk_size=chunk_size)
            )

            # Safety belt: refuse to issue >max_chunks `generate_json`
            # calls for one package (§7.4).
            if len(chunk_list) > settings.column_chunk_max_chunks_per_package:
                plog.error(
                    "package_columns_generation_failed",
                    reason="chunk_count_exceeded_cap",
                    chunk_count=len(chunk_list),
                    cap=settings.column_chunk_max_chunks_per_package,
                )
                writer.append(
                    _failure_marker(
                        inputs=inputs,
                        settings=settings,
                        gen_commit=gen_commit,
                        run_id=request.run_id,
                        reason="chunk_count_exceeded_cap",
                        dry_run=request.dry_run,
                    )
                )
                counters.failed += 1
                continue

            # Per-package fan-out across chunks.
            package_outputs: list[ColumnOutput] = []
            failed_reason: str | None = None
            for chunk in chunk_list:
                chunks_total += 1
                t0 = time.monotonic()
                user_msg = column_prompt.render_user_message(
                    inputs=inputs,
                    chunk_index=chunk.chunk_index,
                    chunk_count=chunk.chunk_count,
                    column_names=chunk.column_names,
                )

                if request.dry_run:
                    plog.info(
                        "would_have_generated_chunk",
                        chunk_index=chunk.chunk_index,
                        chunk_count=chunk.chunk_count,
                        column_count=len(chunk.column_names),
                        prompt_token_estimate=column_prompt.estimate_tokens(
                            user_msg
                        ),
                    )
                    package_outputs.extend(
                        _dry_run_chunk_outputs(chunk)
                    )
                    continue

                assert model is not None and tokenizer is not None
                rendered = _build_chat_prompt(
                    tokenizer=tokenizer, user_msg=user_msg
                )
                try:
                    chunk_outputs = _generate_chunk_with_retry(
                        chunk=chunk,
                        rendered=rendered,
                        model=model,
                        generate_json_list=generate_json_list,
                        settings=settings,
                        log=plog,
                    )
                except _ChunkRetriedSignal as cr:
                    chunks_retried += 1
                    chunk_outputs = cr.outputs
                except ColumnChunkInvariantError as exc:
                    chunks_retried += 1  # counted; the retry happened
                    plog.error(
                        "package_columns_generation_failed",
                        reason="chunk_invariant_violation_after_retry",
                        failed_chunk_index=chunk.chunk_index,
                        error=str(exc),
                    )
                    failed_reason = "chunk_invariant_violation_after_retry"
                    break
                except Exception as exc:
                    plog.exception(
                        "package_columns_generation_failed",
                        reason="generate_json_error",
                        failed_chunk_index=chunk.chunk_index,
                        error=str(exc),
                    )
                    failed_reason = "generate_json_error"
                    break

                plog.info(
                    "package_chunk_generation_done",
                    chunk_index=chunk.chunk_index,
                    chunk_count=chunk.chunk_count,
                    column_count=len(chunk.column_names),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
                package_outputs.extend(chunk_outputs)

            if failed_reason is not None:
                writer.append(
                    _failure_marker(
                        inputs=inputs,
                        settings=settings,
                        gen_commit=gen_commit,
                        run_id=request.run_id,
                        reason=failed_reason,
                        dry_run=request.dry_run,
                    )
                )
                counters.failed += 1
                continue

            # Belt-and-braces per-package check (§8.2).
            try:
                validate_package_output(
                    inputs=inputs, package_outputs=package_outputs
                )
            except ColumnPackageInvariantError as exc:
                plog.error(
                    "package_columns_generation_failed",
                    reason="package_invariant_violation",
                    error=str(exc),
                )
                writer.append(
                    _failure_marker(
                        inputs=inputs,
                        settings=settings,
                        gen_commit=gen_commit,
                        run_id=request.run_id,
                        reason="package_invariant_violation",
                        dry_run=request.dry_run,
                    )
                )
                counters.failed += 1
                continue

            generated_at = datetime.now(UTC)
            for output in package_outputs:
                writer.append(
                    StagedColumnRow(
                        package_id=inputs.package_id,
                        column_name=output.column_name,
                        semantic_type=output.semantic_type,
                        description=output.description,
                        sample_values=list(output.sample_values),
                        embedding=None,
                        generated_at=generated_at,
                        generation_model=settings.generation_model,
                        generation_model_commit=gen_commit,
                        generation_run_id=request.run_id,
                        generation_failed=False,
                        failure_reason=None,
                        dry_run=request.dry_run,
                    )
                )
                columns_generated += 1
            counters.generated += 1
            plog.info(
                "package_columns_generation_done",
                chunk_count=len(chunk_list),
                column_count=len(package_outputs),
            )
    finally:
        writer.close()
        if model is not None:
            del model
            _drop_cuda_cache()

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = ColumnsGenerateRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        input_row_count=input_row_count,
        packages_generated=counters.generated,
        packages_skipped_already_staged=counters.skipped,
        packages_failed=counters.failed,
        chunks_total=chunks_total,
        chunks_retried=chunks_retried,
        columns_generated=columns_generated,
        flush_files_written=writer.files_written,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "columns_generate_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


# ── Internal helpers ──


class _ChunkRetriedSignal(Exception):  # noqa: N818  (control-flow signal, not an error)
    """Internal signal: a chunk failed the per-chunk invariant on the
    first attempt but succeeded after one retry. Carries the retry's
    outputs back to the main loop."""

    def __init__(self, outputs: list[ColumnOutput]) -> None:
        super().__init__()
        self.outputs = outputs


def _generate_chunk_with_retry(
    *,
    chunk: _ColumnChunk,
    rendered: str,
    model: GenerationModel,
    generate_json_list: GenerateJsonListFn,
    settings: Settings,
    log: structlog.BoundLogger,
) -> list[ColumnOutput]:
    """Issue one `generate_json_list` call and validate. On invariant
    violation, retry once at `column_chunk_retry_temperature` (§8.3).
    Two consecutive failures raise `ColumnChunkInvariantError` for the
    main loop to translate into a package-level failure."""
    try:
        response = generate_json_list(
            rendered,
            COLUMNS_GUIDED_JSON_SCHEMA,
            model=model,
            max_tokens=settings.generation_max_tokens,
            temperature=settings.generation_temperature,
        )
        return validate_chunk_output(chunk, response)
    except ColumnChunkInvariantError as exc:
        log.warning(
            "package_chunk_invariant_violation",
            chunk_index=chunk.chunk_index,
            reason="length_or_name_mismatch",
            retry_attempted=True,
            retry_temperature=settings.column_chunk_retry_temperature,
            error=str(exc),
        )

    retry_response = generate_json_list(
        rendered,
        COLUMNS_GUIDED_JSON_SCHEMA,
        model=model,
        max_tokens=settings.generation_max_tokens,
        temperature=settings.column_chunk_retry_temperature,
    )
    outputs = validate_chunk_output(chunk, retry_response)
    raise _ChunkRetriedSignal(outputs=outputs)


def _dry_run_chunk_outputs(chunk: _ColumnChunk) -> list[ColumnOutput]:
    """Placeholder outputs for dry-run. Descriptions are deliberately
    short (just over the 20-char floor) so any accidental load of
    dry-run JSONL would surface as a description that reads
    obviously synthetic."""
    placeholder = "DRY_RUN_PLACEHOLDER_DESCRIPTION"
    return [
        ColumnOutput(
            column_name=name,
            semantic_type=None,
            description=placeholder,
            sample_values=list(chunk.sample_values.get(name, ())[:3]),
        )
        for name in chunk.column_names
    ]


def _failure_marker(
    *,
    inputs: ColumnInputs,
    settings: Settings,
    gen_commit: str | None,
    run_id: str,
    reason: str,
    dry_run: bool,
) -> StagedColumnRow:
    """Build a generation-failed marker row (§7.4 / §8.3 / §9.2). The
    gap-fill check recognises this as 'do not reprocess without
    operator intervention'; the load pass filters it out."""
    return StagedColumnRow(
        package_id=inputs.package_id,
        column_name="__failure_marker__",
        semantic_type=None,
        description="GENERATION_FAILED_PLACEHOLDER_DESCRIPTION_MIN_LEN_OK",
        sample_values=[],
        embedding=None,
        generated_at=datetime.now(UTC),
        generation_model=settings.generation_model,
        generation_model_commit=gen_commit,
        generation_run_id=run_id,
        generation_failed=True,
        failure_reason=reason,
        dry_run=dry_run,
    )


def _build_chat_prompt(*, tokenizer: Any, user_msg: str) -> str:
    """Wrap system + user messages through Qwen's chat template."""
    messages = [
        {"role": "system", "content": column_prompt.COLUMNS_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    rendered: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return rendered


def _read_staged_package_ids_for_columns(
    *, run_id: str, staging_dir: Path
) -> set[str]:
    """Set of `package_id`s already present under
    `stage/<run_id>/columns/*.jsonl`. Includes failure markers (so
    the gap-fill check doesn't reprocess a previously-failed package
    without operator intervention).
    """
    return stage_io.read_staged_package_ids(
        run_id=run_id, artifact="columns", staging_dir=staging_dir
    )


def _vram_allocated_mb() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _drop_cuda_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_invariant(
    summary: ColumnsGenerateRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.packages_generated
        + summary.packages_skipped_already_staged
        + summary.packages_failed
    )
    if total != summary.input_row_count:
        log.error(
            "run_invariant_violated",
            subcommand="columns-generate",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"columns-generate rows accounted-for mismatch: {summary}"
        )
