"""One-shot OpenAI bypass for `columns-generate`.

The canonical `columns-generate` runs Qwen on the GPU box; for a fast
backfill the OpenAI structured-outputs API is the right tool. This
script reads `stage/<run_id>/column_inputs/*.jsonl`, hits OpenAI in
parallel, and writes `stage/<run_id>/columns/openai.jsonl` in the
exact format `columns-embed` + `columns-load` expect.

What stays as-is downstream:
  - `columns-embed` reads any `*.jsonl` under `stage/<run_id>/columns/`
    and rewrites it with embeddings populated.
  - `columns-load` coalesces, validates, MERGEs into `semantic.columns`.

Reused from the package (single source of truth):
  - prompt template + per-chunk rendering (`column_prompt`)
  - chunking + per-chunk + per-package invariant validators
    (`column_generator.chunk_columns`,
     `validate_chunk_output`, `validate_package_output`)
  - on-disk row shape (`StagedColumnRow`)

Lives in `scripts/` rather than the package proper because it is a
one-shot bypass, not a permanent code path; the canonical 4.5
architecture stays Qwen-on-GPU.

Usage::

    cd services/semantic-enrich
    uv pip install -e '.[openai]'
    export OPENAI_API_KEY=sk-...
    uv run python scripts/columns_generate_openai.py \\
        --run-id "$RUN_ID" \\
        --model gpt-4o-mini \\
        --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from openai import APIError, AsyncOpenAI, RateLimitError

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_generator import (
    chunk_columns,
    validate_chunk_output,
    validate_package_output,
)
from semantic_enrich.core.column_prompt import (
    COLUMNS_SYSTEM_PROMPT,
    render_user_message,
)
from semantic_enrich.providers.logging import configure_logging, get_logger
from semantic_enrich.types import (
    ColumnChunkInvariantError,
    ColumnInputs,
    ColumnOutput,
    ColumnPackageInvariantError,
    StagedColumnRow,
)

# OpenAI structured-outputs strict mode requires:
#   - additionalProperties: false on every object
#   - every property in required
#   - no minLength / maxLength / minItems / maxItems
#
# Length constraints (description 20-600 chars, sample_values ≤10)
# are enforced client-side by the `ColumnOutput` pydantic model
# inside `validate_chunk_output` — same path the local Qwen
# pipeline uses. A violation triggers the same single-retry behaviour.
#
# The top-level wrapper `{"columns": [...]}` exists because OpenAI
# strict mode requires a top-level *object*, not an array.
OPENAI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["columns"],
    "properties": {
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "column_name",
                    "semantic_type",
                    "description",
                    "sample_values",
                ],
                "properties": {
                    "column_name": {"type": "string"},
                    "semantic_type": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "sample_values": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ── OpenAI call + retry plumbing ──


async def _call_openai_for_chunk(
    *,
    client: AsyncOpenAI,
    model: str,
    user_msg: str,
    temperature: float,
    rate_limit_backoff: float,
    log: structlog.BoundLogger,
) -> list[dict[str, Any]]:
    """One OpenAI structured-outputs call. Returns the unwrapped
    columns array. Retries on `RateLimitError` / `APIError` with
    exponential backoff (5 attempts, doubling)."""
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": COLUMNS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "columns_chunk_response",
                        "strict": True,
                        "schema": OPENAI_RESPONSE_SCHEMA,
                    },
                },
                temperature=temperature,
            )
        except RateLimitError as exc:
            wait = rate_limit_backoff * (2**attempt)
            log.warning(
                "openai_rate_limited",
                wait_s=wait,
                attempt=attempt + 1,
                error=str(exc),
            )
            await asyncio.sleep(wait)
            continue
        except APIError as exc:
            if attempt == 4:
                raise
            wait = rate_limit_backoff * (2**attempt)
            log.warning(
                "openai_api_error",
                wait_s=wait,
                attempt=attempt + 1,
                error=str(exc),
            )
            await asyncio.sleep(wait)
            continue

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("openai returned empty content")
        payload: dict[str, Any] = json.loads(content)
        columns = payload.get("columns")
        if not isinstance(columns, list):
            raise RuntimeError(
                f"openai returned response without `columns` array: {payload!r}"
            )
        return columns

    raise RuntimeError("exhausted openai rate-limit retries")


async def _generate_chunk_with_invariant_retry(
    *,
    client: AsyncOpenAI,
    model: str,
    chunk: Any,  # `_ColumnChunk` is private to column_generator
    user_msg: str,
    retry_temperature: float,
    rate_limit_backoff: float,
    log: structlog.BoundLogger,
) -> tuple[list[ColumnOutput], bool]:
    """Issue one chunk call; validate. On invariant violation, retry
    once at `retry_temperature` (§8.3). Returns `(outputs, retried)`.
    Two consecutive violations re-raise `ColumnChunkInvariantError`
    for the per-package handler to translate into a failure marker."""
    response = await _call_openai_for_chunk(
        client=client,
        model=model,
        user_msg=user_msg,
        temperature=0.0,
        rate_limit_backoff=rate_limit_backoff,
        log=log,
    )
    try:
        return validate_chunk_output(chunk, response), False
    except ColumnChunkInvariantError as exc:
        log.warning(
            "package_chunk_invariant_violation",
            chunk_index=chunk.chunk_index,
            retry_attempted=True,
            retry_temperature=retry_temperature,
            error=str(exc),
        )

    retry_response = await _call_openai_for_chunk(
        client=client,
        model=model,
        user_msg=user_msg,
        temperature=retry_temperature,
        rate_limit_backoff=rate_limit_backoff,
        log=log,
    )
    outputs = validate_chunk_output(chunk, retry_response)
    return outputs, True


# ── Per-package orchestration ──


async def _generate_one_package(
    *,
    client: AsyncOpenAI,
    model: str,
    inputs: ColumnInputs,
    chunk_size: int,
    max_chunks: int,
    retry_temperature: float,
    rate_limit_backoff: float,
    run_id: str,
    log: structlog.BoundLogger,
) -> tuple[list[StagedColumnRow], dict[str, int]]:
    """Run all chunks for one package; return the rows + per-package
    counters. On any failure, returns a single failure-marker row so
    the gap-fill check doesn't reprocess the package without operator
    intervention."""
    plog = log.bind(package_id=inputs.package_id)
    counters = {"chunks_total": 0, "chunks_retried": 0}

    if not inputs.column_names:
        plog.warning("package_columns_inputs_empty_at_generate")
        return [], counters

    chunk_list = list(chunk_columns(inputs=inputs, chunk_size=chunk_size))
    if len(chunk_list) > max_chunks:
        plog.error(
            "package_columns_generation_failed",
            reason="chunk_count_exceeded_cap",
            chunk_count=len(chunk_list),
            cap=max_chunks,
        )
        return [
            _failure_marker(inputs, model, run_id, "chunk_count_exceeded_cap")
        ], counters

    package_outputs: list[ColumnOutput] = []
    for chunk in chunk_list:
        t0 = time.monotonic()
        user_msg = render_user_message(
            inputs=inputs,
            chunk_index=chunk.chunk_index,
            chunk_count=chunk.chunk_count,
            column_names=chunk.column_names,
        )
        try:
            outputs, retried = await _generate_chunk_with_invariant_retry(
                client=client,
                model=model,
                chunk=chunk,
                user_msg=user_msg,
                retry_temperature=retry_temperature,
                rate_limit_backoff=rate_limit_backoff,
                log=plog,
            )
        except ColumnChunkInvariantError as exc:
            counters["chunks_total"] += 1
            counters["chunks_retried"] += 1
            plog.error(
                "package_columns_generation_failed",
                reason="chunk_invariant_violation_after_retry",
                failed_chunk_index=chunk.chunk_index,
                error=str(exc),
            )
            return [
                _failure_marker(
                    inputs, model, run_id, "chunk_invariant_violation_after_retry"
                )
            ], counters
        except Exception as exc:
            counters["chunks_total"] += 1
            plog.exception(
                "package_columns_generation_failed",
                reason="generate_json_error",
                failed_chunk_index=chunk.chunk_index,
                error=str(exc),
            )
            return [
                _failure_marker(inputs, model, run_id, "generate_json_error")
            ], counters

        counters["chunks_total"] += 1
        if retried:
            counters["chunks_retried"] += 1
        plog.info(
            "package_chunk_generation_done",
            chunk_index=chunk.chunk_index,
            chunk_count=chunk.chunk_count,
            column_count=len(chunk.column_names),
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
        package_outputs.extend(outputs)

    try:
        validate_package_output(inputs=inputs, package_outputs=package_outputs)
    except ColumnPackageInvariantError as exc:
        plog.error(
            "package_columns_generation_failed",
            reason="package_invariant_violation",
            error=str(exc),
        )
        return [
            _failure_marker(inputs, model, run_id, "package_invariant_violation")
        ], counters

    generated_at = datetime.now(UTC)
    rows = [
        StagedColumnRow(
            package_id=inputs.package_id,
            column_name=output.column_name,
            semantic_type=output.semantic_type,
            description=output.description,
            sample_values=list(output.sample_values),
            embedding=None,
            generated_at=generated_at,
            generation_model=model,
            generation_model_commit=None,
            generation_run_id=run_id,
            generation_failed=False,
            failure_reason=None,
            dry_run=False,
        )
        for output in package_outputs
    ]
    plog.info(
        "package_columns_generation_done",
        chunk_count=len(chunk_list),
        column_count=len(package_outputs),
    )
    return rows, counters


def _failure_marker(
    inputs: ColumnInputs, model: str, run_id: str, reason: str
) -> StagedColumnRow:
    return StagedColumnRow(
        package_id=inputs.package_id,
        column_name="__failure_marker__",
        semantic_type=None,
        description="GENERATION_FAILED_PLACEHOLDER_DESCRIPTION_MIN_LEN_OK",
        sample_values=[],
        embedding=None,
        generated_at=datetime.now(UTC),
        generation_model=model,
        generation_model_commit=None,
        generation_run_id=run_id,
        generation_failed=True,
        failure_reason=reason,
        dry_run=False,
    )


# ── I/O helpers ──


def _iter_input_packages(
    *, staging_dir: Path, run_id: str
) -> Iterator[ColumnInputs]:
    inputs_dir = staging_dir / run_id / "column_inputs"
    if not inputs_dir.is_dir():
        raise RuntimeError(
            f"column_inputs_dir_missing: {inputs_dir} — run "
            "`semantic-enrich columns-extract` first."
        )
    for path in sorted(inputs_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield ColumnInputs.model_validate_json(line)


def _read_done_package_ids(*, staging_dir: Path, run_id: str) -> set[str]:
    columns_dir = staging_dir / run_id / "columns"
    if not columns_dir.is_dir():
        return set()
    done: set[str] = set()
    for path in sorted(columns_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = obj.get("package_id")
                if isinstance(pid, str):
                    done.add(pid)
    return done


# ── Main async loop ──


async def _main_async(args: argparse.Namespace) -> int:
    configure_logging()
    log = get_logger("scripts.columns_generate_openai")

    settings = Settings()  # picks up WHENRICH_STAGING_DIR + defaults
    staging_dir = (
        Path(args.staging_dir) if args.staging_dir else settings.staging_dir
    )
    run_id = args.run_id or settings.run_id

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("missing_openai_api_key", hint="export OPENAI_API_KEY=sk-...")
        return 3

    columns_dir = staging_dir / run_id / "columns"
    columns_dir.mkdir(parents=True, exist_ok=True)
    output_path = columns_dir / "openai.jsonl"

    inputs_list = list(_iter_input_packages(staging_dir=staging_dir, run_id=run_id))
    done_ids = _read_done_package_ids(staging_dir=staging_dir, run_id=run_id)
    pending = [pkg for pkg in inputs_list if pkg.package_id not in done_ids]
    if args.limit_packages:
        pending = pending[: args.limit_packages]

    log.info(
        "openai_columns_generate_start",
        run_id=run_id,
        model=args.model,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        input_row_count=len(inputs_list),
        already_done=len(done_ids),
        pending=len(pending),
        output_path=str(output_path),
    )

    started = time.monotonic()

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    totals = {
        "chunks_total": 0,
        "chunks_retried": 0,
        "columns_generated": 0,
        "packages_generated": 0,
        "packages_failed": 0,
        "packages_empty": 0,
    }

    async def _run_one(pkg: ColumnInputs) -> None:
        async with semaphore:
            try:
                rows, counters = await _generate_one_package(
                    client=client,
                    model=args.model,
                    inputs=pkg,
                    chunk_size=args.chunk_size,
                    max_chunks=args.max_chunks,
                    retry_temperature=args.retry_temperature,
                    rate_limit_backoff=args.rate_limit_backoff,
                    run_id=run_id,
                    log=log,
                )
            except Exception as exc:
                log.exception(
                    "package_unexpected_error",
                    package_id=pkg.package_id,
                    error=str(exc),
                )
                rows = [_failure_marker(pkg, args.model, run_id, "unexpected_error")]
                counters = {"chunks_total": 0, "chunks_retried": 0}

            totals["chunks_total"] += counters["chunks_total"]
            totals["chunks_retried"] += counters["chunks_retried"]

            async with write_lock:
                with output_path.open("a") as f:
                    for row in rows:
                        f.write(row.model_dump_json() + "\n")

            if not rows:
                totals["packages_empty"] += 1
            elif rows[0].generation_failed:
                totals["packages_failed"] += 1
            else:
                totals["packages_generated"] += 1
                totals["columns_generated"] += len(rows)

    await asyncio.gather(*(_run_one(pkg) for pkg in pending))

    duration_s = int(time.monotonic() - started)
    log.info(
        "openai_columns_generate_finish",
        run_id=run_id,
        duration_s=duration_s,
        packages_skipped_already_staged=len(done_ids),
        **totals,
    )
    # Mirror the entrypoint shape: dump a JSON summary to stdout so
    # the operator can pipe it into a follow-up command.
    print(
        json.dumps(
            {
                "run_id": run_id,
                "model": args.model,
                "duration_s": duration_s,
                "packages_skipped_already_staged": len(done_ids),
                **totals,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot OpenAI bypass for columns-generate. Reads "
            "stage/<run_id>/column_inputs/*.jsonl, writes "
            "stage/<run_id>/columns/openai.jsonl."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Match a prior columns-extract run. Defaults to WHENRICH_RUN_ID.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model. Default gpt-4o-mini.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Max in-flight package generations. Default 50.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="Columns per chunk. Default 100 (matches WHENRICH_COLUMN_CHUNK_SIZE).",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=20,
        help="Wide-package safety belt. Default 20.",
    )
    parser.add_argument(
        "--retry-temperature", type=float, default=0.2,
        help="Single-retry temperature on per-chunk invariant violation. Default 0.2.",
    )
    parser.add_argument(
        "--rate-limit-backoff", type=float, default=2.0,
        help="Initial backoff (seconds) on rate-limit errors. Default 2.",
    )
    parser.add_argument(
        "--staging-dir", default=None,
        help="Override WHENRICH_STAGING_DIR.",
    )
    parser.add_argument(
        "--limit-packages", type=int, default=None,
        help="Cap pending packages to N (for smoke runs).",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
