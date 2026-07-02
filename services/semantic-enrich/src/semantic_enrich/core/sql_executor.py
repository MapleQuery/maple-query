"""Bounded execution of a guard-approved SELECT.

Thin wrapper around `BqClient.run_bounded_query`. Enforces:
  - hard row cap (client-side, defence in depth over the SQL LIMIT)
  - hard byte cap (`maximum_bytes_billed`)
  - hard timeout (`job.result(timeout=...)`)

No retry — the guard has already accepted the SQL, so the executor's
job is to report the outcome verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings


@dataclass(frozen=True)
class ExecutionResult:
    """Runner-facing execution outcome. Fields map 1:1 into the per-
    question grade."""

    rows: list[dict[str, Any]]
    row_count: int
    bytes_billed: int
    slot_ms: int
    elapsed_ms: int
    timed_out: bool
    error: str | None


def execute(
    *, sql: str, bq: BqClient, settings: Settings
) -> ExecutionResult:
    result = bq.run_bounded_query(
        sql,
        timeout_ms=settings.eval_query_timeout_ms,
        max_bytes_billed=settings.eval_max_bytes_billed,
        row_limit=settings.eval_row_limit,
    )
    return ExecutionResult(
        rows=result.rows,
        row_count=len(result.rows),
        bytes_billed=result.total_bytes_billed,
        slot_ms=result.slot_ms,
        elapsed_ms=result.elapsed_ms,
        timed_out=result.timed_out,
        error=result.error,
    )
