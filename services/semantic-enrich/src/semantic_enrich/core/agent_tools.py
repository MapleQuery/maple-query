"""OpenAI tool schemas + implementations for the agent loop.

Four tools. Names and schemas are frozen — prompt quality tuning
depends on them staying stable. Implementation-side, every tool:

- Validates its arguments against the JSON schema at the runtime
  boundary (Structured Outputs enforces it vendor-side, but the loop
  double-checks so a schema drift becomes a `tool_error`, not a
  Python exception).
- Returns a JSON-serializable dict that goes back to the model as the
  next-turn tool result.
- Emits events into the loop's event bus for the UI-facing SSE stream.

The tools are pure functions of a `ToolContext`; the loop owns the
per-turn mutable state.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.retrieval import (
    embed_question,
    retrieve_columns,
    retrieve_packages,
)
from semantic_enrich.core.sql_executor import execute as execute_sql
from semantic_enrich.core.sql_guard import guard

TOOL_NAMES = (
    "search_datasets",
    "search_columns",
    "sample_rows",
    "run_sql",
)


# ── OpenAI tool schemas (frozen) ──


def tool_schemas() -> list[dict[str, Any]]:
    """Return the list of tool definitions wrapped in OpenAI's outer
    `{"type": "function", "function": …}` envelope.

    A new list is returned on each call so callers can't mutate the
    shared template."""
    return [
        {"type": "function", "function": _SEARCH_DATASETS},
        {"type": "function", "function": _SEARCH_COLUMNS},
        {"type": "function", "function": _SAMPLE_ROWS},
        {"type": "function", "function": _RUN_SQL},
    ]


_SEARCH_DATASETS: dict[str, Any] = {
    "name": "search_datasets",
    "description": (
        "Semantic search over dataset summaries. Returns the top-k "
        "package candidates by cosine similarity to the query. Use "
        "this to discover which datasets might contain data relevant "
        "to the user's question. Call once per distinct sub-question."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of what the user "
                    "wants. Rephrase in your own words to maximize "
                    "retrieval quality."
                ),
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
    },
}


_SEARCH_COLUMNS: dict[str, Any] = {
    "name": "search_columns",
    "description": (
        "Scoped semantic search over columns for one or more packages. "
        "Use after search_datasets to find which columns in the "
        "candidate packages match the question. Do not call this "
        "without first calling search_datasets in the same turn."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["package_ids", "query"],
        "properties": {
            "package_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
            },
            "query": {"type": "string"},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": 15,
            },
        },
    },
}


_SAMPLE_ROWS: dict[str, Any] = {
    "name": "sample_rows",
    "description": (
        "Fetch a small sample of rows from a single package. Use to "
        "check real column names and value shapes before writing SQL. "
        "Cheap; call whenever you're unsure whether a column exists "
        "or what its values look like."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["package_id"],
        "properties": {
            "package_id": {"type": "string"},
            "n": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
    },
}


_RUN_SQL: dict[str, Any] = {
    "name": "run_sql",
    "description": (
        "Execute one SELECT statement against BigQuery. Must reference "
        "only tables in the `raw` and `semantic` datasets. Must include "
        "LIMIT 100 or lower. If the previous run_sql returned zero "
        "rows or an error, do not retry with a trivial edit — "
        "reconsider your candidate columns or packages first."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["sql", "rationale"],
        "properties": {
            "sql": {"type": "string"},
            "rationale": {
                "type": "string",
                "description": (
                    "1-2 sentences on what this query is expected to "
                    "return and why you picked these tables/columns."
                ),
            },
        },
    },
}


# ── Runtime types ──


@dataclass
class LoopState:
    """Per-turn mutable state threaded through every tool call.

    The loop owns this; tools read/write. Keeps tool implementations
    pure functions of their arguments — no hidden module globals."""

    conversation_id: str
    turn_id: str
    question: str
    question_vec: list[float] | None = None
    known_package_ids: set[str] = field(default_factory=set)
    tool_call_count: int = 0
    sql_execution_count: int = 0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    dollars_spent: float = 0.0
    tool_call_ids: list[str] = field(default_factory=list)


EmitFn = Callable[[agent_events.AgentEvent], None]


@dataclass
class ToolContext:
    """Everything a tool needs to run. Constructed once per turn."""

    bq: BqClient
    openai_client: OpenAIClient
    settings: Settings
    state: LoopState
    emit: EmitFn


class InvalidToolArgsError(ValueError):
    """Tool argument validation failed at the runtime boundary."""


# ── Tool implementations ──


def run_search_datasets(*, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    query = _require_str(args, "query")
    k = _optional_int(args, "k", default=5, min_=1, max_=10)

    ctx.emit(agent_events.RetrievalStarted(query=query, k=k))

    # Question embedding is cached per turn — repeat calls with the
    # same query reuse it. Different-query calls re-embed.
    vec = _get_or_embed(ctx=ctx, query=query)

    settings = ctx.settings.model_copy(update={"eval_k_packages": k})
    packages, _latency = retrieve_packages(
        bq=ctx.bq, question_vec=vec, settings=settings
    )
    candidates: list[dict[str, Any]] = [
        {
            "package_id": p.package_id,
            "summary": p.summary,
            "grain": p.grain,
            "measures": list(p.measures),
            "dimensions": list(p.dimensions),
            "date_range_start": p.date_range_start,
            "date_range_end": p.date_range_end,
            "distance": p.distance,
        }
        for p in packages
    ]

    # Track known package IDs so search_columns can enforce the
    # whitelist at runtime.
    for c in candidates:
        ctx.state.known_package_ids.add(str(c["package_id"]))

    ctx.emit(agent_events.DatasetsRanked(candidates=candidates))
    return {"candidates": candidates}


def run_search_columns(
    *, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    package_ids = _require_str_list(args, "package_ids")
    if not package_ids:
        raise InvalidToolArgsError("package_ids must be non-empty")
    if len(package_ids) > 10:
        raise InvalidToolArgsError("package_ids must have at most 10 entries")
    query = _require_str(args, "query")
    k = _optional_int(args, "k", default=15, min_=1, max_=30)

    unknown = [p for p in package_ids if p not in ctx.state.known_package_ids]
    if unknown:
        raise InvalidToolArgsError(
            f"invalid_package_id: {unknown!r} not returned by "
            "search_datasets in this turn"
        )

    vec = _get_or_embed(ctx=ctx, query=query)
    settings = ctx.settings.model_copy(update={"eval_k_columns": k})
    columns, _latency = retrieve_columns(
        bq=ctx.bq,
        question_vec=vec,
        scoped_packages=list(package_ids),
        settings=settings,
    )
    candidates = [
        {
            "package_id": c.package_id,
            "column_name": c.column_name,
            "semantic_type": c.semantic_type,
            "description": c.description,
            "sample_values": list(c.sample_values),
            "distance": c.distance,
        }
        for c in columns
    ]
    ctx.emit(
        agent_events.ColumnsRanked(
            package_ids=list(package_ids), candidates=candidates
        )
    )
    return {"candidates": candidates}


def run_sample_rows(
    *, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    package_id = _require_str(args, "package_id")
    n = _optional_int(args, "n", default=5, min_=1, max_=10)

    project_id = ctx.settings.gcp_project_id
    if not project_id:
        raise InvalidToolArgsError(
            "sample_rows requires WHENRICH_GCP_PROJECT_ID to be set"
        )

    sql = _SAMPLE_ROWS_SQL.format(
        project_id=project_id,
        dataset=ctx.settings.bq_dataset_raw,
        rows_table=ctx.settings.bq_rows_table,
        documents_table=ctx.settings.bq_documents_table,
    )
    params = [
        bigquery.ScalarQueryParameter("pkg", "STRING", package_id),
        bigquery.ScalarQueryParameter("n", "INT64", n),
    ]
    rows = [dict(r) for r in ctx.bq.query_rows(sql, params=params)]
    normalized = [_jsonable(r) for r in rows]
    ctx.emit(
        agent_events.SampleRows(package_id=package_id, rows=normalized)
    )
    return {"package_id": package_id, "rows": normalized}


def run_run_sql(*, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    sql = _require_str(args, "sql")
    rationale = _require_str(args, "rationale")

    ctx.emit(agent_events.SqlGenerated(sql=sql, rationale=rationale))

    # Per-turn SQL-execution budget. Guard-rejected calls do not count;
    # this check runs against successful executions only, so an
    # over-budget attempt short-circuits before touching BQ.
    if ctx.state.sql_execution_count >= ctx.settings.agent_max_sql_executions:
        return {
            "status": "budget_exceeded",
            "cap": ctx.settings.agent_max_sql_executions,
            "message": (
                "Per-turn SQL execution budget exhausted. Reconsider "
                "your approach before requesting another run_sql."
            ),
        }

    guard_result = guard(sql=sql, bq=ctx.bq, settings=ctx.settings)
    ctx.emit(
        agent_events.SqlGuarded(
            accepted=guard_result.accepted,
            reason=guard_result.reason,
            sql_final=guard_result.sql_final,
            dry_run_bytes=guard_result.dry_run_bytes,
        )
    )
    if not guard_result.accepted:
        # Guard rejections come back to the model as tool results so it
        # can correct hallucinated identifiers and retry. Cost is one
        # dry-run.
        return {
            "status": "guard_rejected",
            "reason": guard_result.reason,
            "sql_final": guard_result.sql_final,
        }

    execution = execute_sql(
        sql=guard_result.sql_final, bq=ctx.bq, settings=ctx.settings
    )
    ctx.state.sql_execution_count += 1

    normalized_rows = [_jsonable(r) for r in execution.rows]
    sample_for_model = normalized_rows[:20]
    call_id = uuid.uuid4().hex
    ctx.emit(
        agent_events.SqlExecuted(
            row_count=execution.row_count,
            bytes_billed=execution.bytes_billed,
            elapsed_ms=execution.elapsed_ms,
            sample_rows=normalized_rows[:3],
        )
    )
    ctx.emit(
        agent_events.Rows(
            sql_call_id=call_id, rows=normalized_rows, is_last=True
        )
    )
    if execution.timed_out or execution.error:
        return {
            "status": "execution_error",
            "reason": execution.error or "timed_out",
            "row_count": 0,
            "bytes_billed": execution.bytes_billed,
            "elapsed_ms": execution.elapsed_ms,
        }
    return {
        "status": "ok",
        "row_count": execution.row_count,
        "bytes_billed": execution.bytes_billed,
        "elapsed_ms": execution.elapsed_ms,
        "rows": sample_for_model,
        "truncated": execution.row_count > len(sample_for_model),
    }


def dispatch(
    *, ctx: ToolContext, tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Route a tool call to the matching implementation.

    Callers pass the parsed `arguments` JSON from OpenAI. Runtime
    validation lives inside each tool.
    """
    if tool_name == "search_datasets":
        return run_search_datasets(ctx=ctx, args=args)
    if tool_name == "search_columns":
        return run_search_columns(ctx=ctx, args=args)
    if tool_name == "sample_rows":
        return run_sample_rows(ctx=ctx, args=args)
    if tool_name == "run_sql":
        return run_run_sql(ctx=ctx, args=args)
    raise InvalidToolArgsError(f"unknown_tool: {tool_name!r}")


# ── Helpers ──


def _get_or_embed(*, ctx: ToolContext, query: str) -> list[float]:
    # For the current turn the user's original question is embedded up
    # front and stored in state.question_vec. Follow-up tool queries
    # rephrase, and we re-embed those. The state cache holds the *most
    # recent* embedding so back-to-back identical rephrasings share it.
    cache_attr = f"_embed_{hash(query)}"
    cached = getattr(ctx.state, cache_attr, None)
    if isinstance(cached, list):
        return cached
    if ctx.state.question_vec is None and query == ctx.state.question:
        vec = embed_question(
            openai_client=ctx.openai_client,
            question=query,
            settings=ctx.settings,
        )
        ctx.state.question_vec = vec
        setattr(ctx.state, cache_attr, vec)
        return vec
    if query == ctx.state.question and ctx.state.question_vec is not None:
        return ctx.state.question_vec
    vec = embed_question(
        openai_client=ctx.openai_client,
        question=query,
        settings=ctx.settings,
    )
    setattr(ctx.state, cache_attr, vec)
    return vec


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidToolArgsError(f"missing_or_empty: {key!r}")
    return value


def _require_str_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key)
    if not isinstance(value, list) or not all(
        isinstance(v, str) for v in value
    ):
        raise InvalidToolArgsError(f"invalid_string_list: {key!r}")
    return list(value)


def _optional_int(
    args: dict[str, Any], key: str, *, default: int, min_: int, max_: int
) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidToolArgsError(f"non_integer: {key!r}")
    if value < min_ or value > max_:
        raise InvalidToolArgsError(
            f"out_of_range: {key!r} must be in [{min_}, {max_}]"
        )
    return int(value)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[str(k)] = _coerce(v)
    return out


def _coerce(value: Any) -> Any:
    # BQ row values can be datetime/date/Decimal/bytes — cast anything
    # non-JSON-native to str so json.dumps never blows up when the loop
    # ships the tool result back to OpenAI.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return str(value)


# `sample_rows` uses the raw.rows JOIN raw.documents shape from 3.3.1.
# `LIMIT` is inlined as a query parameter and the tool caps `n` at 10
# via the schema.
_SAMPLE_ROWS_SQL = """
SELECT r.row
FROM `{project_id}.{dataset}.{rows_table}` AS r
JOIN `{project_id}.{dataset}.{documents_table}` AS d USING (document_id)
WHERE d.package_id = @pkg
LIMIT @n
""".strip()


# Re-exported so the loop and tests can reach it without importing
# `time` and asserting perf inline. Currently only used to defensively
# stamp tool durations in future work.
_MONOTONIC = time.monotonic
