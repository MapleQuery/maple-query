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

import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import sqlglot
import sqlglot.expressions as exp
from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.retrieval import (
    embed_question,
    fetch_document_samples,
    retrieve_columns,
    retrieve_documents,
    retrieve_packages,
)
from semantic_enrich.core.sql_executor import execute as execute_sql
from semantic_enrich.core.sql_guard import guard
from semantic_enrich.providers import braintrust_tracing
from semantic_enrich.providers.logging import get_logger

TOOL_NAMES = (
    "search_datasets",
    "search_columns",
    "list_documents",
    "sample_rows",
    "run_sql",
    "describe_corpus",
)

_LOG = get_logger("semantic_enrich.agent_tools")


# ── OpenAI tool schemas (frozen) ──


def tool_schemas() -> list[dict[str, Any]]:
    """Return the list of tool definitions wrapped in OpenAI's outer
    `{"type": "function", "function": …}` envelope.

    A new list is returned on each call so callers can't mutate the
    shared template."""
    return [
        {"type": "function", "function": _SEARCH_DATASETS},
        {"type": "function", "function": _SEARCH_COLUMNS},
        {"type": "function", "function": _LIST_DOCUMENTS},
        {"type": "function", "function": _SAMPLE_ROWS},
        {"type": "function", "function": _RUN_SQL},
        {"type": "function", "function": _DESCRIBE_CORPUS},
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


_LIST_DOCUMENTS: dict[str, Any] = {
    "name": "list_documents",
    "description": (
        "List loaded `raw.documents` rows for one or more candidate "
        "packages, together with each doc's actual JSON key set. Every "
        "run_sql needs a LITERAL `document_id IN (...)` filter picked "
        "from this tool's output — raw.rows is clustered by document_id "
        "and only a literal IN-list gets plan-time pruning. Also use "
        "the per-doc `columns` list to check that the columns you want "
        "actually appear in the doc you inline (different docs in the "
        "same package can have disjoint key sets)."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["package_ids"],
        "properties": {
            "package_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
            },
            "required_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Only return documents whose columns contain every "
                    "listed name. Use when you already know which "
                    "columns your SQL will reference — the returned "
                    "documents are then guaranteed safe to inline "
                    "together for those columns."
                ),
            },
        },
    },
}


_SAMPLE_ROWS: dict[str, Any] = {
    "name": "sample_rows",
    "description": (
        "Fetch a small sample of rows from a single package. Rows come "
        "back with their `document_id` and the row body parsed as a "
        "JSON object. Use to check real column names and value shapes "
        "before writing SQL."
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


_DESCRIBE_CORPUS: dict[str, Any] = {
    "name": "describe_corpus",
    "description": (
        "Statistics about the MapleQuery corpus: dataset, document, "
        "and row counts, and data freshness. Use for questions about "
        "the corpus itself rather than about the data content."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {},
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
    known_document_ids: set[str] = field(default_factory=set)
    # doc_id → the actual JSON key set for that doc, as surfaced by
    # list_documents. Consumed by the run_sql doc/column pairing check
    # so a `JSON_VALUE(..., '$.<col>')` reference against a doc whose
    # `columns` list doesn't include `<col>` is caught as a tool error
    # rather than silently returning all-NULL.
    doc_columns: dict[str, list[str]] = field(default_factory=dict)
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
    # Exported turn-span string for explicit tool-span parenting. None
    # when tracing is off or the caller drives `run_turn` untraced.
    trace_parent: str | None = None


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
            "title": p.title,
            "summary": p.summary,
            "grain": p.grain,
            "measures": list(p.measures),
            "dimensions": list(p.dimensions),
            "date_range_start": p.date_range_start,
            "date_range_end": p.date_range_end,
            "distance": p.distance,
            # Normalized cosine similarity — the form the prompt (and
            # the weak-retrieval policy) can reason about directly.
            "similarity": round(1 - p.distance, 4),
        }
        for p in packages
    ]
    top_similarity = max(
        (float(c["similarity"]) for c in candidates), default=None
    )

    # Track known package IDs so search_columns can enforce the
    # whitelist at runtime.
    for c in candidates:
        ctx.state.known_package_ids.add(str(c["package_id"]))

    ctx.emit(
        agent_events.DatasetsRanked(
            candidates=candidates, top_similarity=top_similarity
        )
    )
    return {"candidates": candidates, "top_similarity": top_similarity}


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


def run_list_documents(
    *, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    package_ids = _require_str_list(args, "package_ids")
    if not package_ids:
        raise InvalidToolArgsError("package_ids must be non-empty")
    if len(package_ids) > 10:
        raise InvalidToolArgsError("package_ids must have at most 10 entries")
    required_columns: list[str] | None = None
    if args.get("required_columns") is not None:
        required_columns = _require_str_list(args, "required_columns")

    unknown = [p for p in package_ids if p not in ctx.state.known_package_ids]
    if unknown:
        raise InvalidToolArgsError(
            f"invalid_package_id: {unknown!r} not returned by "
            "search_datasets in this turn"
        )

    documents, _latency = retrieve_documents(
        bq=ctx.bq, package_ids=list(package_ids), settings=ctx.settings
    )
    samples_by_doc = fetch_document_samples(
        bq=ctx.bq,
        doc_ids=[d.document_id for d in documents],
        settings=ctx.settings,
    )
    payload: list[dict[str, Any]] = []
    for d in documents:
        entry: dict[str, Any] = {
            "document_id": d.document_id,
            "package_id": d.package_id,
            "title": d.title,
            "row_count": d.row_count,
            "resource_last_modified": (
                d.resource_last_modified.isoformat()
                if d.resource_last_modified is not None
                else None
            ),
            "columns": list(d.columns),
        }
        samples = samples_by_doc.get(d.document_id)
        if samples:
            entry["column_samples"] = samples
        if (
            _generated_header_ratio(d.columns)
            > ctx.settings.agent_generated_header_ratio
        ):
            entry["quality"] = "low_generated_headers"
        payload.append(entry)
    # Garbage-header docs sort after every clean doc — demoted, not
    # dropped, so their sample values can still rescue them.
    payload.sort(key=lambda e: "quality" in e)

    # State tracks every doc listed (including any the filter below
    # excludes) so the run_sql pairing check can still validate a doc
    # the model insists on using.
    for entry in payload:
        doc_id = str(entry["document_id"])
        ctx.state.known_document_ids.add(doc_id)
        cols = entry.get("columns") or []
        ctx.state.doc_columns[doc_id] = [str(c) for c in cols]

    result: dict[str, Any] = {"documents": payload}
    filtered_out: list[dict[str, Any]] = []
    if required_columns:
        kept: list[dict[str, Any]] = []
        for entry in payload:
            missing = [
                c for c in required_columns if c not in set(entry["columns"])
            ]
            if missing:
                filtered_out.append(
                    {
                        "document_id": entry["document_id"],
                        "missing_columns": missing,
                    }
                )
            else:
                kept.append(entry)
        if kept:
            result["documents"] = kept
            if filtered_out:
                result["filtered_out"] = filtered_out
        else:
            # An empty listing would push the model toward surrender;
            # return everything and flag the filter as unsatisfiable.
            filtered_out = []
            result["required_columns_unsatisfiable"] = True

    ctx.emit(
        agent_events.DocumentsListed(
            package_ids=list(package_ids),
            documents=result["documents"],
            filtered_out=filtered_out or None,
            required_columns_unsatisfiable=bool(
                result.get("required_columns_unsatisfiable", False)
            ),
        )
    )
    return result


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

    # `row` is a native JSON object; the SDK returns it to Python as a
    # dict, so the model can reference keys verbatim.
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

    # Deterministic server-side fixes the prompt used to beg for. The
    # rewritten SQL is what gets pairing-checked, guarded, executed,
    # and shown on the evidence rail; the `normalizations` field in the
    # tool result teaches the model the corrected form.
    sql, tables_rewritten = normalize_table_references(
        sql, settings=ctx.settings
    )
    sql, json_paths_quoted = autoquote_json_paths(sql)
    normalizations: dict[str, Any] = {}
    if json_paths_quoted:
        normalizations["json_paths_quoted"] = json_paths_quoted
        _LOG.info(
            "agent_autoquote_applied",
            json_paths=json_paths_quoted,
            count=len(json_paths_quoted),
        )
    if tables_rewritten:
        normalizations["tables_rewritten"] = tables_rewritten
        _LOG.info(
            "agent_table_ref_normalized",
            tables=tables_rewritten,
            count=len(tables_rewritten),
        )

    def _result(payload: dict[str, Any]) -> dict[str, Any]:
        if normalizations:
            payload["normalizations"] = normalizations
        return payload

    # Doc/column pairing check. If every JSON_VALUE(..., '$.<col>')
    # reference doesn't line up with the `columns` list of every doc in
    # the WHERE IN, refuse before hitting the SQL guard so the model
    # gets a targeted error telling it which column is missing from
    # which doc — rather than a silent all-NULL result post-execution.
    violations, pairing_msg = check_doc_column_pairing(
        sql=sql, state=ctx.state
    )
    if violations:
        short_reason = (
            f"doc_column_pairing_violation: {len(violations)} column(s) "
            f"referenced not in the inlined document(s)"
        )
        ctx.emit(
            agent_events.SqlGuarded(
                accepted=False,
                reason=short_reason,
                sql_final=sql,
                dry_run_bytes=None,
            )
        )
        return _result(
            {
                "status": "column_not_in_doc",
                "reason": short_reason,
                "message": pairing_msg,
                "violations": violations,
            }
        )

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
        return _result(
            {
                "status": "guard_rejected",
                "reason": guard_result.reason,
                "sql_final": guard_result.sql_final,
            }
        )

    execution = execute_sql(
        sql=guard_result.sql_final, bq=ctx.bq, settings=ctx.settings
    )
    ctx.state.sql_execution_count += 1

    normalized_rows = [_jsonable(r) for r in execution.rows]
    null_ratio_warning = compute_null_ratio_warning(
        sql=guard_result.sql_final,
        rows=normalized_rows,
        settings=ctx.settings,
    )
    sample_for_model = normalized_rows[:20]
    call_id = uuid.uuid4().hex
    ctx.emit(
        agent_events.SqlExecuted(
            row_count=execution.row_count,
            bytes_billed=execution.bytes_billed,
            elapsed_ms=execution.elapsed_ms,
            sample_rows=normalized_rows[:3],
            null_ratio_warning=null_ratio_warning,
        )
    )
    ctx.emit(
        agent_events.Rows(
            sql_call_id=call_id, rows=normalized_rows, is_last=True
        )
    )
    if execution.timed_out or execution.error:
        return _result(
            {
                "status": "execution_error",
                "reason": execution.error or "timed_out",
                "row_count": 0,
                "bytes_billed": execution.bytes_billed,
                "elapsed_ms": execution.elapsed_ms,
            }
        )
    payload: dict[str, Any] = {
        "status": "ok",
        "row_count": execution.row_count,
        "bytes_billed": execution.bytes_billed,
        "elapsed_ms": execution.elapsed_ms,
        "rows": sample_for_model,
        "truncated": execution.row_count > len(sample_for_model),
    }
    if null_ratio_warning is not None:
        payload["null_ratio_warning"] = null_ratio_warning
    return _result(payload)


def run_describe_corpus(
    *, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    project_id = ctx.settings.gcp_project_id
    if not project_id:
        raise InvalidToolArgsError(
            "describe_corpus requires WHENRICH_GCP_PROJECT_ID to be set"
        )
    ttl = ctx.settings.agent_snapshot_refresh_seconds
    with _CORPUS_STATS_LOCK:
        cached = _CORPUS_STATS_CACHE.get(project_id)
        if cached is not None and time.monotonic() - cached[0] < ttl:
            return dict(cached[1])
    stats = _fetch_corpus_stats(ctx=ctx, project_id=project_id)
    with _CORPUS_STATS_LOCK:
        _CORPUS_STATS_CACHE[project_id] = (time.monotonic(), stats)
    return dict(stats)


_CORPUS_DESCRIPTION = (
    "Canadian federal open-data CSVs from open.canada.ca, loaded into "
    "a BigQuery warehouse."
)

# In-process TTL cache, keyed by project id. Corpus stats change on
# warehouse loads, not per turn; repeated calls within the snapshot
# refresh window are free.
_CORPUS_STATS_LOCK = threading.Lock()
_CORPUS_STATS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def reset_corpus_stats_cache() -> None:
    """Test seam — clears the in-process describe_corpus cache."""
    with _CORPUS_STATS_LOCK:
        _CORPUS_STATS_CACHE.clear()


def _fetch_corpus_stats(
    *, ctx: ToolContext, project_id: str
) -> dict[str, Any]:
    s = ctx.settings
    # Row count comes from table metadata — free, no bytes scanned.
    # `raw.rows` is never queried here.
    rows_total = ctx.bq.table_num_rows(
        f"{project_id}.{s.bq_dataset_raw}.{s.bq_rows_table}"
    )
    sql = _CORPUS_STATS_SQL.format(
        project_id=project_id,
        semantic_dataset=s.bq_dataset_semantic,
        datasets_table=s.bq_datasets_table,
        raw_dataset=s.bq_dataset_raw,
        documents_table=s.bq_documents_table,
    )
    rows = list(ctx.bq.query_rows(sql))
    row = rows[0] if rows else {}
    latest = row.get("latest_load_at")
    if isinstance(latest, datetime):
        latest = latest.isoformat()
    return {
        "packages": int(row.get("packages") or 0),
        "documents_loaded": int(row.get("documents_loaded") or 0),
        "rows_total": rows_total,
        "latest_load_at": str(latest) if latest is not None else None,
        "corpus_description": _CORPUS_DESCRIPTION,
    }


# COUNTs over the small metadata tables only; the MAX(loaded_at) mirrors
# the snapshot-hash provider's freshness query.
_CORPUS_STATS_SQL = """
SELECT
  (SELECT COUNT(*)
   FROM `{project_id}.{semantic_dataset}.{datasets_table}`) AS packages,
  (SELECT COUNT(*)
   FROM `{project_id}.{raw_dataset}.{documents_table}`
   WHERE load_status = 'loaded') AS documents_loaded,
  (SELECT CAST(MAX(loaded_at) AS STRING)
   FROM `{project_id}.{semantic_dataset}.{datasets_table}`) AS latest_load_at
""".strip()


_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_datasets": run_search_datasets,
    "search_columns": run_search_columns,
    "list_documents": run_list_documents,
    "sample_rows": run_sample_rows,
    "run_sql": run_run_sql,
    "describe_corpus": run_describe_corpus,
}


def dispatch(
    *, ctx: ToolContext, tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Route a tool call to the matching implementation, wrapped in a
    `tool.<name>` Braintrust span when tracing is on.

    Callers pass the parsed `arguments` JSON from OpenAI. Runtime
    validation lives inside each tool. Span inputs/outputs are digests,
    never full row payloads — spans carry topology, cost, and decisions;
    the row evidence stays in the loop's own event stream.
    """
    impl = _IMPLS.get(tool_name)
    if impl is None:
        raise InvalidToolArgsError(f"unknown_tool: {tool_name!r}")
    if not (
        ctx.settings.agent_trace_sessions and braintrust_tracing.is_enabled()
    ):
        return impl(ctx=ctx, args=args)
    with braintrust_tracing.trace_span(
        name=f"tool.{tool_name}",
        parent=ctx.trace_parent,
        input_=_args_digest(tool_name, args),
    ) as span:
        try:
            result = impl(ctx=ctx, args=args)
        except Exception as exc:
            # The span must close with an error status even when the
            # tool blows up — the loop upstream converts the exception
            # into a tool_error message for the model.
            span.log(output={"status": "tool_error", "message": str(exc)})
            raise
        span.log(
            output={
                "status": str(result.get("status", "ok")),
                **_result_digest(tool_name, result),
            }
        )
        return result


# ── Span digests ──
#
# Keep spans small: full SQL text and rationale are worth logging for
# run_sql; row payloads truncate to the first few rows; candidate lists
# reduce to identifiers + distance.

_SPAN_ROWS_CAP = 3


def _args_digest(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "run_sql":
        return {"sql": args.get("sql"), "rationale": args.get("rationale")}
    # The remaining tools take short queries and id lists only.
    return dict(args)


def _result_digest(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool_name in ("search_datasets", "search_columns"):
        candidates = result.get("candidates") or []
        keep = ("package_id", "column_name", "distance")
        return {
            "candidates": [
                {k: c.get(k) for k in keep if k in c}
                for c in candidates
                if isinstance(c, dict)
            ],
            "candidate_count": len(candidates),
        }
    if tool_name == "list_documents":
        documents = result.get("documents") or []
        return {
            "documents": [
                {
                    "document_id": d.get("document_id"),
                    "package_id": d.get("package_id"),
                    "row_count": d.get("row_count"),
                    "column_count": len(d.get("columns") or []),
                }
                for d in documents
                if isinstance(d, dict)
            ],
            "document_count": len(documents),
        }
    if tool_name == "sample_rows":
        rows = result.get("rows") or []
        return {"rows": rows[:_SPAN_ROWS_CAP], "row_count": len(rows)}
    if tool_name == "run_sql":
        digest = {
            k: result.get(k)
            for k in (
                "row_count",
                "bytes_billed",
                "elapsed_ms",
                "reason",
                "normalizations",
                "null_ratio_warning",
            )
            if k in result
        }
        rows = result.get("rows")
        if isinstance(rows, list):
            digest["rows"] = rows[:_SPAN_ROWS_CAP]
        return digest
    if tool_name == "describe_corpus":
        return {
            k: result.get(k)
            for k in (
                "packages",
                "documents_loaded",
                "rows_total",
                "latest_load_at",
            )
            if k in result
        }
    return {}


# ── run_sql normalization: table refs + JSONPath auto-quoting ──


def _mask_string_literals(sql: str) -> str:
    """Blank the contents of quoted string literals, preserving length.

    Positions line up 1:1 with the original, so regex spans found on the
    masked text can be applied back to the original. Keeps a `raw.rows`
    inside a string literal from ever being rewritten."""
    out = list(sql)
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n and sql[i] != quote:
                if sql[i] == "\\" and i + 1 < n:
                    out[i] = " "
                    i += 1
                out[i] = " "
                i += 1
        i += 1
    return "".join(out)


def _table_ref_pattern(settings: Settings) -> re.Pattern[str]:
    raw = re.escape(settings.bq_dataset_raw)
    rows = re.escape(settings.bq_rows_table)
    placeholder = r"(?i:<project(?:_id)?>|PROJECT_ID)"
    ref = (
        # `<project>.raw.rows` / PROJECT_ID.raw.rows — any backtick mix.
        rf"`?{placeholder}`?\.`?{raw}`?\.`?{rows}`?"
        # `raw.rows` (backticked, no project).
        rf"|`{raw}\.{rows}`"
        # bare raw.rows (no project; FROM/JOIN anchor below rules out a
        # project-qualified reference matching here).
        rf"|{raw}\.{rows}\b"
    )
    return re.compile(rf"(?i:\b(FROM|JOIN))\s+({ref})")


def normalize_table_references(
    sql: str, *, settings: Settings
) -> tuple[str, list[str]]:
    """Rewrite bare / placeholder `raw.rows` references after FROM and
    JOIN to the fully-qualified form. Returns `(sql, rewritten_refs)`.

    Scans a literal-masked copy so string literals are never touched.
    The guard still runs after this and stays the authority on which
    tables are allowed — normalization never widens what it accepts."""
    project_id = settings.gcp_project_id
    if not project_id:
        return sql, []
    canonical = (
        f"`{project_id}.{settings.bq_dataset_raw}.{settings.bq_rows_table}`"
    )
    masked = _mask_string_literals(sql)
    pieces: list[str] = []
    rewritten: list[str] = []
    last = 0
    for m in _table_ref_pattern(settings).finditer(masked):
        start, end = m.span(2)
        pieces.append(sql[last:start])
        pieces.append(canonical)
        rewritten.append(sql[start:end])
        last = end
    pieces.append(sql[last:])
    return "".join(pieces), rewritten


_JSON_FUNC_RE = re.compile(
    r"(?i)\b(?:JSON_VALUE|JSON_QUERY|JSON_EXTRACT|JSON_EXTRACT_SCALAR)\s*\("
)
_BARE_SEGMENT_OK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def autoquote_json_paths(sql: str) -> tuple[str, list[str]]:
    """Double-quote non-identifier segments in literal JSONPath args of
    JSON function calls. Returns `(sql, original_paths_that_changed)`.

    Bare segments like `2020-21_Expenditures` silently return NULL for
    every row; the correct rewrite is unambiguous, so the tool applies
    it rather than bouncing the call back to the model. Anything the
    scanner can't parse confidently is left untouched — the guard
    remains the rejection path for exotic shapes."""
    masked = _mask_string_literals(sql)
    replacements: list[tuple[int, int, str, str]] = []
    for m in _JSON_FUNC_RE.finditer(masked):
        span = _second_arg_span(sql, m.end() - 1)
        if span is None:
            continue
        start, end = span
        arg = sql[start:end]
        stripped = arg.strip()
        if (
            len(stripped) < 2
            or not stripped.startswith("'")
            or not stripped.endswith("'")
        ):
            continue  # not a string literal (parameter, concatenation)
        path = stripped[1:-1]
        if "'" in path or "\\" in path:
            continue  # escapes — not confidently parseable
        quoted = _quote_json_path(path)
        if quoted is None:
            continue
        lead = start + arg.index(stripped)
        replacements.append(
            (lead, lead + len(stripped), f"'{quoted}'", path)
        )
    if not replacements:
        return sql, []
    # A nested JSON call's path arg can sit before its enclosing call's
    # second arg — apply replacements in position order.
    replacements.sort(key=lambda r: r[0])
    pieces = []
    changed_paths = []
    last = 0
    for start, end, text, path in replacements:
        pieces.append(sql[last:start])
        pieces.append(text)
        changed_paths.append(path)
        last = end
    pieces.append(sql[last:])
    return "".join(pieces), changed_paths


def _second_arg_span(sql: str, open_paren: int) -> tuple[int, int] | None:
    """Span of a call's second top-level argument, or None when the call
    has fewer or more than exactly two arguments (the JSON functions we
    rewrite take exactly two) or runs off the end. Tracks paren depth
    and skips string literals."""
    i = open_paren + 1
    n = len(sql)
    depth = 1
    arg_index = 0
    arg_start = i
    while i < n:
        ch = sql[i]
        if ch == "'" or ch == '"':
            quote = ch
            i += 1
            while i < n and sql[i] != quote:
                if sql[i] == "\\":
                    i += 1
                i += 1
            if i >= n:
                return None
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return (arg_start, i) if arg_index == 1 else None
        elif ch == "," and depth == 1:
            if arg_index == 1:
                return None  # a third argument — leave it to the guard
            arg_index += 1
            arg_start = i + 1
        i += 1
    return None


def _quote_json_path(path: str) -> str | None:
    """Rewrite a JSONPath so every non-identifier segment is quoted.

    Returns None when nothing changed or the path can't be parsed
    confidently (array subscripts, embedded quotes, empty segments)."""
    if not path.startswith("$") or "[" in path:
        return None
    segments: list[str] = []
    changed = False
    i = 1
    n = len(path)
    while i < n:
        if path[i] != ".":
            return None
        i += 1
        if i < n and path[i] == '"':
            close = path.find('"', i + 1)
            if close == -1:
                return None
            segments.append(path[i : close + 1])
            i = close + 1
        else:
            j = i
            while j < n and path[j] != ".":
                if path[j] == '"':
                    return None
                j += 1
            segment = path[i:j]
            if not segment:
                return None
            if _BARE_SEGMENT_OK_RE.fullmatch(segment):
                segments.append(segment)
            else:
                segments.append(f'"{segment}"')
                changed = True
            i = j
    if not changed:
        return None
    return "$" + "".join(f".{s}" for s in segments)


# ── run_sql NULL-ratio advisory ──


_NULL_RATIO_MESSAGE = (
    "These columns are mostly NULL in the result. This usually means "
    "the referenced key does not semantically match the row bodies of "
    "the inlined documents — a mislabeled or wrong column, or the "
    "wrong document. Reconsider the column and document choice "
    "(list_documents sample_values help) before treating this as "
    "'no data'."
)


def compute_null_ratio_warning(
    *, sql: str, rows: list[dict[str, Any]], settings: Settings
) -> dict[str, Any] | None:
    """Advisory when result columns are mostly NULL. None when clean.

    Zero-row results never warn (that is a different signal), and
    neither do `document_id` nor the outputs of ungrouped aggregate
    functions — a NULL SUM over zero matching rows is visible to the
    model already."""
    if not rows:
        return None
    excluded = {"document_id"} | _scalar_aggregate_columns(sql)
    all_columns: dict[str, None] = {}
    for row in rows:
        for col in row:
            all_columns.setdefault(col, None)
    flagged: dict[str, float] = {}
    for col in all_columns:
        if col in excluded:
            continue
        nulls = sum(1 for row in rows if row.get(col) is None)
        ratio = nulls / len(rows)
        if ratio >= settings.agent_null_ratio_threshold:
            flagged[col] = round(ratio, 4)
    if not flagged:
        return None
    _LOG.info(
        "agent_null_ratio_warning_emitted",
        columns=sorted(flagged),
        row_count=len(rows),
    )
    return {"columns": flagged, "message": _NULL_RATIO_MESSAGE}


def _scalar_aggregate_columns(sql: str) -> set[str]:
    """Output names of aggregate expressions in GROUP-BY-less SELECTs.

    Only ungrouped aggregates can produce rows from zero inputs, so
    those are the columns the advisory must not flag. Grouped
    aggregates over zero rows produce zero rows — already excluded."""
    try:
        tree = sqlglot.parse_one(sql, dialect="bigquery")
    except Exception:
        return set()
    if tree is None:
        return set()
    names: set[str] = set()
    for select in tree.find_all(exp.Select):
        if select.args.get("group") is not None:
            continue
        for projection in select.expressions:
            if projection.find(exp.AggFunc) is None:
                continue
            name = projection.alias_or_name
            if name:
                names.add(name)
    return names


# ── list_documents quality flag ──


_GENERATED_COL_RE = re.compile(r"__col_\d+")


def _generated_header_ratio(columns: tuple[str, ...] | list[str]) -> float:
    if not columns:
        return 0.0
    generated = sum(
        1 for c in columns if _GENERATED_COL_RE.fullmatch(c) is not None
    )
    return generated / len(columns)


# ── Doc/column pairing check ──


# `$.<key>` — the top-level JSONPath key, either bare or double-quoted.
# Bare keys allow leading digits and hyphens here because we want to
# CATCH those references (the prompt tells the model to quote them,
# but this regex is the safety net for when the model doesn't). If a
# bare `$.2020-21_Foo` sneaks through we still want to know the model
# intended the key `2020-21_Foo` so we can pairing-check it.
_JSONPATH_TOP_KEY_RE = re.compile(
    r"""\$\.(?:"([^"]+)"|([A-Za-z0-9_][A-Za-z0-9_\-]*))"""
)

# Extract literal `document_id IN ('a', 'b', ...)` predicates. Only the
# literal shape counts — a subquery IN or a JOIN is already rejected
# by the sql_guard, so we don't need to defend against them here.
_DOC_IDS_IN_RE = re.compile(
    r"""document_id\s+IN\s*\(([^)]+)\)""", re.IGNORECASE
)
_ID_LITERAL_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _extract_json_path_columns(sql: str) -> set[str]:
    keys: set[str] = set()
    for m in _JSONPATH_TOP_KEY_RE.finditer(sql):
        key = m.group(1) or m.group(2)
        if key:
            keys.add(key)
    return keys


def _extract_inlined_document_ids(sql: str) -> set[str]:
    ids: set[str] = set()
    for m in _DOC_IDS_IN_RE.finditer(sql):
        inner = m.group(1)
        for id_m in _ID_LITERAL_RE.finditer(inner):
            ids.add(id_m.group(1))
    return ids


def check_doc_column_pairing(
    *, sql: str, state: LoopState
) -> tuple[list[dict[str, Any]], str | None]:
    """Verify every JSONPath column reference exists in every inlined doc.

    Returns `(violations, formatted_message)`. Empty violations = clean.

    Skips the check when the model hasn't populated `doc_columns` this
    turn (e.g. an inline retry that reuses a prior turn's known doc_id
    that we can't validate). The sql_guard's document_id filter check
    still rejects unknown-doc-shape queries; this check adds the
    column-level layer on top.
    """
    if not state.doc_columns:
        return [], None

    columns_referenced = _extract_json_path_columns(sql)
    doc_ids_referenced = _extract_inlined_document_ids(sql)

    if not columns_referenced or not doc_ids_referenced:
        return [], None

    violations: list[dict[str, Any]] = []
    for doc_id in sorted(doc_ids_referenced):
        available = state.doc_columns.get(doc_id)
        if available is None:
            continue
        available_set = set(available)
        for col in sorted(columns_referenced):
            if col in available_set:
                continue
            other_docs = sorted(
                d for d, cols in state.doc_columns.items() if col in cols
            )
            violations.append(
                {
                    "column": col,
                    "doc_id": doc_id,
                    "available_columns": list(available),
                    "other_docs_with_column": other_docs,
                }
            )
    if not violations:
        return [], None

    lines = [
        "doc_column_pairing_violation: the SQL references columns that "
        "do not exist in the document(s) it inlined. This is the most "
        "common cause of an all-NULL result."
    ]
    for v in violations:
        lines.append(
            f"  - Column '{v['column']}' is NOT in document '{v['doc_id']}'. "
            f"That doc's columns are: {v['available_columns']}."
        )
        if v["other_docs_with_column"]:
            lines.append(
                "    The column DOES exist in these documents: "
                f"{v['other_docs_with_column']}."
            )
        else:
            lines.append(
                "    The column does not exist in any document from "
                "list_documents this turn. If you meant this string as "
                "a VALUE (e.g. an organization name), it is likely a "
                "value in one of the doc's own text columns "
                "(Organization / Description / Name-like). Use "
                "`JSON_VALUE(..., '$.Organization') = '<the string>'` "
                "instead of referencing it as a column."
            )
    lines.append(
        "Fix: either inline a different document_id from list_documents "
        "that contains the columns you need, or drop the missing column "
        "reference. Do not retry with the same doc/column pairing."
    )
    return violations, "\n".join(lines)


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
SELECT
  r.document_id,
  r.row_index,
  r.row AS row
FROM `{project_id}.{dataset}.{rows_table}` AS r
JOIN `{project_id}.{dataset}.{documents_table}` AS d USING (document_id)
WHERE d.package_id = @pkg
LIMIT @n
""".strip()


# Re-exported so the loop and tests can reach it without importing
# `time` and asserting perf inline. Currently only used to defensively
# stamp tool durations in future work.
_MONOTONIC = time.monotonic
