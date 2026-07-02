"""Argument validation + basic dispatch for the four agent tools."""
from __future__ import annotations

import math
from typing import Any

import pytest

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def _ctx(
    *,
    bq: FakeBqClient | None = None,
    openai_client: FakeOpenAIClient | None = None,
    question: str = "what did we spend on housing?",
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    bq = bq or FakeBqClient()
    openai_client = openai_client or FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
    )
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question=question
    )
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=openai_client,
        settings=_settings(),
        state=state,
        emit=events.append,
    )
    return ctx, events


def test_tool_schemas_shape() -> None:
    schemas = agent_tools.tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == list(agent_tools.TOOL_NAMES)
    for s in schemas:
        params = s["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert "required" in params


def test_search_datasets_populates_known_ids() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "summary": "housing",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.1,
            }
        ],
    )
    ctx, events = _ctx(bq=bq)
    result = agent_tools.run_search_datasets(
        ctx=ctx, args={"query": "housing", "k": 3}
    )
    assert result["candidates"][0]["package_id"] == "pkg-1"
    assert "pkg-1" in ctx.state.known_package_ids
    kinds = [e.event_type for e in events]
    assert "retrieval_started" in kinds
    assert "datasets_ranked" in kinds


def test_search_columns_rejects_unknown_package_id() -> None:
    ctx, _ = _ctx()
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_search_columns(
            ctx=ctx,
            args={"package_ids": ["never-seen"], "query": "spend"},
        )


def test_search_columns_accepts_known_package_id() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "column_name": "TOT_EXP",
                "semantic_type": "currency",
                "description": "spend",
                "sample_values": [],
                "distance": 0.15,
            }
        ],
    )
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_search_columns(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "query": "spend"},
    )
    assert result["candidates"][0]["column_name"] == "TOT_EXP"


def test_sample_rows_shape() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "raw.rows",
        [
            {
                "document_id": "doc-1",
                "row_index": 0,
                "row": {"Amount": 1},
            },
            {
                "document_id": "doc-1",
                "row_index": 1,
                "row": {"Amount": 2},
            },
        ],
    )
    ctx, events = _ctx(bq=bq)
    result = agent_tools.run_sample_rows(
        ctx=ctx, args={"package_id": "pkg-1", "n": 2}
    )
    assert result["package_id"] == "pkg-1"
    assert result["rows"][0]["document_id"] == "doc-1"
    assert result["rows"][0]["row"] == {"Amount": 1}
    assert any(e.event_type == "sample_rows" for e in events)


def test_list_documents_rejects_unknown_package_id() -> None:
    ctx, _ = _ctx()
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(
            ctx=ctx, args={"package_ids": ["nope"]}
        )


def test_list_documents_returns_docs_and_columns() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "Housing 2020",
                "row_count": 42,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [{"document_id": "doc-1", "columns": ["Amount", "Organization"]}],
    )
    ctx, events = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    doc = result["documents"][0]
    assert doc["document_id"] == "doc-1"
    assert doc["columns"] == ["Amount", "Organization"]
    assert "doc-1" in ctx.state.known_document_ids
    assert any(e.event_type == "documents_listed" for e in events)


def test_run_sql_guard_rejection_becomes_tool_result() -> None:
    bq = FakeBqClient()
    ctx, events = _ctx(bq=bq)
    # Non-select statement — guard rejects on forbidden keyword.
    args: dict[str, Any] = {
        "sql": "INSERT INTO foo VALUES (1)",
        "rationale": "malicious",
    }
    result = agent_tools.run_run_sql(ctx=ctx, args=args)
    assert result["status"] == "guard_rejected"
    assert "reason" in result
    assert any(e.event_type == "sql_guarded" for e in events)
    # Failed guard does not consume a SQL execution.
    assert ctx.state.sql_execution_count == 0


def test_run_sql_success_emits_sql_executed() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"n": 1}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    ctx, events = _ctx(bq=bq)
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": (
                "SELECT 1 AS n FROM `proj.raw.rows` "
                "WHERE document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "test",
        },
    )
    assert result["status"] == "ok"
    assert result["row_count"] == 1
    kinds = [e.event_type for e in events]
    assert "sql_guarded" in kinds
    assert "sql_executed" in kinds
    assert "rows" in kinds
    assert ctx.state.sql_execution_count == 1


def test_run_sql_budget_exceeded_short_circuits() -> None:
    ctx, events = _ctx()
    ctx.state.sql_execution_count = ctx.settings.agent_max_sql_executions
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": "SELECT 1 AS n FROM `proj.raw.rows` WHERE document_id IN ('d') LIMIT 10",
            "rationale": "test",
        },
    )
    assert result["status"] == "budget_exceeded"
    # No guard event because we short-circuited.
    assert not any(e.event_type == "sql_guarded" for e in events)


def test_dispatch_unknown_tool_raises() -> None:
    ctx, _ = _ctx()
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.dispatch(ctx=ctx, tool_name="nope", args={})


def test_dispatch_routes_to_impl() -> None:
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    ctx, _ = _ctx(bq=bq)
    result = agent_tools.dispatch(
        ctx=ctx,
        tool_name="search_datasets",
        args={"query": "housing"},
    )
    assert result == {"candidates": []}
