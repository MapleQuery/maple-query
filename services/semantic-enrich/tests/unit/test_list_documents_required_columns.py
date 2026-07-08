"""`required_columns` filter on list_documents.

Replaces the "compute the SET INTERSECTION in your head" prompt prose:
returned docs are guaranteed safe to inline together for the listed
columns.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def _bq_with_docs() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-A",
                "package_id": "pkg-1",
                "title": "A",
                "row_count": 10,
                "resource_last_modified": None,
            },
            {
                "document_id": "doc-B",
                "package_id": "pkg-1",
                "title": "B",
                "row_count": 20,
                "resource_last_modified": None,
            },
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [
            {
                "document_id": "doc-A",
                "columns": ["FISCAL_YEAR", "Amount", "Org"],
            },
            {"document_id": "doc-B", "columns": ["Amount", "Description"]},
        ],
    )
    return bq


def _ctx(
    bq: FakeBqClient,
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    state.known_package_ids.add("pkg-1")
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=_settings(),
        state=state,
        emit=events.append,
    )
    return ctx, events


def test_filter_keeps_only_satisfying_docs() -> None:
    ctx, events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["FISCAL_YEAR", "Amount"],
        },
    )
    assert [d["document_id"] for d in result["documents"]] == ["doc-A"]
    assert result["filtered_out"] == [
        {"document_id": "doc-B", "missing_columns": ["FISCAL_YEAR"]}
    ]
    assert "required_columns_unsatisfiable" not in result
    listed = [e for e in events if e.event_type == "documents_listed"]
    assert isinstance(listed[0], agent_events.DocumentsListed)
    assert listed[0].filtered_out == result["filtered_out"]


def test_all_docs_satisfying_omits_filtered_out() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": ["Amount"]},
    )
    assert len(result["documents"]) == 2
    assert "filtered_out" not in result


def test_unsatisfiable_returns_full_list_with_flag() -> None:
    """An empty result would push the model toward surrender — return
    the unfiltered listing plus the flag instead."""
    ctx, events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["NOT_A_COLUMN"],
        },
    )
    assert result["required_columns_unsatisfiable"] is True
    assert [d["document_id"] for d in result["documents"]] == [
        "doc-A",
        "doc-B",
    ]
    assert "filtered_out" not in result
    listed = [e for e in events if e.event_type == "documents_listed"]
    assert isinstance(listed[0], agent_events.DocumentsListed)
    assert listed[0].required_columns_unsatisfiable is True
    assert listed[0].filtered_out is None


def test_empty_required_columns_list_is_a_noop() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": []},
    )
    assert len(result["documents"]) == 2
    assert "filtered_out" not in result
    assert "required_columns_unsatisfiable" not in result


def test_invalid_required_columns_type_rejected() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(
            ctx=ctx,
            args={"package_ids": ["pkg-1"], "required_columns": [1, 2]},
        )


def test_package_ids_bounds_still_enforced() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(ctx=ctx, args={"package_ids": []})
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(
            ctx=ctx, args={"package_ids": [f"pkg-{i}" for i in range(11)]}
        )


def test_filtered_out_docs_still_tracked_for_pairing_check() -> None:
    """A doc the filter removed stays in state so run_sql can still
    pairing-check it if the model inlines it anyway."""
    ctx, _events = _ctx(_bq_with_docs())
    agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["FISCAL_YEAR"],
        },
    )
    assert "doc-B" in ctx.state.known_document_ids
    assert ctx.state.doc_columns["doc-B"] == ["Amount", "Description"]


def test_schema_declares_required_columns_param() -> None:
    schemas = {
        s["function"]["name"]: s["function"] for s in agent_tools.tool_schemas()
    }
    props: dict[str, Any] = schemas["list_documents"]["parameters"][
        "properties"
    ]
    assert "required_columns" in props
    # Optional — not in required.
    assert "required_columns" not in (
        schemas["list_documents"]["parameters"]["required"]
    )
