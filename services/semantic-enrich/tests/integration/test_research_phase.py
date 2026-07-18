"""The extracted research phase, driven through the v2 pipeline.

Port of the v1 loop integration bar: canned tool_calls → final
message, tool errors surfaced as tool messages (not crashes), batch
concurrency semantics, and the trace capture the later phases consume.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _deps(
    *,
    settings: Settings,
    bq: FakeBqClient,
    openai: FakeOpenAIClient,
) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-v2-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _request(question: str = "how much housing spend?") -> ChatRequest:
    return ChatRequest(conversation_id="c1", history=[], question=question)


def test_tool_call_then_answer_emits_v1_event_shape() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "title": "Housing Starts",
                "summary": "housing",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.1,
            }
        ],
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "search_datasets",
                        "arguments": {"query": "housing spend", "k": 3},
                    }
                ],
            },
            {"content": "Housing spend was $X ([Housing Starts](/datasets/pkg-1))."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert outcome.final_message.startswith("Housing spend")
    kinds = [e.event_type for e in outcome.events]
    assert "retrieval_started" in kinds
    assert "datasets_ranked" in kinds
    assert "message_delta" in kinds
    assert kinds[-1] == "done"
    assert outcome.tool_call_count == 1
    # Retrieval confidence captured for the later phases.
    ranked = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.DatasetsRanked)
    ]
    assert ranked[0].top_similarity == 0.9


def test_tool_error_comes_back_as_tool_message_and_loop_continues() -> None:
    bq = FakeBqClient()
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "bad_1",
                        "name": "search_datasets",
                        # Missing required "query" → InvalidToolArgsError.
                        "arguments": {"k": 3},
                    }
                ],
            },
            {"content": "Could not search; nothing to report."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert any(
        isinstance(e, agent_events.ToolError) for e in outcome.events
    )
    assert outcome.events[-1].event_type == "done"
    # The model saw a tool_error result, not silence.
    final_messages = openai.chat_calls[-1]["messages"]
    errors = [
        m
        for m in final_messages
        if m.get("role") == "tool" and "tool_error" in m["content"]
    ]
    assert len(errors) == 1


def test_parallel_batch_events_stay_grouped_per_call() -> None:
    settings = _settings(agent_parallel_tool_calls=3)
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "x"},
                    },
                    {
                        "id": "c2",
                        "name": "search_datasets",
                        "arguments": {"query": "y"},
                    },
                ]
            },
            {"content": "done searching."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert outcome.tool_call_count == 2
    kinds = [e.event_type for e in outcome.events]
    # Both calls' event pairs present, grouped (started before ranked).
    assert kinds.count("retrieval_started") == 2
    assert kinds.count("datasets_ranked") == 2
    first_started = kinds.index("retrieval_started")
    assert kinds[first_started + 1] == "datasets_ranked"
    assert kinds[-1] == "done"


def test_trace_lands_in_turn_record() -> None:
    """packages researched via list_documents and columns referenced in
    run_sql surface in the turn_record event."""
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "title": "Housing Starts",
                "summary": "housing",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.2,
            }
        ],
    )
    bq.register_query(
        "FROM `proj.raw.documents`",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "source_url": "http://x",
                "row_count": 10,
                "columns": ["FISCAL_YEAR", "Amount"],
            }
        ],
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "housing"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "list_documents",
                        "arguments": {"package_ids": ["pkg-1"]},
                    }
                ]
            },
            {"content": "answer citing [Housing Starts](/datasets/pkg-1)."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(records) == 1
    assert records[0].record["packages"] == [
        {"package_id": "pkg-1", "title": "Housing Starts"}
    ]
    assert outcome.events[-1].event_type == "done"
