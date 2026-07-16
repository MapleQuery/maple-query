"""Loop-side enforcement of the weak-retrieval reformulation policy.

The prompt asks for the right behaviour; these tests pin that the v2
research loop enforces it: the reformulated retry after a weak search
rides free of the tool budget and emits a `reformulation` event, the
duplicate-query guard replays from cache and charges, and the whole
policy is inert when the first retrieval is strong.
"""
from __future__ import annotations

import json
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
    *, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient
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


def _request(question: str = "immigration processing time?") -> ChatRequest:
    return ChatRequest(conversation_id="c1", history=[], question=question)


def _package_row(
    *, package_id: str, title: str, distance: float
) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "title": title,
        "summary": "s",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": distance,
    }


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "search_datasets",
        "arguments": {"query": query},
    }


def _tool_messages(openai: FakeOpenAIClient) -> list[dict[str, Any]]:
    return [
        m
        for m in openai.chat_calls[-1]["messages"]
        if m.get("role") == "tool"
    ]


def test_weak_search_retry_emits_reformulation_and_is_not_charged() -> None:
    # Budget of 1: the failed attempt spends it entirely; the retry
    # must still execute (rule: the retry is never priced out).
    settings = _settings(agent_max_tool_calls=1)
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [_package_row(package_id="p1", title="Weak Hit", distance=0.8)],
    )
    bq.register_query(
        "VECTOR_SEARCH",
        [_package_row(package_id="p2", title="Strong Hit", distance=0.1)],
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "immigration PR times")]},
            {
                "tool_calls": [
                    _search_call(
                        "c2", "permanent resident application processing"
                    )
                ]
            },
            {"content": "answer from [Strong Hit](/datasets/p2)."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    reformulations = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.Reformulation)
    ]
    assert len(reformulations) == 1
    assert reformulations[0].original_query == "immigration PR times"
    assert (
        reformulations[0].reformulated_query
        == "permanent resident application processing"
    )
    assert reformulations[0].top_similarity_before == 0.2
    # Both searches executed…
    kinds = [e.event_type for e in outcome.events]
    assert kinds.count("datasets_ranked") == 2
    # …but only the first was billed, and no budget refusal happened.
    assert outcome.tool_call_count == 1
    assert "budget_exceeded" not in kinds
    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert records[0].record["reformulations_used"] == 1


def test_duplicate_query_short_circuits_from_cache_and_is_charged() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [_package_row(package_id="p1", title="Weak Hit", distance=0.8)],
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "immigration PR times")]},
            # Same query modulo case/whitespace → duplicate guard.
            {"tool_calls": [_search_call("c2", "Immigration  PR Times")]},
            {"content": "giving up."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    # No reformulation credit for an identical query.
    assert not any(
        isinstance(e, agent_events.Reformulation) for e in outcome.events
    )
    # Replayed from cache: one retrieval, but both calls billed.
    kinds = [e.event_type for e in outcome.events]
    assert kinds.count("datasets_ranked") == 1
    assert outcome.tool_call_count == 2
    duplicate_msgs = [
        m
        for m in _tool_messages(openai)
        if "identical query" in m["content"]
    ]
    assert len(duplicate_msgs) == 1
    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert records[0].record["reformulations_used"] == 0


def test_cap_reached_guidance_switches_to_clarify_steer() -> None:
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query(
            "VECTOR_SEARCH",
            [_package_row(package_id="p1", title="Weak", distance=0.9)],
        )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "query one")]},
            {"tool_calls": [_search_call("c2", "query two")]},
            {"content": "Which IRCC program do you mean?"},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    run_turn_collected(request=_request(), deps=deps)

    tool_contents = [
        json.loads(m["content"]) for m in _tool_messages(openai)
    ]
    # First weak result: reformulate guidance. Second (the retry, cap
    # now spent): the steer to clarify-or-caveat, no more searching.
    assert "Reformulate once" in tool_contents[0]["guidance"]
    assert "Do not search again" in tool_contents[1]["guidance"]
    assert "ONE clarifying question" in tool_contents[1]["guidance"]


def test_strong_first_search_leaves_policy_inert() -> None:
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query(
            "VECTOR_SEARCH",
            [_package_row(package_id="p1", title="Strong", distance=0.1)],
        )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "housing starts")]},
            {"tool_calls": [_search_call("c2", "housing completions")]},
            {"content": "answer citing [Strong](/datasets/p1)."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert not any(
        isinstance(e, agent_events.Reformulation) for e in outcome.events
    )
    # A second search without a weak signal is an ordinary billed call.
    assert outcome.tool_call_count == 2
    for m in _tool_messages(openai):
        payload = json.loads(m["content"])
        assert payload["retrieval_quality"] == "ok"
        assert "guidance" not in payload
