"""End-to-end agent loop with FakeOpenAI + FakeBq.

Covers the happy path (tool call → SQL → final answer), the budget
path (loop forever until budget_exceeded), and the cache path (same
question twice → replay).
"""
from __future__ import annotations

import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import (
    ChatRequest,
    LoopDeps,
    load_system_prompt,
    run_turn_collected,
)
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
    )


def _prompt_bits(settings: Settings) -> tuple[str, str]:
    return load_system_prompt(settings.agent_system_prompt_path, settings)


def _deps(
    *, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient
) -> LoopDeps:
    prompt, prompt_hash = _prompt_bits(settings)
    return LoopDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt=prompt,
        prompt_hash=prompt_hash,
        cache=ResponseCache(
            max_entries=10,
            max_value_bytes=1_000_000,
            ttl_seconds=60,
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )


def test_prompt_template_renders_and_hashes() -> None:
    settings = _settings()
    prompt, digest = _prompt_bits(settings)
    assert "MapleQuery" in prompt
    assert "raw.rows" in prompt
    # The pairing rule is enforced tool-side now (run_sql pairing check
    # + list_documents required_columns); the prompt keeps a one-liner
    # pointing at required_columns instead of the old HARD RULE prose.
    assert "required_columns" in prompt
    assert "document_id" in prompt
    assert len(digest) == 64


def test_happy_path_tool_call_then_answer() -> None:
    settings = _settings()
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
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
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
            {"content": "Housing spend for 2020 was $X (pkg-1)."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    request = ChatRequest(
        conversation_id="c1",
        history=[],
        question="how much did we spend on housing?",
    )
    outcome = run_turn_collected(request=request, deps=deps)
    assert outcome.final_message.startswith("Housing spend")
    kinds = [e.event_type for e in outcome.events]
    assert kinds[0] == "turn_start"
    assert "retrieval_started" in kinds
    assert "datasets_ranked" in kinds
    assert "message_delta" in kinds
    assert kinds[-1] == "done"
    assert outcome.tool_call_count == 1


def test_budget_exceeded_forces_final_answer() -> None:
    settings = _settings().model_copy(
        update={"agent_max_tool_calls": 2}
    )
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    bq.register_query("VECTOR_SEARCH", [])
    bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            # Turn 1: model asks for a dataset search.
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "x"},
                    }
                ]
            },
            # Turn 2: another search — hits budget on next round.
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "search_datasets",
                        "arguments": {"query": "y"},
                    }
                ]
            },
            # Turn 3: model tries a third — should be blocked by budget.
            {
                "tool_calls": [
                    {
                        "id": "c3",
                        "name": "search_datasets",
                        "arguments": {"query": "z"},
                    }
                ]
            },
            # Turn 4: forced to answer after budget_exceeded.
            {"content": "I hit my budget; here's the best I've got."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="pick something"
        ),
        deps=deps,
    )
    types = [e.event_type for e in outcome.events]
    assert "budget_exceeded" in types
    assert types[-1] == "done"


def test_cache_hit_on_second_run() -> None:
    settings = _settings()
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {"content": "no data available."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    req = ChatRequest(
        conversation_id="c1", history=[], question="what happened?"
    )
    first = run_turn_collected(request=req, deps=deps)
    assert first.cache_hit is False

    # Second run — cache hit; no additional OpenAI calls.
    calls_before = len(openai.chat_calls)
    second = run_turn_collected(request=req, deps=deps)
    assert second.cache_hit is True
    assert any(
        isinstance(e, agent_events.CacheHit) for e in second.events
    )
    assert len(openai.chat_calls) == calls_before


def test_invalid_history_returns_error_event() -> None:
    settings = _settings()
    bq = FakeBqClient()
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[{"role": "wizard", "content": "bad"}],
            question="anything",
        ),
        deps=deps,
    )
    assert any(
        isinstance(e, agent_events.ErrorEvent)
        and e.reason == "invalid_history"
        for e in outcome.events
    )


def test_prompt_template_file_exists() -> None:
    settings = _settings()
    path = settings.agent_system_prompt_path
    assert isinstance(path, Path)
    assert path.exists(), f"agent prompt template missing at {path}"


def _assert_every_tool_call_answered(openai: FakeOpenAIClient) -> None:
    """The OpenAI-protocol invariant: in every request's message list,
    each assistant tool_calls id must be followed by a tool message
    with that tool_call_id — the API 400s the whole request otherwise
    (the turn-crash failure the 2026-07 fixture runs hit)."""
    for call in openai.chat_calls:
        messages = call["messages"]
        answered = {
            m.get("tool_call_id")
            for m in messages
            if m.get("role") == "tool"
        }
        for m in messages:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                assert tc["id"] in answered, (
                    f"tool_call_id {tc['id']!r} has no tool response"
                )


def test_parallel_batch_straddling_budget_answers_every_call() -> None:
    """A parallel batch bigger than the remaining budget must refuse
    the overflow calls with tool messages, not drop them."""
    settings = _settings().model_copy(update={"agent_max_tool_calls": 2})
    bq = FakeBqClient()
    for _ in range(3):
        bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            # One parallel batch of 3 against a budget of 2.
            {
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "name": "search_datasets",
                        "arguments": {"query": q},
                    }
                    for i, q in enumerate(("x", "y", "z"))
                ]
            },
            {"content": "Best effort with two searches."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="pick something"
        ),
        deps=deps,
    )
    types = [e.event_type for e in outcome.events]
    assert "budget_exceeded" in types
    assert types[-1] == "done"
    assert outcome.tool_call_count == 2
    _assert_every_tool_call_answered(openai)
    # The refused third call got an explicit budget refusal.
    final_messages = openai.chat_calls[-1]["messages"]
    refusals = [
        m
        for m in final_messages
        if m.get("role") == "tool"
        and m.get("tool_call_id") == "c2"
        and "budget_exceeded" in m["content"]
    ]
    assert len(refusals) == 1


def test_budget_block_after_cap_answers_every_call() -> None:
    """Tool calls requested after the cap is already spent must get
    budget-refusal tool messages before the forced-answer nudge."""
    settings = _settings().model_copy(update={"agent_max_tool_calls": 1})
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "x"},
                    }
                ]
            },
            # Cap is spent; this request must be refused, not dropped.
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "search_datasets",
                        "arguments": {"query": "y"},
                    }
                ]
            },
            {"content": "Answering from what I have."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="pick something"
        ),
        deps=deps,
    )
    types = [e.event_type for e in outcome.events]
    assert "budget_exceeded" in types
    assert types[-1] == "done"
    _assert_every_tool_call_answered(openai)
