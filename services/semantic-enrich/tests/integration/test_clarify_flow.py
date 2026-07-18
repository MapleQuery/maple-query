"""The clarify-instead-of-surrender branch, across two turns.

A clarifying question is a plain final message; the pipeline tags the
turn record `outcome: "clarified"` with the searches tried, and the
follow-up turn (the user's answer, carrying that record back via
`ChatRequest.turn_records`) gets the failed phrasings as a system
hint and loses the option to ask a second consecutive question.
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


def _weak_row() -> dict[str, Any]:
    return {
        "package_id": "p1",
        "title": "Closest Candidate",
        "summary": "s",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": 0.9,
    }


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "search_datasets",
        "arguments": {"query": query},
    }


def _run_clarify_turn(
    *, turn_records: list[dict[str, Any]] | None = None
) -> tuple[Any, FakeOpenAIClient]:
    """One turn that searches twice with weak results and ends in a
    clarifying question."""
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [_weak_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "immigration PR times")]},
            {"tool_calls": [_search_call("c2", "IRCC processing time")]},
            {
                "content": (
                    "Which program are you asking about — permanent "
                    "residence, citizenship, or work permits?"
                )
            },
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    request = ChatRequest(
        conversation_id="c1",
        history=[],
        question="How fast is immigration processing?",
        turn_records=list(turn_records or []),
    )
    return run_turn_collected(request=request, deps=deps), openai


def _record(outcome: Any) -> dict[str, Any]:
    events = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(events) == 1
    return events[0].record


def test_cap_reached_turn_ends_in_clarify_record() -> None:
    outcome, _ = _run_clarify_turn()
    record = _record(outcome)
    assert record["outcome"] == "clarified"
    assert [s["query"] for s in record["searches_tried"]] == [
        "immigration PR times",
        "IRCC processing time",
    ]
    assert all(
        s["retrieval_quality"] == "weak" for s in record["searches_tried"]
    )


def test_answered_turn_is_not_tagged_clarify() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [{**_weak_row(), "distance": 0.1, "title": "Strong"}],
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "housing starts")]},
            {"content": "answer citing [Strong](/datasets/p1)."},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="housing?"
        ),
        deps=deps,
    )
    # No successful SQL in this scripted turn → a no-data claim, but
    # decidedly not a clarify.
    assert _record(outcome)["outcome"] == "no_data"


def test_followup_turn_receives_failed_search_hint() -> None:
    first, _ = _run_clarify_turn()
    _followup, openai = _run_clarify_turn(turn_records=[_record(first)])

    system_msgs = [
        m
        for m in openai.chat_calls[0]["messages"]
        if m.get("role") == "system"
    ]
    hints = [m for m in system_msgs if "previous turn" in m["content"]]
    assert len(hints) == 1
    assert "immigration PR times" in hints[0]["content"]
    assert "IRCC processing time" in hints[0]["content"]
    assert "Do not repeat" in hints[0]["content"]


def test_second_consecutive_clarify_is_suppressed() -> None:
    first, _ = _run_clarify_turn()
    _followup, openai = _run_clarify_turn(turn_records=[_record(first)])

    # The cap-reached guidance in the follow-up turn must drop the
    # clarify option and require a best-effort caveated answer.
    tool_contents = [
        json.loads(m["content"])
        for m in openai.chat_calls[-1]["messages"]
        if m.get("role") == "tool"
    ]
    steers = [
        c["guidance"]
        for c in tool_contents
        if "Do not search again" in c.get("guidance", "")
    ]
    assert steers
    for guidance in steers:
        assert "do not ask another clarifying question" in guidance
        assert "ONE clarifying question" not in guidance
