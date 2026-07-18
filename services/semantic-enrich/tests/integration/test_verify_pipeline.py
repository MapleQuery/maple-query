"""The verify phase driven through the v2 pipeline.

The headline flow: first candidate misses a dimension → checker says
retry → research re-enters with the gap hint → second candidate ships
caveated. Plus the skip conditions (clarify candidates and
budget-forced answers are never checked) and shadow-mode transparency
end to end.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent.verify import AnswerFitVerifier
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    overrides.setdefault("agent_verify_mode", "act")
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
        verifier=AnswerFitVerifier.from_settings(settings),
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _strong_row(title: str = "Housing Grants") -> dict[str, Any]:
    return {
        "package_id": "pkg-1",
        "title": title,
        "summary": "s",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": 0.1,
    }


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "search_datasets",
        "arguments": {"query": query},
    }


def _request(question: str = "grants across provinces?") -> ChatRequest:
    return ChatRequest(conversation_id="c1", history=[], question=question)


def _record_of(outcome: Any) -> dict[str, Any]:
    events = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(events) == 1
    return events[0].record


def test_retry_then_caveat_flow() -> None:
    settings = _settings()
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [_strong_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            # research leg 1
            {"tool_calls": [_search_call("c1", "housing grants")]},
            {"content": "national total was $X."},
            # research leg 2 (verify retry)
            {"tool_calls": [_search_call("c2", "provincial housing grants")]},
            {"content": "per-province totals were $Y."},
        ],
        structured_responses=[
            {
                "fits": False,
                "confidence": 0.95,
                "gap": "per-province breakdown",
                "action": "retry",
                "retry_hint": "provincial grant columns",
            },
            {
                "fits": False,
                "confidence": 0.9,
                "gap": "the territories",
                "action": "caveat",
                "retry_hint": None,
            },
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    # Second candidate shipped, under the second check's caveat.
    assert outcome.final_message == (
        "**Partial answer:** this does not cover the territories.\n\n"
        "per-province totals were $Y."
    )
    verifications = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.Verification)
    ]
    assert [v.action for v in verifications] == ["retry", "caveat"]
    assert all(v.enforced for v in verifications)
    # Research phase ran twice; the retry leg saw the gap hint.
    kinds = [e.event_type for e in outcome.events]
    assert kinds.count("phase_start") >= 4  # triage, memory, research x2…
    retry_leg_messages = openai.chat_calls[-1]["messages"]
    hints = [
        m
        for m in retry_leg_messages
        if m.get("role") == "system"
        and "Your previous answer missed: per-province breakdown" in m["content"]
    ]
    assert len(hints) == 1
    record = _record_of(outcome)
    assert record["outcome"] == "answered_with_caveat"
    assert record["verify_retries_used"] == 1
    # Both legs' tool calls billed normally (strong retrieval, no
    # reformulation credit).
    assert outcome.tool_call_count == 2


def test_clarify_candidate_skips_verification() -> None:
    # A cap-steered clarifying question is not a claim to check: the
    # checker must not run at all.
    settings = _settings()
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query(
            "VECTOR_SEARCH", [{**_strong_row("Weak"), "distance": 0.9}]
        )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "query one")]},
            {"tool_calls": [_search_call("c2", "query two")]},
            {"content": "Which program do you mean?"},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert openai.structured_calls == []
    assert not any(
        isinstance(e, agent_events.Verification) for e in outcome.events
    )
    assert _record_of(outcome)["outcome"] == "clarify"


def test_budget_forced_answer_skips_verification() -> None:
    settings = _settings(agent_max_tool_calls=1)
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [_strong_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "x")]},
            {"tool_calls": [_search_call("c2", "y")]},
            {"content": "best effort from what I have."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert openai.structured_calls == []
    assert outcome.final_message == "best effort from what I have."


def test_log_mode_ships_unchanged_with_shadow_event() -> None:
    settings = _settings(agent_verify_mode="log")
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [_strong_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "housing grants")]},
            {"content": "national total was $X."},
        ],
        structured_responses=[
            {
                "fits": False,
                "confidence": 0.95,
                "gap": "per-province breakdown",
                "action": "retry",
                "retry_hint": None,
            }
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert outcome.final_message == "national total was $X."
    verifications = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.Verification)
    ]
    assert len(verifications) == 1
    assert verifications[0].enforced is False
    # No retry leg in shadow mode.
    assert _record_of(outcome)["verify_retries_used"] == 0
    assert _record_of(outcome)["outcome"] == "answered"
