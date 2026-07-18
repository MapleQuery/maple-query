"""Turn budgets under the v2 orchestrator.

The meters live on `TurnContext`, so a verify-triggered research
retry pays into the same tool-call budget as the first pass, and the
wall-clock timeout inside the research phase produces the v1
`turn_timeout` → `error` terminal sequence.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
    Verdict,
)
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


class RetryOnceVerifier:
    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict:
        return Verdict(action="accept" if final else "retry")


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
    verifier: Any = None,
) -> PipelineDeps:
    deps = PipelineDeps(
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
    if verifier is not None:
        deps.verifier = verifier
    return deps


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "search_datasets",
        "arguments": {"query": query},
    }


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _strong_row() -> dict[str, Any]:
    """A confident hit: keeps retrieval_quality "ok" so the
    reformulation policy stays inert and searches bill normally —
    these tests are about the budget, not weak retrieval."""
    return {
        "package_id": "pkg-1",
        "title": "Strong Hit",
        "summary": "s",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": 0.1,
    }


def test_tool_budget_is_shared_across_verify_retry() -> None:
    """First research pass spends 2 of 3 tool calls; the retry leg asks
    for 2 more — only 1 slot remains, so the overflow call is refused
    and budget_exceeded fires in the retry leg."""
    settings = _settings(agent_max_tool_calls=3)
    bq = FakeBqClient()
    for _ in range(3):
        bq.register_query("VECTOR_SEARCH", [_strong_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            # research pass 1: batch of two searches, then a draft.
            {"tool_calls": [_search_call("c1", "x"), _search_call("c2", "y")]},
            {"content": "draft answer."},
            # research pass 2 (verify retry): two more searches — only
            # one slot left in the shared budget.
            {"tool_calls": [_search_call("c3", "z"), _search_call("c4", "w")]},
            {"content": "final answer."},
        ],
    )
    deps = _deps(
        settings=settings, bq=bq, openai=openai,
        verifier=RetryOnceVerifier(),
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="q?"
        ),
        deps=deps,
    )

    assert outcome.final_message == "final answer."
    assert outcome.tool_call_count == 3  # never exceeds the global cap
    over = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.BudgetExceeded)
    ]
    assert len(over) == 1
    assert over[0].which == "tool_calls"
    # The refused call still got a budget-refusal tool message so the
    # OpenAI protocol invariant holds.
    final_messages = openai.chat_calls[-1]["messages"]
    refusals = [
        m
        for m in final_messages
        if m.get("role") == "tool"
        and m.get("tool_call_id") == "c4"
        and "budget_exceeded" in m["content"]
    ]
    assert len(refusals) == 1
    types = [e.event_type for e in outcome.events]
    assert types[-1] == "done"
    assert types.count("done") == 1


def test_budget_block_forces_final_answer() -> None:
    """Cap already spent → next batch is refused wholesale and the
    model is nudged into a forced final answer (v1 semantics)."""
    settings = _settings(agent_max_tool_calls=1)
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [_strong_row()])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [_search_call("c1", "x")]},
            {"tool_calls": [_search_call("c2", "y")]},
            {"content": "answering from what I have."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="q?"
        ),
        deps=deps,
    )
    types = [e.event_type for e in outcome.events]
    assert "budget_exceeded" in types
    assert outcome.tool_call_count == 1
    assert outcome.final_message == "answering from what I have."
    assert types[-1] == "done"
    # The forced-answer nudge went in as a synthetic user message.
    final_messages = openai.chat_calls[-1]["messages"]
    assert any(
        m.get("role") == "user"
        and "tool-call budget" in str(m.get("content", ""))
        for m in final_messages
    )
    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    # A budget-forced best-effort with no SQL behind it is a no-data
    # claim in the record schema.
    assert records[0].record["outcome"] == "no_data"


def test_wallclock_timeout_produces_v1_terminal_sequence() -> None:
    """Timeout inside the research phase → `turn_timeout` then `error`
    with reason turn_timeout, and no `done`."""
    settings = _settings(agent_turn_timeout_seconds=-1)
    bq = FakeBqClient()
    openai = FakeOpenAIClient(chat_script=[{"content": "unused"}])
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="q?"
        ),
        deps=deps,
    )
    types = [e.event_type for e in outcome.events]
    assert openai.chat_calls == []  # timed out before the model call
    assert types[-2:] == ["turn_timeout", "error"]
    error = outcome.events[-1]
    assert isinstance(error, agent_events.ErrorEvent)
    assert error.reason == "turn_timeout"
    assert "done" not in types


def test_openai_error_surfaces_as_error_event_without_done() -> None:
    class ExplodingOpenAI(FakeOpenAIClient):
        def chat_with_tools(self, **kwargs: Any) -> Any:
            raise RuntimeError("rate limit exceeded, temporarily down")

    settings = _settings()
    deps = _deps(
        settings=settings, bq=FakeBqClient(), openai=ExplodingOpenAI()
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="q?"
        ),
        deps=deps,
    )
    error = outcome.events[-1]
    assert isinstance(error, agent_events.ErrorEvent)
    assert error.reason == "openai_error"
    assert error.retryable is True
    types = [e.event_type for e in outcome.events]
    assert "done" not in types
