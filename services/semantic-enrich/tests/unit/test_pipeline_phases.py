"""v2 orchestrator phase dispatch.

Phases are exercised through recording doubles: the tests assert call
order, the short-circuit paths (triage answer, cache replay) skipping
later phases, the verify-retry gate, and the terminal guarantee that
every path ends with exactly one `done` or `error`.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import (
    NoopMemory,
    PipelineDeps,
    RecallOutcome,
    ResearchResult,
    TriageOutcome,
    TurnContext,
    Verdict,
)
from semantic_enrich.core.agent.pipeline import (
    run_turn,
    run_turn_collected,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.openai_fakes import FakeOpenAIClient


class RecordingTriage:
    def __init__(
        self, calls: list[str], outcome: TriageOutcome | None = None
    ) -> None:
        self.calls = calls
        self.outcome = outcome or TriageOutcome(category="in_scope")

    def classify(self, ctx: TurnContext) -> TriageOutcome:
        self.calls.append("triage")
        return self.outcome


class RecordingMemory(NoopMemory):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def recall(self, ctx: TurnContext) -> RecallOutcome:
        self.calls.append("memory.recall")
        return super().recall(ctx)

    def commit(self, ctx: TurnContext) -> None:
        self.calls.append("memory.commit")
        super().commit(ctx)


class RecordingVerifier:
    """Pops scripted verdict actions; records each check call."""

    def __init__(
        self, calls: list[str], script: list[str] | None = None
    ) -> None:
        self.calls = calls
        self.script = list(script or [])

    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict:
        self.calls.append(f"verify(final={final})")
        action = self.script.pop(0) if self.script else "accept"
        assert action in ("accept", "retry")
        return Verdict(action=action)  # type: ignore[arg-type]


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
    openai: FakeOpenAIClient,
    calls: list[str],
    triage: TriageOutcome | None = None,
    verify_script: list[str] | None = None,
) -> PipelineDeps:
    return PipelineDeps(
        bq=object(),  # type: ignore[arg-type]  # never touched by these scripts
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-v2-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
        triage=RecordingTriage(calls, triage),
        memory=RecordingMemory(calls),
        verifier=RecordingVerifier(calls, verify_script),
    )


def _request(question: str = "what happened?") -> ChatRequest:
    return ChatRequest(
        conversation_id="c1", history=[], question=question
    )


def _terminal_events(
    events: list[agent_events.AgentEvent],
) -> list[str]:
    return [
        e.event_type for e in events if e.event_type in ("done", "error")
    ]


def _phase_sequence(events: list[agent_events.AgentEvent]) -> list[str]:
    return [
        e.phase
        for e in events
        if isinstance(e, agent_events.PhaseStart)
    ]


def test_phases_run_in_order_on_the_happy_path() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "the answer."}])
    deps = _deps(settings=_settings(), openai=openai, calls=calls)
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert calls == [
        "triage",
        "memory.recall",
        "verify(final=False)",
        "memory.commit",
    ]
    assert _phase_sequence(outcome.events) == [
        "triage",
        "memory",
        "research",
        "verify",
        "answer",
    ]
    assert outcome.final_message == "the answer."
    assert _terminal_events(outcome.events) == ["done"]


def test_triage_short_circuit_skips_memory_and_research() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "unused"}])
    deps = _deps(
        settings=_settings(),
        openai=openai,
        calls=calls,
        triage=TriageOutcome(
            category="off_scope",
            short_circuit="I only answer questions about the corpus.",
        ),
    )
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert calls == ["triage", "memory.commit"]
    assert openai.chat_calls == []
    assert outcome.final_message.startswith("I only answer")
    assert _phase_sequence(outcome.events) == ["triage", "answer"]
    # The short-circuit still owes the FE a turn_start.
    types = [e.event_type for e in outcome.events]
    assert "turn_start" in types
    assert _terminal_events(outcome.events) == ["done"]


def test_cache_replay_skips_research_and_verify() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "first run."}])
    deps = _deps(settings=_settings(), openai=openai, calls=calls)
    request = _request()

    first = run_turn_collected(request=request, deps=deps)
    assert first.cache_hit is False
    calls.clear()
    chat_calls_before = len(openai.chat_calls)

    second = run_turn_collected(request=request, deps=deps)
    assert second.cache_hit is True
    assert calls == ["triage", "memory.recall"]
    assert len(openai.chat_calls) == chat_calls_before
    assert any(
        isinstance(e, agent_events.CacheHit) for e in second.events
    )
    assert second.final_message == "first run."
    assert _terminal_events(second.events) == ["done"]


def test_verify_retry_reenters_research_once() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(
        chat_script=[
            {"content": "draft answer."},
            {"content": "revised answer."},
        ]
    )
    deps = _deps(
        settings=_settings(),
        openai=openai,
        calls=calls,
        verify_script=["retry", "accept"],
    )
    outcome = run_turn_collected(request=_request(), deps=deps)

    assert calls == [
        "triage",
        "memory.recall",
        "verify(final=False)",
        "verify(final=True)",
        "memory.commit",
    ]
    assert _phase_sequence(outcome.events).count("research") == 2
    assert outcome.final_message == "revised answer."
    assert _terminal_events(outcome.events) == ["done"]


def test_verify_retry_capped_by_setting() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(
        chat_script=[
            {"content": "draft."},
            {"content": "retry one."},
            {"content": "never requested."},
        ]
    )
    deps = _deps(
        settings=_settings(agent_verify_max_retries=1),
        openai=openai,
        calls=calls,
        # Always retry: the cap, not the verdict, must stop the loop.
        verify_script=["retry", "retry"],
    )
    outcome = run_turn_collected(request=_request(), deps=deps)

    # Initial research + exactly one retry; two model calls total.
    assert len(openai.chat_calls) == 2
    assert _phase_sequence(outcome.events).count("research") == 2
    assert _terminal_events(outcome.events) == ["done"]


def test_invalid_history_yields_error_before_any_phase() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "unused"}])
    deps = _deps(settings=_settings(), openai=openai, calls=calls)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[{"role": "wizard", "content": "bad"}],
            question="anything",
        ),
        deps=deps,
    )
    assert calls == []
    types = [e.event_type for e in outcome.events]
    assert types == ["turn_start", "error"]
    error = outcome.events[-1]
    assert isinstance(error, agent_events.ErrorEvent)
    assert error.reason == "invalid_history"


def test_turn_record_carries_the_v2_skeleton() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
    deps = _deps(settings=_settings(), openai=openai, calls=calls)
    outcome = run_turn_collected(request=_request("q?"), deps=deps)

    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(records) == 1
    record = records[0].record
    assert record["loop_impl"] == "v2"
    assert record["question"] == "q?"
    assert record["answer"] == "answer."
    assert record["terminal_reason"] == "final_answer"
    assert record["turn_id"] == outcome.turn_id


def test_turn_records_field_is_accepted_and_ignored() -> None:
    calls: list[str] = []
    openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
    deps = _deps(settings=_settings(), openai=openai, calls=calls)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="q?",
            turn_records=[{"turn_id": "t-prior", "packages": ["pkg-9"]}],
        ),
        deps=deps,
    )
    assert outcome.final_message == "answer."
    assert _terminal_events(outcome.events) == ["done"]


def test_events_stream_lazily_not_all_at_once() -> None:
    """The orchestrator is a generator: the first event must arrive
    before the model call happens (phase_start precedes research)."""
    openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
    deps = _deps(settings=_settings(), openai=openai, calls=[])
    stream = run_turn(request=_request(), deps=deps)
    first = next(stream)
    assert isinstance(first, agent_events.PhaseStart)
    assert first.phase == "triage"
    assert openai.chat_calls == []
    rest = list(stream)
    assert rest[-1].event_type == "done"
