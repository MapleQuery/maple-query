"""Triage phase routing + fail-open behaviour.

The classifier is a `FakeOpenAIClient` popping canned structured
outputs, so every branch is exercised without the vendor: category
routing in `act` mode, shadow (`log`) mode never short-circuiting,
and every fail-open path (error, timeout, malformed output, low
confidence, clarify without a question) continuing to research.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent.phases import PipelineDeps, TurnContext
from semantic_enrich.core.agent.triage import QueryTriage
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


@pytest.fixture(autouse=True)
def _fresh_corpus_cache() -> None:
    agent_tools.reset_corpus_stats_cache()


def _settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {
        "gcp_project_id": "proj",
        "openai_api_key": "sk-test",
        "agent_triage_mode": "act",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _classifier_output(
    category: str,
    *,
    confidence: float = 0.95,
    off_scope_reason: str | None = None,
    deflection_hint: str | None = None,
    clarify_question: str | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "confidence": confidence,
        "reason": "test classification",
        "off_scope_reason": off_scope_reason,
        "deflection_hint": deflection_hint,
        "clarify_question": clarify_question,
    }


def _ctx(
    *,
    settings: Settings,
    openai: FakeOpenAIClient,
    bq: FakeBqClient | None = None,
    question: str = "how much did canada spend on airfare",
    history: list[dict[str, Any]] | None = None,
) -> TurnContext:
    deps = PipelineDeps(
        bq=bq if bq is not None else FakeBqClient(),  # type: ignore[arg-type]
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1",
            history=history or [],
            question=question,
        ),
        deps=deps,
    )


def _triage_result(
    events: list[agent_events.AgentEvent],
) -> agent_events.TriageResult:
    results = [
        e for e in events if isinstance(e, agent_events.TriageResult)
    ]
    assert len(results) == 1
    return results[0]


def test_in_scope_continues_and_is_enforced() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[_classifier_output("in_scope")]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "in_scope"
    assert outcome.short_circuit is None
    result = _triage_result(outcome.events)
    assert result.category == "in_scope"
    assert result.enforced is True
    # The classifier call was charged into the shared meters.
    assert ctx.tokens_in_total > 0
    assert any(
        isinstance(e, agent_events.CostUpdate) for e in outcome.events
    )


def test_off_scope_short_circuits_with_the_template() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output(
                "off_scope",
                off_scope_reason="provincial",
                deflection_hint="federal travel expenses by department",
            )
        ]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "off_scope"
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.startswith("MapleQuery answers")
    assert "federal travel expenses" in outcome.short_circuit
    assert _triage_result(outcome.events).enforced is True


def test_clarify_short_circuits_with_the_classifier_question() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output(
                "clarify",
                clarify_question="Which years should I compare?",
            )
        ]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "clarify"
    assert outcome.short_circuit == "Which years should I compare?"


def test_meta_short_circuits_from_corpus_stats() -> None:
    bq = FakeBqClient()
    bq.table_num_rows_by_ref["proj.raw.rows"] = 12_345_678
    bq.register_query(
        "AS packages",
        [
            {
                "packages": 210,
                "documents_loaded": 950,
                "latest_load_at": "2026-07-01T00:00:00",
            }
        ],
    )
    openai = FakeOpenAIClient(
        structured_responses=[_classifier_output("meta")]
    )
    ctx = _ctx(
        settings=_settings(),
        openai=openai,
        bq=bq,
        question="how many rows of data do you have access to?",
    )
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "meta"
    assert outcome.short_circuit is not None
    assert "12,345,678" in outcome.short_circuit


def test_log_mode_classifies_but_never_short_circuits() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output("off_scope", off_scope_reason="news")
        ]
    )
    ctx = _ctx(settings=_settings(agent_triage_mode="log"), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "off_scope"
    assert outcome.short_circuit is None
    result = _triage_result(outcome.events)
    assert result.category == "off_scope"
    assert result.enforced is False


def test_low_confidence_fails_open_but_logs_the_category() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output("off_scope", confidence=0.5)
        ]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.short_circuit is None
    result = _triage_result(outcome.events)
    # The disagreement is preserved for shadow-mode tuning …
    assert result.category == "off_scope"
    # … but not acted on.
    assert result.enforced is False


def test_malformed_output_fails_open() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[{"category": "banana", "confidence": "high"}]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "in_scope"
    assert outcome.short_circuit is None
    result = _triage_result(outcome.events)
    assert result.category == "in_scope"
    assert result.enforced is False


def test_classifier_error_fails_open() -> None:
    class RaisingClient(FakeOpenAIClient):
        def generate_structured(self, **kwargs: Any) -> Any:
            raise RuntimeError("vendor down")

    ctx = _ctx(settings=_settings(), openai=RaisingClient())
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "in_scope"
    assert outcome.short_circuit is None
    assert _triage_result(outcome.events).confidence == 0.0


def test_classifier_timeout_fails_open_within_the_deadline() -> None:
    class SlowClient(FakeOpenAIClient):
        def generate_structured(self, **kwargs: Any) -> Any:
            time.sleep(0.5)
            return super().generate_structured(**kwargs)

    openai = SlowClient(
        structured_responses=[
            _classifier_output("off_scope", off_scope_reason="news")
        ]
    )
    ctx = _ctx(
        settings=_settings(agent_triage_timeout_ms=30), openai=openai
    )
    started = time.monotonic()
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)
    elapsed = time.monotonic() - started

    assert outcome.category == "in_scope"
    assert outcome.short_circuit is None
    # The phase honours its own deadline, not the vendor's.
    assert elapsed < 0.4


def test_clarify_without_a_question_fails_open() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output("clarify", clarify_question="  ")
        ]
    )
    ctx = _ctx(settings=_settings(), openai=openai)
    outcome = QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    assert outcome.category == "in_scope"
    assert outcome.short_circuit is None
    assert _triage_result(outcome.events).enforced is False


def test_context_hint_from_the_last_user_message() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[_classifier_output("in_scope")]
    )
    ctx = _ctx(
        settings=_settings(),
        openai=openai,
        question="what about 2021?",
        history=[
            {"role": "user", "content": "federal housing spending"},
            {"role": "assistant", "content": "here are the numbers"},
        ],
    )
    QueryTriage.from_settings(ctx.deps.settings).classify(ctx)

    prompt = openai.structured_calls[0]["prompt"]
    assert "Previous topic: federal housing spending" in prompt
    assert "Question: what about 2021?" in prompt


def test_classifier_call_shape() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[_classifier_output("in_scope")]
    )
    settings = _settings(agent_triage_model="gpt-4o-mini")
    ctx = _ctx(settings=settings, openai=openai)
    QueryTriage.from_settings(settings).classify(ctx)

    call = openai.structured_calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["schema_name"] == "triage"
    assert call["timeout_s"] == pytest.approx(2.0)
