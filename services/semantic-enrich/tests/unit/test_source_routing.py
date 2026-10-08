"""Source routing: triage names where an answer lives, research sees
only that source's tools, and a routed turn that reads no data widens
to every tool before it may answer — so routing never does worse than
not routing."""

from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_tools
from semantic_enrich.core.agent import triage as triage_mod
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    TriageOutcome,
    TurnContext,
)
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


class FixedTriage:
    def __init__(self, source: str, confidence: float) -> None:
        self.source = source
        self.confidence = confidence

    def classify(self, ctx: TurnContext) -> TriageOutcome:
        return TriageOutcome(category="in_scope", source=self.source, source_confidence=self.confidence)


def _deps(openai: FakeOpenAIClient, triage: Any, **overrides: Any) -> PipelineDeps:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        agent_verify_mode="off",
        **overrides,
    )
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash",
        cache=ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60),
        snapshot_hash_provider=lambda: "snap",
    )
    deps.triage = triage
    return deps


def _tool_names(call: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in call["tools"]}


def test_routed_turn_sees_only_its_tools() -> None:
    openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
    deps = _deps(openai, FixedTriage("statcan", 0.95))
    run_turn_collected(request=ChatRequest(conversation_id="c", history=[], question="cpi?"), deps=deps)
    first = _tool_names(openai.chat_calls[0])
    assert first == agent_tools.ROUTE_TOOLS["statcan"]


def test_routed_turn_without_data_widens_before_answering() -> None:
    openai = FakeOpenAIClient(
        chat_script=[{"content": "the routed tools cannot answer."}, {"content": "final."}]
    )
    deps = _deps(openai, FixedTriage("payments", 0.9))
    outcome = run_turn_collected(
        request=ChatRequest(conversation_id="c", history=[], question="q?"), deps=deps
    )
    assert len(openai.chat_calls) == 2
    assert _tool_names(openai.chat_calls[0]) == agent_tools.ROUTE_TOOLS["payments"]
    assert _tool_names(openai.chat_calls[1]) == set(agent_tools.TOOL_NAMES)
    assert outcome.final_message.startswith("final.")


def test_unsure_or_mixed_triage_does_not_narrow() -> None:
    for triage in (FixedTriage("statcan", 0.5), FixedTriage("mixed", 0.99)):
        openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
        run_turn_collected(
            request=ChatRequest(conversation_id="c", history=[], question="q?"),
            deps=_deps(openai, triage),
        )
        assert _tool_names(openai.chat_calls[0]) == set(agent_tools.TOOL_NAMES)
        assert len(openai.chat_calls) == 1


def test_routing_off_never_narrows() -> None:
    openai = FakeOpenAIClient(chat_script=[{"content": "answer."}])
    run_turn_collected(
        request=ChatRequest(conversation_id="c", history=[], question="q?"),
        deps=_deps(openai, FixedTriage("statcan", 0.99), agent_source_routing="off"),
    )
    assert _tool_names(openai.chat_calls[0]) == set(agent_tools.TOOL_NAMES)


def test_classifier_output_without_a_source_still_validates() -> None:
    parsed = {
        "category": "in_scope",
        "confidence": 0.9,
        "reason": "r",
        "off_scope_reason": None,
        "deflection_hint": None,
        "clarify_question": None,
    }
    c = triage_mod._validate(parsed)
    assert c is not None and c.source == "mixed" and c.source_confidence == 0.0
    c = triage_mod._validate({**parsed, "source": "payments", "source_confidence": 0.8})
    assert c is not None and c.source == "payments"
    c = triage_mod._validate({**parsed, "source": "bogus", "source_confidence": 7})
    assert c is not None and c.source == "mixed" and c.source_confidence == 0.0
