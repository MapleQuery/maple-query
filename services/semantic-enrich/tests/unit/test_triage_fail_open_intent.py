"""A failed classification must not become a confident one.

Triage runs on a 3.5s timeout and, when the classifier returns nothing,
falls through to `category="in_scope", confidence=0.0`. That default is
indistinguishable downstream from a real `in_scope` ruling, so a typed
descriptive question ("summarize the datasets you have on federal
employee travel") stopped being routed to the descriptive rubric and was
judged by the answer-fit checker instead — which asks whether the turn
produced data, correctly answers no, and prepends a caveat.

Observed in production, verbatim:

    **Partial answer:** this does not cover a summary of the datasets,
    including their specific contents and any relevant details about
    their scope or limitations.

    I found two relevant datasets on federal employee travel:
    1. Proactive Disclosure - Travel Expenses: ...

The caveat contradicts the answer directly beneath it. Roughly 7% of
production turns hit the timeout, and the same fixture question flipped
category between two runs — a race, not a model disagreeing with itself.

These drive the real verifier and the real triage phase. An earlier
draft reimplemented the pipeline's intent stamp inside the test file,
which would have passed no matter what the pipeline did.
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
)
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
        **overrides,
    )


def _ctx(*, settings: Settings, responses: list[dict[str, Any]]) -> TurnContext:
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
            structured_responses=responses,
        ),
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=4, max_value_bytes=100_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="summarize the datasets you have on federal employee travel",
        ),
        deps=deps,
    )


def _result() -> ResearchResult:
    """A descriptive answer: real content, no SQL behind it — exactly the
    shape the fit checker calls a partial answer."""
    return ResearchResult(
        candidate_answer=(
            "I found two relevant datasets on federal employee travel: "
            "Proactive Disclosure - Travel Expenses, and Annual "
            "Expenditures on Travel, Hospitality and Conferences."
        ),
        terminal_reason="final_answer",
        packages_cited=["pkg-travel"],
    )


def _checker_says_unfit() -> dict[str, Any]:
    return {
        "fits": False,
        "confidence": 0.9,
        "gap": (
            "a summary of the datasets, including their specific contents"
        ),
        "action": "caveat",
        "retry_hint": None,
    }


def _verification(events: list[Any]) -> agent_events.Verification:
    found = [e for e in events if isinstance(e, agent_events.Verification)]
    assert found
    return found[-1]


def test_a_known_intent_still_enforces_the_caveat() -> None:
    """The control. Nothing about the guard weakens verification on a
    turn whose question was actually read."""
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_checker_says_unfit()])
    assert ctx.turn_intent_known is True

    verdict = AnswerFitVerifier.from_settings(settings).check(ctx, _result())

    # A caveat ships as a rewritten message, not as a distinct action.
    assert verdict.composed_message is not None
    assert verdict.composed_message.startswith("**Partial answer:**")
    assert _verification(verdict.events).enforced is True


def test_an_unknown_intent_does_not_caveat_the_answer() -> None:
    """The fix. Same checker verdict, same answer — but triage never read
    the question, so the caveat that would contradict it is not applied."""
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_checker_says_unfit()])
    ctx.turn_intent_known = False

    verdict = AnswerFitVerifier.from_settings(settings).check(ctx, _result())

    # The answer ships exactly as the research phase produced it.
    assert verdict.composed_message is None


def test_the_verdict_is_still_recorded_when_intent_is_unknown() -> None:
    """Demoted to shadow, not skipped. The check still runs and still
    reports, so these turns stay visible in the fits-rate rather than
    vanishing from the metric that would show the problem."""
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_checker_says_unfit()])
    ctx.turn_intent_known = False

    verdict = AnswerFitVerifier.from_settings(settings).check(ctx, _result())

    event = _verification(verdict.events)
    assert event.fits is False
    assert event.enforced is False


def test_shadow_mode_is_unaffected_by_the_guard() -> None:
    """With verify already in `log` there is nothing to demote."""
    settings = _settings(agent_verify_mode="log")
    ctx = _ctx(settings=settings, responses=[_checker_says_unfit()])
    ctx.turn_intent_known = False

    verdict = AnswerFitVerifier.from_settings(settings).check(ctx, _result())

    assert verdict.composed_message is None
    assert _verification(verdict.events).enforced is False


def test_intent_is_known_by_default() -> None:
    """Every turn that never touches triage — the v1 loop, replays —
    keeps today's behaviour."""
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_checker_says_unfit()])
    assert ctx.turn_intent_known is True
