"""The answer-fit verifier: verdict → disposition mapping.

Every path through the act-mode gates (confidence demotion, retry
cap, final-check retry ban, clarify guards) plus shadow-mode
transparency and the fail-open posture. The checker itself is a
scripted FakeOpenAI — these tests pin the enforcement, not the
model's judgment.
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


class _ExplodingOpenAI(FakeOpenAIClient):
    def generate_structured(self, **kwargs: Any) -> Any:
        raise RuntimeError("checker down")


def _ctx(
    *,
    settings: Settings,
    responses: list[dict[str, Any]] | None = None,
    openai: FakeOpenAIClient | None = None,
    prior_clarify: bool = False,
) -> TurnContext:
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=openai
        or FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
            structured_responses=responses or [],
        ),
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    ctx = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="compare housing grants across provinces since 2020",
        ),
        deps=deps,
    )
    ctx.state.prior_clarify = prior_clarify
    return ctx


def _result(
    *, answer: str = "the answer.", sql_ok: bool = True
) -> ResearchResult:
    sql_runs = (
        [
            {
                "sql": "SELECT 1 FROM r WHERE document_id IN ('d') LIMIT 5",
                "status": "ok",
                "row_count": 5,
                "null_ratio_warning": None,
            }
        ]
        if sql_ok
        else []
    )
    return ResearchResult(
        candidate_answer=answer,
        terminal_reason="final_answer",
        sql_runs=sql_runs,
    )


def _checker_says(
    *,
    fits: bool = False,
    action: str = "caveat",
    confidence: float = 0.9,
    gap: str | None = "per-province figures",
    retry_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "fits": fits,
        "confidence": confidence,
        "gap": gap,
        "action": action,
        "retry_hint": retry_hint,
    }


def _verification(ctx_events: list[Any]) -> agent_events.Verification:
    found = [
        e for e in ctx_events if isinstance(e, agent_events.Verification)
    ]
    assert len(found) == 1
    return found[0]


def _verifier(settings: Settings) -> AnswerFitVerifier:
    return AnswerFitVerifier.from_settings(settings)


# ── clean answer ──


def test_fits_ships_unchanged() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(fits=True, action="answer", gap=None)],
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.composed_message is None
    assert verdict.outcome_override is None
    event = _verification(verdict.events)
    assert event.fits is True
    assert event.enforced is True


# ── caveat ──


def test_caveat_prepends_template_and_tags_outcome() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_checker_says()])
    verdict = _verifier(settings).check(ctx, _result(answer="found $X."))
    assert verdict.action == "accept"
    assert verdict.composed_message == (
        "**Partial answer:** this does not cover per-province figures."
        "\n\nfound $X."
    )
    assert verdict.outcome_override == "answered_with_caveat"


def test_low_confidence_non_answer_demotes_to_caveat() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="retry", confidence=0.5)],
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.composed_message is not None
    assert verdict.composed_message.startswith("**Partial answer:**")


def test_empty_gap_non_answer_demotes_to_answer() -> None:
    # Nothing to compose a caveat or hint from → ship unchanged.
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="caveat", gap=None)],
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.composed_message is None


# ── retry ──


def test_retry_returns_retry_with_gap_hint() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[
            _checker_says(
                action="retry",
                gap="per-province breakdown",
                retry_hint="provincial grant columns",
                confidence=0.95,
            )
        ],
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "retry"
    assert len(verdict.hints) == 1
    assert "per-province breakdown" in verdict.hints[0].text
    assert "provincial grant columns" in verdict.hints[0].text


def test_final_check_forbids_retry() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="retry", confidence=0.95)],
    )
    verdict = _verifier(settings).check(ctx, _result(), final=True)
    assert verdict.action == "accept"
    assert verdict.outcome_override == "answered_with_caveat"


def test_exhausted_retry_budget_falls_back_to_caveat() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="retry", confidence=0.95)],
    )
    ctx.verify_retries_used = settings.agent_verify_max_retries
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.outcome_override == "answered_with_caveat"


# ── clarify ──


def test_clarify_on_surrender_composes_question() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[
            _checker_says(
                action="clarify", gap="which program", confidence=0.9
            )
        ],
    )
    verdict = _verifier(settings).check(ctx, _result(sql_ok=False))
    assert verdict.action == "accept"
    assert verdict.outcome_override == "clarified"
    assert verdict.composed_message is not None
    assert "which program" in verdict.composed_message
    assert verdict.composed_message.rstrip().endswith(
        "helps me search better."
    )


def test_clarify_never_withholds_real_data() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="clarify", confidence=0.9)],
    )
    verdict = _verifier(settings).check(ctx, _result(sql_ok=True))
    assert verdict.outcome_override == "answered_with_caveat"
    assert verdict.composed_message is not None
    assert verdict.composed_message.startswith("**Partial answer:**")


def test_clarify_suppressed_after_previous_clarify() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="clarify", confidence=0.9)],
        prior_clarify=True,
    )
    verdict = _verifier(settings).check(ctx, _result(sql_ok=False))
    assert verdict.outcome_override == "answered_with_caveat"


# ── shadow mode ──


def test_log_mode_never_alters_output() -> None:
    settings = _settings(agent_verify_mode="log")
    ctx = _ctx(
        settings=settings,
        responses=[_checker_says(action="retry", confidence=0.95)],
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.composed_message is None
    assert verdict.hints == []
    event = _verification(verdict.events)
    assert event.fits is False
    assert event.action == "retry"  # the checker's verdict, logged
    assert event.enforced is False


# ── fail-open ──


def test_checker_error_fails_open() -> None:
    settings = _settings()
    ctx = _ctx(
        settings=settings,
        openai=_ExplodingOpenAI(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
    )
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.composed_message is None
    event = _verification(verdict.events)
    assert event.fits is True
    assert event.enforced is False
    assert event.reason == "checker_error"


def test_schema_invalid_output_fails_open() -> None:
    settings = _settings()
    # Empty response queue → the fake's canned SQL dict, which is not
    # a verification verdict → invalid_output.
    ctx = _ctx(settings=settings, responses=[])
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    event = _verification(verdict.events)
    assert event.enforced is False
    assert event.reason == "invalid_output"


def test_off_mode_skips_the_call_entirely() -> None:
    settings = _settings(agent_verify_mode="off")
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
    )
    ctx = _ctx(settings=settings, openai=openai)
    verdict = _verifier(settings).check(ctx, _result())
    assert verdict.action == "accept"
    assert verdict.events == []
    assert openai.structured_calls == []
