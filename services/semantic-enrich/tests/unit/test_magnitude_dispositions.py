"""The magnitude gate wired into AnswerFitVerifier.check: hard->retry,
hard-final->caveat, soft->caveat, none->fit-only, and shadow mode.

The fit checker is a scripted FakeOpenAI; these pin the deterministic
numeric enforcement and its composition with the fit verdict.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.derivation import Derivation
from semantic_enrich.core.agent.grounding import (
    CrossSourceVerdict,
    GroundingReport,
)
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
    overrides.setdefault("agent_magnitude_mode", "act")
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,
    )


def _ctx(*, settings: Settings, responses: list[dict[str, Any]] | None = None) -> TurnContext:
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
            structured_responses=responses or [],
        ),
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return TurnContext.begin(
        request=ChatRequest(conversation_id="c1", history=[], question="total spending?"),
        deps=deps,
    )


def _deriv(
    *, result_value: float, unit_scale: str = "unknown", source_row_estimate: int = 1412
) -> Derivation:
    return Derivation(
        source_packages=("pkg-1",),
        source_documents=("doc-1",),
        dataset_titles=("Ledger 2020-21",),
        aggregation="SUM",
        value_columns=("Amount",),
        group_by_columns=(),
        predicate_shape="",
        sql_shape="",
        row_count=1,
        source_row_estimate=source_row_estimate,
        result_value=result_value,
        result_label="total",
        unit_scale=unit_scale,  # type: ignore[arg-type]
        unit_source="unresolved",
        complete=True,
    )


def _result(
    *,
    answer: str = "Total spending was about $8.",
    derivations: list[Derivation] | None = None,
    grounding: GroundingReport | None = None,
) -> ResearchResult:
    return ResearchResult(
        candidate_answer=answer,
        terminal_reason="final_answer",
        sql_runs=[{"sql": "SELECT 1", "status": "ok", "row_count": 1, "null_ratio_warning": None}],
        derivations=derivations or [],
        grounding=grounding,
    )


def _fit(action: str = "answer", fits: bool = True, gap: str | None = None) -> dict[str, Any]:
    return {"fits": fits, "confidence": 0.9, "gap": gap, "action": action, "retry_hint": None}


def _verifier(settings: Settings) -> AnswerFitVerifier:
    return AnswerFitVerifier.from_settings(settings)


def _cross_source() -> GroundingReport:
    return GroundingReport(
        grounding="grounded",
        headline_value=9e11,
        matched=True,
        cross_source_sum=CrossSourceVerdict(
            flagged=True, packages=("a", "b"), fiscal_years=("2024-25", "2025-26")
        ),
    )


# ── hard -> retry ──


def test_hard_floor_retries_before_fit_call() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[])  # no fit response scripted
    verdict = _verifier(settings).check(ctx, _result(derivations=[_deriv(result_value=8.2)]))
    assert verdict.action == "retry"
    assert verdict.hints and "implausibly small" in verdict.hints[0].text
    # The fit checker was never called — magnitude short-circuited.
    assert ctx.deps.openai_client.structured_calls == []  # type: ignore[attr-defined]


def test_hard_floor_final_caveats_and_runs_fit() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    verdict = _verifier(settings).check(
        ctx, _result(derivations=[_deriv(result_value=8.2)]), final=True
    )
    assert verdict.action == "accept"
    assert verdict.composed_message is not None
    assert "Check this figure" in verdict.composed_message
    assert verdict.outcome_override == "answered_with_caveat"
    # The two verdicts are discriminable so the fits-rate metric (which
    # consumes only kind="fit") is not polluted by the magnitude event.
    kinds = {
        e.kind
        for e in verdict.events
        if isinstance(e, agent_events.Verification)
    }
    assert kinds == {"fit", "magnitude"}


# ── cross-source -> caveat composes ──


def test_cross_source_caveat_composes_with_answer() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    verdict = _verifier(settings).check(
        ctx,
        _result(
            answer="The total was $900.84B.",
            derivations=[_deriv(result_value=9e11, unit_scale="dollars", source_row_estimate=100)],
            grounding=_cross_source(),
        ),
        final=True,
    )
    assert verdict.action == "accept"
    assert verdict.composed_message is not None
    assert "double-count" in verdict.composed_message
    assert "The total was $900.84B." in verdict.composed_message


def test_cross_source_caveats_not_retries_even_with_retry_available() -> None:
    # A cross-source sum may be a legitimate multi-year total; it must
    # surface a caveat, never spend a retry that would just reproduce it.
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    verdict = _verifier(settings).check(  # final defaults False -> retry available
        ctx,
        _result(
            answer="The total was $900.84B.",
            derivations=[_deriv(result_value=9e11, unit_scale="dollars", source_row_estimate=100)],
            grounding=_cross_source(),
        ),
    )
    assert verdict.action == "accept"
    assert verdict.composed_message is not None
    assert "double-count" in verdict.composed_message


# ── soft -> caveat ──


def test_soft_unknown_units_caveats() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    # A plausible large total with unknown scale -> soft units caveat.
    verdict = _verifier(settings).check(
        ctx,
        _result(
            answer="It was $5.2 billion.",
            derivations=[_deriv(result_value=5.2e9, unit_scale="unknown", source_row_estimate=10)],
        ),
    )
    assert verdict.action == "accept"
    assert verdict.composed_message is not None
    assert "Units unverified" in verdict.composed_message


# ── none -> fit only ──


def test_no_finding_leaves_fit_path_untouched() -> None:
    settings = _settings()
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    verdict = _verifier(settings).check(
        ctx,
        _result(
            answer="It was $4.5 billion.",
            derivations=[_deriv(result_value=4.5e11, unit_scale="dollars", source_row_estimate=1000)],
        ),
    )
    assert verdict.action == "accept"
    assert verdict.composed_message is None


# ── shadow mode ──


def test_shadow_mode_emits_but_does_not_alter() -> None:
    settings = _settings(agent_magnitude_mode="log")
    ctx = _ctx(settings=settings, responses=[_fit(action="answer", fits=True)])
    verdict = _verifier(settings).check(ctx, _result(derivations=[_deriv(result_value=8.2)]))
    assert verdict.action == "accept"
    assert verdict.composed_message is None
    # The magnitude verdict is still emitted for the shadow record.
    mags = [
        e for e in verdict.events
        if isinstance(e, agent_events.Verification) and not e.enforced
        and "absurd_floor" in e.reason
    ]
    assert len(mags) == 1
