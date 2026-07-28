"""Every condition that must suppress an offer.

The single most important one is retrieval quality: chips offered over
datasets nothing scored as relevant would train users to ignore all
chips, which is worse than a clean "I don't have this."
"""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.evidence import (
    EvidencePackage,
    SearchEvidence,
)
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent.suggest import (
    OFFER_OUTCOMES,
    build_suggestions,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"


def _ctx(
    *,
    quality: str | None = "ok",
    turn_records: list[Any] | None = None,
    scope: tuple[str, ...] = (),
    **overrides: Any,
) -> TurnContext:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,
    )
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(),
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=4, max_value_bytes=100_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    ctx = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="q",
            turn_records=turn_records or [],
            scope_package_ids=scope,
        ),
        deps=deps,
    )
    if quality is not None:
        ctx.trace.searches.append(
            {"query": "q", "retrieval_quality": quality}
        )
    ctx.trace.packages_researched.append(PKG)
    ctx.state.doc_package["doc-0"] = PKG
    ctx.state.doc_columns["doc-0"] = ["a", "b"]
    return ctx


_EVIDENCE = SearchEvidence(
    packages=(
        EvidencePackage(package_id=PKG, title="Estimates", column_count=2),
    ),
    queries_tried=("q",),
    truncated=0,
)
_EMPTY = SearchEvidence(packages=(), queries_tried=(), truncated=0)
_RESULT = ResearchResult(
    candidate_answer="x", terminal_reason="final_answer"
)


def _build(ctx: TurnContext, evidence: Any = _EVIDENCE, outcome: str = "no_data") -> list[Any]:
    return build_suggestions(ctx, _RESULT, evidence, outcome)


def test_kill_switch_suppresses_everything() -> None:
    assert _build(_ctx(agent_suggestions_enabled=False)) == []


@pytest.mark.parametrize("outcome", sorted(OFFER_OUTCOMES))
def test_offer_outcomes_produce_suggestions(outcome: str) -> None:
    assert _build(_ctx(), outcome=outcome)


@pytest.mark.parametrize(
    "outcome", ["answered", "clarified", "deflected", "error"]
)
def test_other_outcomes_produce_none(outcome: str) -> None:
    assert _build(_ctx(), outcome=outcome) == []


def test_no_packages_produces_none() -> None:
    assert _build(_ctx(), evidence=_EMPTY) == []


def test_all_weak_retrieval_produces_none() -> None:
    """The false-invitation guard. Below the similarity floor the loop
    has no dataset worth exploring."""
    assert _build(_ctx(quality="weak")) == []


def test_no_search_at_all_produces_none() -> None:
    assert _build(_ctx(quality=None)) == []


def test_a_scoped_turn_needs_no_search_of_its_own() -> None:
    """A clicked chip goes straight to `list_documents`, so a scoped
    turn has no search to score. Its packages came from a previous
    turn's offers, which already cleared this gate — re-applying it
    here would silently kill every follow-up offer in a chain."""
    assert _build(_ctx(quality=None, scope=(PKG,)), outcome="explored")


def test_chain_cap_stops_an_endless_ride() -> None:
    explored = {"outcome": "explored"}
    assert _build(_ctx(turn_records=[explored, explored]))
    assert _build(_ctx(turn_records=[explored] * 3)) == []


def test_the_chain_cap_counts_only_consecutive_explorations() -> None:
    explored = {"outcome": "explored"}
    answered = {"outcome": "answered"}
    # An answer in the middle breaks the chain: the user got somewhere.
    assert _build(
        _ctx(turn_records=[explored, explored, answered, explored])
    )


def test_malformed_turn_records_never_raise() -> None:
    for records in ([None], ["nope"], [{"outcome": 5}], [{}]):
        assert _build(_ctx(turn_records=list(records)))
