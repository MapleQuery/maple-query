"""The explore rubric end to end: intent, modes, and the M6 boundary.

Two things are being pinned. That explore intent arrives from *either*
signal — a clicked chip's package scope or a typed exploratory question
— and produces the same descriptive handling. And that the rubric's one
substantive judgement, "a description must not assert a computed
total", is reachable without loosening anything numeric.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TriageOutcome,
    TurnContext,
    is_descriptive,
)
from semantic_enrich.core.agent.pipeline import (
    PipelineOutcome,
    run_turn_collected,
)
from semantic_enrich.core.agent.verify import (
    AnswerFitVerifier,
    assemble_explore_inputs,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
TITLE = "Supplementary Estimates B, 2025-26"
SUMMARY = "This dataset covers spending authorities across 3 columns."
SUMMARY_WITH_TOTAL = "The dataset totals $4.2M across all votes."


class _ExploreTriage:
    """Classifies every turn as a typed exploratory question."""

    def classify(self, ctx: TurnContext) -> TriageOutcome:
        return TriageOutcome(category="explore")


def _settings(**overrides: Any) -> Settings:
    overrides.setdefault("agent_verify_mode", "act")
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
    triage: Any = None,
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
        verifier=AnswerFitVerifier.from_settings(settings),
    )
    if triage is not None:
        deps.triage = triage
    return deps


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _bq() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": PKG,
                "title": TITLE,
                "summary": "estimates",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.1,
            }
        ],
    )
    bq.register_query(
        "FROM `proj.raw.documents`",
        [
            {
                "document_id": "doc-1",
                "package_id": PKG,
                "title": TITLE,
                "source_url": "http://x",
                "row_count": 4120,
            }
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING",
        BoundedQueryResult(
            rows=[
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": '{"Department": "TBS", "Vote": "1"}',
                }
            ],
            total_bytes_billed=64,
            slot_ms=1,
            elapsed_ms=1,
            timed_out=False,
            error=None,
        ),
    )
    return bq


def _call(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def _caveat_check() -> dict[str, Any]:
    return {
        "fits": False,
        "confidence": 0.95,
        "gap": "a total it never computed",
        "action": "caveat",
    }


def _typed_script(answer: str = SUMMARY) -> list[dict[str, Any]]:
    return [
        _call("c1", "search_datasets", {"query": "supplementary estimates"}),
        _call("c2", "list_documents", {"package_ids": [PKG]}),
        {"content": answer},
    ]


def _run(
    *,
    settings: Settings,
    script: list[dict[str, Any]],
    checks: Any = None,
    scope: tuple[str, ...] = (),
    triage: Any = None,
) -> tuple[PipelineOutcome, FakeOpenAIClient]:
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=script,
        structured_responses=checks,
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="what's in the supplementary estimates?",
            scope_package_ids=scope,
        ),
        deps=_deps(
            settings=settings, bq=_bq(), openai=openai, triage=triage
        ),
    )
    return outcome, openai


def _record(outcome: PipelineOutcome) -> dict[str, Any]:
    events = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(events) == 1
    return events[0].record


# ── intent arrives from either signal ──


def test_typed_exploratory_question_continues_the_pipeline() -> None:
    """Unlike every other non-in_scope category, `explore` does not
    short-circuit: exploration needs the research loop."""
    outcome, _ = _run(
        settings=_settings(),
        script=_typed_script(),
        triage=_ExploreTriage(),
        checks=[{"fits": True, "confidence": 0.9, "gap": None, "action": "answer"}],
    )
    kinds = [e.event_type for e in outcome.events]
    assert "documents_listed" in kinds  # research really ran
    record = _record(outcome)
    assert record["category"] == "explore"
    assert record["outcome"] == "explored"


def test_intent_comes_from_scope_or_triage_or_both() -> None:
    settings = _settings()
    deps = _deps(settings=settings, bq=_bq(), openai=FakeOpenAIClient())
    result = ResearchResult(candidate_answer="x", terminal_reason="final_answer")

    def ctx_for(*, scope: tuple[str, ...], category: str) -> TurnContext:
        ctx = TurnContext.begin(
            request=ChatRequest(
                conversation_id="c",
                history=[],
                question="q",
                scope_package_ids=scope,
            ),
            deps=deps,
        )
        if scope or category == "explore":
            ctx.turn_intent = "explore"
        return ctx

    assert is_descriptive(ctx_for(scope=(PKG,), category="in_scope"), result)
    assert is_descriptive(ctx_for(scope=(), category="explore"), result)
    assert is_descriptive(ctx_for(scope=(PKG,), category="explore"), result)
    assert not is_descriptive(ctx_for(scope=(), category="in_scope"), result)


# ── modes ──


def test_log_mode_emits_an_unenforced_event_and_ships_unchanged() -> None:
    outcome, _ = _run(
        settings=_settings(agent_verify_explore_mode="log"),
        script=_typed_script(),
        triage=_ExploreTriage(),
        checks=[_caveat_check()],
    )
    verifications = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.Verification)
    ]
    assert [v.kind for v in verifications] == ["explore"]
    assert verifications[0].enforced is False
    assert outcome.final_message == SUMMARY


def test_act_mode_prepends_the_caveat_and_keeps_the_outcome() -> None:
    outcome, _ = _run(
        settings=_settings(agent_verify_explore_mode="act"),
        script=_typed_script(),
        triage=_ExploreTriage(),
        checks=[_caveat_check()],
    )
    assert outcome.final_message.startswith(
        "**Note:** this description does not cover a total it never "
        "computed."
    )
    assert SUMMARY in outcome.final_message
    # A caveated exploration is still an exploration.
    assert _record(outcome)["outcome"] == "explored"


def test_checker_failure_ships_the_answer_unchanged() -> None:
    """Fail-open, same posture as every other gate here: an unparseable
    verdict degrades to `answer`, never to a blocked turn."""
    outcome, _ = _run(
        settings=_settings(agent_verify_explore_mode="act"),
        script=_typed_script(),
        triage=_ExploreTriage(),
        checks=[{"fits": False, "confidence": "not-a-number", "action": "caveat", "gap": "x"}],
    )
    assert outcome.final_message == SUMMARY


# ── the numeric boundary ──


def test_claims_a_total_is_deterministic_not_a_judgement() -> None:
    settings = _settings()
    deps = _deps(settings=settings, bq=_bq(), openai=FakeOpenAIClient())
    ctx = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c",
            history=[],
            question="q",
            scope_package_ids=(PKG,),
        ),
        deps=deps,
    )
    plain = assemble_explore_inputs(
        ctx,
        ResearchResult(
            candidate_answer=SUMMARY, terminal_reason="final_answer"
        ),
    )
    with_total = assemble_explore_inputs(
        ctx,
        ResearchResult(
            candidate_answer=SUMMARY_WITH_TOTAL,
            terminal_reason="final_answer",
        ),
    )
    assert plain["claims_a_total"] is False
    assert with_total["claims_a_total"] is True


def test_numeric_follow_up_keeps_the_full_path() -> None:
    """M6 non-regression, re-asserted on the intent-based predicate: an
    explore-intent turn that ran SQL is a numeric answer and takes the
    fit checker, grounding, and its derivation event."""
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 4_200_000.0}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    sql = (
        "SELECT SUM(CAST(JSON_VALUE(r.row, '$.Vote') AS FLOAT64)) AS total "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            _call("c1", "search_datasets", {"query": "estimates"}),
            _call("c2", "list_documents", {"package_ids": [PKG]}),
            _call("c3", "run_sql", {"sql": sql, "rationale": "sum"}),
            {"content": "Votes totalled $4.2M."},
        ],
        structured_responses=[
            {
                "fits": True,
                "confidence": 0.95,
                "gap": None,
                "action": "answer",
                "retry_hint": None,
            }
        ],
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="total the votes",
            scope_package_ids=(PKG,),
        ),
        deps=_deps(
            settings=_settings(),
            bq=bq,
            openai=openai,
            triage=_ExploreTriage(),
        ),
    )
    # The numeric fit checker ran, not the descriptive one.
    assert [c["schema_name"] for c in openai.structured_calls] == ["verify"]
    assert any(
        isinstance(e, agent_events.DerivationEvent) for e in outcome.events
    )
    assert _record(outcome)["outcome"] == "answered"


# ── shadow-run fixes: the two evidence bugs the first run exposed ──


def test_a_dollar_amount_inside_a_dataset_title_is_not_a_claim() -> None:
    """The first shadow run flagged a faithful description as claiming
    an ungrounded total. The only `$` in the answer was inside a dataset
    *name* — "Dashboard for infrastructure projects worth $20 million
    and over". A figure in a citation label is part of a title, not a
    claim the answer is making."""
    settings = _settings()
    deps = _deps(settings=settings, bq=_bq(), openai=FakeOpenAIClient())
    ctx = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c",
            history=[],
            question="q",
            scope_package_ids=(PKG,),
        ),
        deps=deps,
    )
    cited_only = (
        "The infrastructure data includes [Dashboard for infrastructure "
        "projects worth $20 million and over](/datasets/abc) and "
        "[Programs under $5M](/datasets/def)."
    )
    real_claim = (
        "The [Infrastructure dashboard](/datasets/abc) totals $4.2M "
        "across all projects."
    )
    assert not assemble_explore_inputs(
        ctx,
        ResearchResult(
            candidate_answer=cited_only, terminal_reason="final_answer"
        ),
    )["claims_a_total"]
    # The narrowing must not blind the check to a genuine total stated
    # alongside a citation — that is the whole numeric-trust boundary.
    assert assemble_explore_inputs(
        ctx,
        ResearchResult(
            candidate_answer=real_claim, terminal_reason="final_answer"
        ),
    )["claims_a_total"]


def test_unscoped_explore_turn_omits_scope_packages_entirely() -> None:
    """A typed exploratory question names no datasets. Sending
    `scope_packages: []` invited the rubric to read the absence of a
    scope as the answer having gone out of scope — which is what it did.
    The key is absent instead, so the wrong-dataset condition is
    unanswerable rather than falsely answerable."""
    settings = _settings()
    deps = _deps(settings=settings, bq=_bq(), openai=FakeOpenAIClient())
    result = ResearchResult(
        candidate_answer=SUMMARY, terminal_reason="final_answer"
    )

    unscoped = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c", history=[], question="q"
        ),
        deps=deps,
    )
    unscoped.turn_intent = "explore"
    scoped = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c",
            history=[],
            question="q",
            scope_package_ids=(PKG,),
        ),
        deps=deps,
    )

    assert "scope_packages" not in assemble_explore_inputs(unscoped, result)
    scoped_inputs = assemble_explore_inputs(scoped, result)
    assert scoped_inputs["scope_packages"] == [
        {"package_id": PKG, "title": None}
    ]
