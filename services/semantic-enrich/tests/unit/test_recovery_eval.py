"""The guided-recovery regression lock — deterministic tier.

Every case in `recovery_eval.CASES` that does not need a real
classification is driven end to end through the pipeline on scripted
fakes, and graded on shape: outcome tag, evidence footer, suggestion
count, derivation presence, event ordering, and the global ban on a
substantive answer being replaced by a clarifying question.

Free, offline, and runs on every change. That is the point — the
milestone's promises are shape promises, and a lock that costs a dollar
a run is a lock that gets skipped.
"""
from __future__ import annotations

import json
import math
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent import recovery_eval
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import (
    PipelineOutcome,
    run_turn_collected,
)
from semantic_enrich.core.agent.recovery_eval import (
    CASES,
    RecoveryCase,
    deterministic_cases,
    grade,
)
from semantic_enrich.core.agent.verify import AnswerFitVerifier
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
TITLE = "Supplementary Estimates B, 2025-26"
SURRENDER = (
    "The search did not return columns specific to air travel "
    "expenditures. I recommend checking with the relevant government "
    "departments for more detailed disclosures."
)
SUMMARY = (
    "This dataset covers departmental spending authorities for "
    "2025-26 across 30 columns and 4,120 rows."
)
CLARIFY_Q = "Which program or department did you have in mind?"

_SUM_SQL = (
    "SELECT SUM(CAST(JSON_VALUE(r.row, '$.c1') AS FLOAT64)) AS total "
    "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
)


# ── harness ──


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


def _bq(*, weak: bool = False, columns: int = 30) -> FakeBqClient:
    bq = FakeBqClient()
    for _ in range(3):
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
                    # 0.95 distance → 0.05 similarity, below the 0.30
                    # floor; 0.1 → 0.9, comfortably above.
                    "distance": 0.95 if weak else 0.1,
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
                    "row_json": json.dumps(
                        {f"c{i}": str(i) for i in range(columns)}
                    ),
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


def _check(action: str, *, fits: bool = False) -> dict[str, Any]:
    return {
        "fits": fits,
        "confidence": 0.95,
        "gap": "columns specific to air travel",
        "action": action,
        "retry_hint": None,
    }


def _observe(outcome: PipelineOutcome) -> recovery_eval.Observation:
    record = next(
        e.record
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    )
    offers = [
        e for e in outcome.events if isinstance(e, agent_events.Suggestions)
    ]
    return recovery_eval.Observation(
        outcome=str(record["outcome"]),
        message=outcome.final_message,
        suggestion_count=sum(len(e.items) for e in offers),
        derivation_count=sum(
            1
            for e in outcome.events
            if isinstance(e, agent_events.DerivationEvent)
        ),
        event_types=tuple(e.event_type for e in outcome.events),
    )


def _run(
    case: RecoveryCase,
    *,
    script: list[dict[str, Any]],
    checks: list[dict[str, Any]] | None = None,
    bq: FakeBqClient | None = None,
    settings: Settings | None = None,
) -> recovery_eval.Observation:
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=script,
        structured_responses=checks,
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question=case.question,
            scope_package_ids=case.scope_package_ids,
            turn_records=[
                {"v": 1, "outcome": o} for o in case.prior_outcomes
            ],
        ),
        deps=_deps(
            settings=settings or _settings(),
            bq=bq or _bq(),
            openai=openai,
        ),
    )
    return _observe(outcome)


# ── one runner per case ──


def _run_case(case: RecoveryCase) -> recovery_eval.Observation:
    search = _call("c1", "search_datasets", {"query": "air travel"})
    listing = _call("c2", "list_documents", {"package_ids": [PKG]})

    if case.id in ("air-travel-2025", "chain-cap"):
        return _run(
            case,
            script=[search, listing, {"content": SURRENDER}],
            checks=[_check("answer", fits=True)],
        )

    if case.id == "air-travel-summarize":
        return _run(case, script=[listing, {"content": SUMMARY}])

    if case.id in ("air-travel-total", "clean-total"):
        bq = _bq()
        bq.bounded_default = BoundedQueryResult(
            rows=[{"total": 4_200_000.0}],
            total_bytes_billed=1024,
            slot_ms=1,
            elapsed_ms=5,
            timed_out=False,
            error=None,
        )
        script = [
            listing if case.scope_package_ids else search,
            *([] if case.scope_package_ids else [listing]),
            _call("c3", "run_sql", {"sql": _SUM_SQL, "rationale": "sum"}),
            {"content": "The total was $4.2M."},
        ]
        return _run(
            case,
            script=script,
            checks=[_check("answer", fits=True)],
            bq=bq,
        )

    if case.id == "below-floor":
        # Weak retrieval, no SQL, and a checker that asks the user to
        # narrow down — the one case where a clarify replacement is the
        # correct outcome.
        return _run(
            case,
            script=[search, {"content": CLARIFY_Q}],
            checks=[_check("clarify")],
            bq=_bq(weak=True),
        )

    raise AssertionError(f"no runner for case {case.id!r}")


# ── the lock ──


@pytest.mark.parametrize(
    "case", deterministic_cases(), ids=lambda c: c.id
)
def test_recovery_case(case: RecoveryCase) -> None:
    result = grade(case, _run_case(case))
    assert result.passed, f"{case.id}: " + "; ".join(result.failures)


def test_no_case_ships_a_clarify_replacement_by_accident() -> None:
    """The headline bug, asserted across the whole fixture rather than
    per case — so it cannot be reintroduced by someone adding a case
    without having read the parent doc."""
    for case in deterministic_cases():
        if not case.forbid_clarify_replacement:
            continue
        observed = _run_case(case)
        assert not observed.message.startswith(
            recovery_eval.CLARIFY_MARKER
        ), case.id


def test_the_fixture_keeps_its_negative_cases() -> None:
    """Half the cases are negative on purpose: *not* offering is as much
    of the design as offering, and it is the half that rots silently. A
    lint on the count so a negative case cannot be quietly dropped."""
    negatives = {
        c.id for c in CASES if c.expect_suggestions == (0, 0)
    }
    assert negatives == {
        # A working answer needs no recovery.
        "clean-total",
        # A scoped numeric follow-up is an answer, not an invitation.
        "air-travel-total",
        # Below the retrieval floor there is nothing worth exploring.
        "below-floor",
        # The treadmill guard.
        "chain-cap",
        # Corpus-wide questions are meta, not exploration.
        "meta-boundary",
    }
    # Five of eight. If this drops, someone has removed a guard rather
    # than a case.
    assert len(negatives) * 2 > len(CASES)


def test_the_case_count_is_pinned() -> None:
    assert len(CASES) == 8
    assert len(deterministic_cases()) == 6


def test_m6_non_regression_case_demands_a_derivation() -> None:
    """The single most important row in the fixture. If guided recovery
    ever becomes a route to an untraced number, this is what catches
    it."""
    case = recovery_eval.case_by_id("air-travel-total")
    assert case.expect_derivation is True
    assert case.scope_package_ids  # a *scoped* numeric follow-up
    assert _run_case(case).derivation_count >= 1


def test_grader_reports_every_failure_not_just_the_first() -> None:
    case = recovery_eval.case_by_id("air-travel-2025")
    result = grade(
        case,
        recovery_eval.Observation(
            outcome="answered",
            message="no footer here",
            suggestion_count=0,
            derivation_count=0,
            event_types=("turn_record", "suggestions"),
        ),
    )
    assert not result.passed
    joined = " ".join(result.failures)
    assert "outcome" in joined
    assert "footer" in joined
    assert "suggestions" in joined
    assert "ordering" in joined
