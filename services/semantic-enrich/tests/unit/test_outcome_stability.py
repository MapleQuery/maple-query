"""The evidence footer must never change the turn's outcome tag.

`_outcome` detects a clarify partly by testing `"?" in message`, and
the footer is appended to that message — so gating the footer on an
outcome derived from the footered message would be circular. The
pipeline resolves it by computing the outcome once, on the pre-footer
message, and threading it to both the footer decision and the turn
record.

That resolution is only safe while the footer provably cannot flip the
tag. This file pins the invariant directly, so a future footer template
cannot silently break it: for every turn shape, `_outcome` on the
pre-footer message equals `_outcome` on the post-footer message.
"""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.evidence import (
    EvidencePackage,
    SearchEvidence,
    collect_evidence,
    compose_footer,
)
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent.pipeline import _compose, _outcome
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _ctx(
    *, triage: str = "in_scope", clarify_steer: bool = False
) -> TurnContext:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
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
            conversation_id="c1", history=[], question="q"
        ),
        deps=deps,
    )
    ctx.triage_category = triage
    ctx.state.clarify_steer_issued = clarify_steer
    # A surrender that searched: the state every footer is built from.
    ctx.state.search_results["air travel"] = {
        "candidates": [
            # Adversarial on purpose — a question mark in the evidence
            # is the one thing that could flip the tag.
            {"package_id": "p1", "title": "Who Flew Where? 2025-26"},
            {"package_id": "p2", "title": "Public Accounts"},
        ]
    }
    ctx.state.doc_columns["d1"] = [f"c{i}" for i in range(312)]
    ctx.state.doc_package["d1"] = "p1"
    ctx.trace.packages_researched.append("p1")
    ctx.trace.searches.append({"query": "what covers air travel?"})
    return ctx


def _result(*, sql_ok: bool, answer: str) -> ResearchResult:
    return ResearchResult(
        candidate_answer=answer,
        terminal_reason="final_answer",
        sql_runs=[{"status": "ok" if sql_ok else "error"}],
        packages_cited=["p1"],
    )


# Every turn shape that reaches the finish step, including the two the
# clarify detector keys on.
_TURNS: list[tuple[str, dict[str, Any], bool, str]] = [
    ("surrender", {}, False, "No dataset covers air travel spending."),
    (
        "surrender_after_steer",
        {"clarify_steer": True},
        False,
        "No dataset covers air travel spending.",
    ),
    (
        "clarify",
        {"clarify_steer": True},
        False,
        "Could you narrow that down to a department?",
    ),
    ("answered", {}, True, "The total was $4.2M."),
    (
        "answered_question_shaped",
        {"clarify_steer": True},
        True,
        "The total was $4.2M. Want the per-province split?",
    ),
]


@pytest.mark.parametrize(
    ("name", "ctx_kwargs", "sql_ok", "message"),
    _TURNS,
    ids=[t[0] for t in _TURNS],
)
def test_footer_never_changes_the_outcome(
    name: str, ctx_kwargs: dict[str, Any], sql_ok: bool, message: str
) -> None:
    ctx = _ctx(**ctx_kwargs)
    result = _result(sql_ok=sql_ok, answer=message)

    before = _outcome(ctx, message=message, result=result)
    composed = _compose(
        ctx, message=message, result=result, outcome=before
    )
    after = _outcome(ctx, message=composed, result=result)

    assert after == before, name


def test_the_footer_is_actually_exercised_by_the_invariant() -> None:
    """Guard against the parametrized test passing vacuously — at least
    one of those turns must genuinely grow a footer."""
    ctx = _ctx()
    result = _result(sql_ok=False, answer="No dataset covers this.")
    composed = _compose(
        ctx,
        message=result.candidate_answer,
        result=result,
        outcome="no_data",
    )
    assert "**What I searched:**" in composed


def test_composed_footer_carries_no_question_mark() -> None:
    """The mechanical half of the invariant: the footer cannot
    introduce the character the clarify detector keys on, whatever the
    dataset titles and model-authored queries happen to contain."""
    ctx = _ctx()
    footer = compose_footer(collect_evidence(ctx, None))
    assert "Who Flew Where 2025-26" in footer
    assert "?" not in footer


def test_footer_never_removes_a_question_mark() -> None:
    footer = compose_footer(
        SearchEvidence(
            packages=(
                EvidencePackage(
                    package_id="p1", title="D", column_count=None
                ),
            ),
            queries_tried=(),
            truncated=0,
        )
    )
    assert "?" in "Which department?" + footer
