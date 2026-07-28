"""The evidence footer stops saying "most unnamed" once the names exist.

This is the note that made the whole problem visible: a real session
rendered `(17 columns, most unnamed)` and the honest question followed.
Once recovery names those columns the note has to retire itself, or the
footer keeps reporting a defect that has been fixed — and the browse chip,
which keys off the same flag, keeps refusing to offer a dataset that is
now perfectly browsable.

Both directions are asserted on the *same document*, because the claim is
about a transition, not about two unrelated shapes.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.evidence import (
    collect_evidence,
    compose_footer,
    mostly_unnamed,
)
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

# The observed shape: two real names, seven positional keys.
COLUMNS = [
    "Forward_Sortation_Area",
    "Number_of_Unique_Applicants",
    *[f"__col_{i}" for i in range(1, 8)],
]
# What the document's own header row calls those seven.
RECOVERED = {
    "__col_1": "Under 25",
    "__col_2": "25-34",
    "__col_3": "35-44",
    "__col_4": "45-54",
    "__col_5": "55-64",
    "__col_6": "65+",
    "__col_7": "Total",
}


def _ctx(**overrides: Any) -> TurnContext:
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
    return TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1", history=[], question="housing top-up total?"
        ),
        deps=deps,
    )


def _turn(*, recovered: bool) -> TurnContext:
    ctx = _ctx()
    ctx.state.search_results["q"] = {
        "candidates": [
            {"package_id": "pkg-1", "title": "One-time top-up to the CHB"}
        ]
    }
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    ctx.state.doc_columns["d1"] = list(COLUMNS)
    if recovered:
        ctx.state.doc_recovered_names["d1"] = dict(RECOVERED)
        ctx.state.doc_header_row["d1"] = 2
    return ctx


def _result() -> ResearchResult:
    return ResearchResult(
        candidate_answer="no data",
        terminal_reason="final_answer",
        packages_cited=["pkg-1"],
    )


def test_without_recovery_the_package_reads_as_mostly_unnamed() -> None:
    package = collect_evidence(_turn(recovered=False), _result()).packages[0]
    assert package.headers_unnamed is True
    assert package.column_count == 9


def test_with_recovery_the_note_clears_itself() -> None:
    package = collect_evidence(_turn(recovered=True), _result()).packages[0]
    assert package.headers_unnamed is False
    # Same document, same column count — only the names changed.
    assert package.column_count == 9


def test_the_rendered_footer_drops_the_note() -> None:
    before = compose_footer(collect_evidence(_turn(recovered=False), _result()))
    after = compose_footer(collect_evidence(_turn(recovered=True), _result()))
    assert "most unnamed" in before
    assert "most unnamed" not in after
    # The count survives the transition; only the caveat goes.
    assert "9 columns" in before
    assert "9 columns" in after


def test_the_browse_chip_inherits_the_same_transition() -> None:
    """The chip reads the same flag, so a recovered dataset becomes
    offerable without a second definition of "unnamed"."""
    assert mostly_unnamed(_turn(recovered=False), "pkg-1") is True
    assert mostly_unnamed(_turn(recovered=True), "pkg-1") is False


def test_a_partial_recovery_that_stays_mostly_unnamed_keeps_the_note() -> None:
    """Recovery is not all-or-nothing. Naming one of seven leaves the
    package majority-positional, and the footer should still say so."""
    ctx = _turn(recovered=False)
    ctx.state.doc_recovered_names["d1"] = {"__col_1": "Under 25"}
    package = collect_evidence(ctx, _result()).packages[0]
    assert package.headers_unnamed is True


def test_recovered_names_do_not_inflate_the_column_count() -> None:
    """A recovered name replaces a positional key rather than joining
    it — otherwise the footer would double-count every recovered column
    and promise a browse surface twice its real size."""
    counts = {
        recovered: collect_evidence(
            _turn(recovered=recovered), _result()
        ).packages[0].column_count
        for recovered in (False, True)
    }
    assert counts[False] == counts[True] == len(COLUMNS)
