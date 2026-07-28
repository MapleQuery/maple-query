"""The evidence extractor: pure, over state the turn already holds.

The ordering contract is the interesting part — packages the loop
actually listed precede ranked-but-unopened candidates, because the
listed ones are the datasets it opened and can describe. Everything
else here is about degrading honestly: a package that was ranked but
never listed has no column count, and never gets a guessed one.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.evidence import (
    MAX_PACKAGES,
    MAX_QUERIES,
    collect_evidence,
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
            conversation_id="c1", history=[], question="air travel spend?"
        ),
        deps=deps,
    )


def _search(*candidates: tuple[str, str | None]) -> dict[str, Any]:
    return {
        "candidates": [
            {"package_id": pid, "title": title} for pid, title in candidates
        ]
    }


def _result(*packages: str) -> ResearchResult:
    return ResearchResult(
        candidate_answer="no data",
        terminal_reason="final_answer",
        packages_cited=list(packages),
    )


def _list_doc(
    ctx: TurnContext, *, doc: str, package: str, columns: int
) -> None:
    """Mirror what `list_documents` writes onto LoopState."""
    ctx.state.doc_columns[doc] = [f"col_{i}" for i in range(columns)]
    ctx.state.doc_package[doc] = package


def test_listed_packages_precede_ranked_only_candidates() -> None:
    ctx = _ctx()
    ctx.state.search_results["estimates"] = _search(
        ("pkg-ranked", "Public Accounts"),
        ("pkg-listed", "Supplementary Estimates B"),
    )
    _list_doc(ctx, doc="d1", package="pkg-listed", columns=312)
    ctx.trace.packages_researched.append("pkg-listed")

    evidence = collect_evidence(ctx, _result("pkg-listed"))

    assert [p.package_id for p in evidence.packages] == [
        "pkg-listed",
        "pkg-ranked",
    ]
    assert evidence.packages[0].column_count == 312
    assert evidence.packages[0].title == "Supplementary Estimates B"


def test_ranked_only_package_has_no_column_count() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Ranked Only"))

    evidence = collect_evidence(ctx, _result())

    assert len(evidence.packages) == 1
    # Never estimated: an omitted count is honest, a guessed one is the
    # class of error the numeric-trust milestone existed to remove.
    assert evidence.packages[0].column_count is None


def test_missing_doc_columns_does_not_raise() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Listed But Empty"))
    # Listed, but the doc→package map never got populated (a tool error
    # between the two side effects, or a doc with no columns).
    ctx.trace.packages_researched.append("pkg-1")

    evidence = collect_evidence(ctx, _result("pkg-1"))

    assert evidence.packages[0].column_count is None
    assert evidence.truncated == 0


def test_packages_beyond_the_cap_are_counted_not_listed() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(
        *((f"pkg-{i}", f"Dataset {i}") for i in range(7))
    )

    evidence = collect_evidence(ctx, _result())

    assert len(evidence.packages) == MAX_PACKAGES
    assert evidence.truncated == 7 - MAX_PACKAGES


def test_package_seen_in_both_sources_appears_once() -> None:
    ctx = _ctx()
    ctx.state.search_results["first"] = _search(("pkg-1", "Estimates"))
    ctx.state.search_results["second"] = _search(
        ("pkg-1", "Estimates"), ("pkg-2", "Accounts")
    )
    ctx.trace.packages_researched.append("pkg-1")

    evidence = collect_evidence(ctx, _result("pkg-1"))

    assert [p.package_id for p in evidence.packages] == ["pkg-1", "pkg-2"]
    assert evidence.truncated == 0


def test_queries_are_deduped_capped_and_ordered() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "D"))
    for query in (
        "air travel expenditures",
        "air travel expenditures",
        "travel costs 2025-26",
        "transportation and communications",
        "flight spending",
    ):
        ctx.trace.searches.append({"query": query})

    evidence = collect_evidence(ctx, _result())

    assert evidence.queries_tried == (
        "air travel expenditures",
        "travel costs 2025-26",
        "transportation and communications",
    )
    assert len(evidence.queries_tried) == MAX_QUERIES


def test_empty_turn_yields_empty_evidence() -> None:
    evidence = collect_evidence(_ctx(), None)
    assert evidence.packages == ()
    assert evidence.queries_tried == ()
    assert evidence.truncated == 0


def test_untitled_candidate_keeps_its_id() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", None))

    evidence = collect_evidence(ctx, _result())

    assert evidence.packages[0].title is None
    assert evidence.packages[0].package_id == "pkg-1"


def test_first_listed_document_is_the_representative() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Estimates"))
    ctx.trace.packages_researched.append("pkg-1")
    # `list_documents` sorts clean docs ahead of generated-header ones,
    # so the first doc recorded for a package is the one the model was
    # steered at.
    _list_doc(ctx, doc="clean", package="pkg-1", columns=312)
    _list_doc(ctx, doc="messy", package="pkg-1", columns=4)

    evidence = collect_evidence(ctx, _result("pkg-1"))

    assert evidence.packages[0].column_count == 312


def test_a_scoped_turn_resolves_titles_without_having_searched() -> None:
    """The regression that shipped: a clicked chip goes straight to
    `list_documents`, so `search_results` is empty and the only title
    available is the one that tool recorded. Reading search results
    alone rendered a raw uuid everywhere a dataset name belongs — in the
    footer and in every chip built from it."""
    ctx = _ctx()
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    ctx.state.doc_title["d1"] = "Supplementary Estimates B"
    ctx.state.doc_columns["d1"] = ["a", "b"]
    assert ctx.state.search_results == {}

    evidence = collect_evidence(ctx, _result("pkg-1"))

    assert evidence.packages[0].title == "Supplementary Estimates B"
    assert evidence.packages[0].column_count == 2


def test_search_titles_win_over_document_titles() -> None:
    """A package's own title is a better name than one document's."""
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Package Title"))
    ctx.state.doc_package["d1"] = "pkg-1"
    ctx.state.doc_title["d1"] = "Some Document, 2024"
    ctx.trace.packages_researched.append("pkg-1")

    evidence = collect_evidence(ctx, _result("pkg-1"))

    assert evidence.packages[0].title == "Package Title"


def test_column_count_spans_every_document_in_the_package() -> None:
    """The undercount seen live: a housing package reported "(3
    columns)" in the footer while its eight documents — one breakdown
    per table — carried nine distinct columns between them. Reading
    only the first document told the user something false about the
    dataset, and the browse chip built on the same number promised the
    wrong count."""
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Housing Benefit"))
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["t1"] = "pkg-1"
    ctx.state.doc_columns["t1"] = ["Applicants", "Total"]
    ctx.state.doc_package["t2"] = "pkg-1"
    ctx.state.doc_columns["t2"] = ["Gender", "Total"]  # `Total` repeats
    ctx.state.doc_package["t3"] = "pkg-1"
    ctx.state.doc_columns["t3"] = ["Age Group", "Province"]

    evidence = collect_evidence(ctx, _result("pkg-1"))

    # Deduped by name — heterogeneous documents in one package
    # routinely repeat a column.
    assert evidence.packages[0].column_count == 5


def test_documents_of_other_packages_are_not_counted() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "A"), ("pkg-2", "B"))
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    ctx.state.doc_columns["d1"] = ["a", "b"]
    ctx.state.doc_package["d2"] = "pkg-2"
    ctx.state.doc_columns["d2"] = ["c", "d", "e"]

    counts = {
        p.package_id: p.column_count
        for p in collect_evidence(ctx, _result("pkg-1")).packages
    }
    assert counts == {"pkg-1": 2, "pkg-2": 3}


def test_mostly_generated_headers_is_flagged() -> None:
    """A dataset whose header row never parsed at ingest carries
    `__col_N` placeholders instead of names. It is worth saying so: a
    dataset whose columns cannot be named cannot be queried by name, and
    a user who reads that in the footer is spared the three turns it
    otherwise takes to discover it."""
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Housing Benefit"))
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    # The real shape, from the live package: 2 named, 7 generated.
    ctx.state.doc_columns["d1"] = [
        "Forward_Sortation_Area",
        "Number_of_Unique_Applicants",
        *[f"__col_{i}" for i in range(1, 8)],
    ]

    package = collect_evidence(ctx, _result("pkg-1")).packages[0]

    assert package.column_count == 9
    assert package.headers_unnamed is True


def test_a_readable_package_is_not_flagged() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = _search(("pkg-1", "Estimates"))
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    ctx.state.doc_columns["d1"] = ["Department", "Vote", "__col_1"]

    package = collect_evidence(ctx, _result("pkg-1")).packages[0]

    assert package.headers_unnamed is False


def test_the_footer_the_chip_and_the_tool_share_one_threshold() -> None:
    """Three consumers, one definition of "unnamed" — so they cannot
    drift apart."""
    ctx = _ctx(agent_generated_header_ratio=0.9)
    ctx.state.search_results["q"] = _search(("pkg-1", "Estimates"))
    ctx.trace.packages_researched.append("pkg-1")
    ctx.state.doc_package["d1"] = "pkg-1"
    # 0.8 generated: over the default 0.5, under this turn's 0.9.
    ctx.state.doc_columns["d1"] = ["a", "b"] + [
        f"__col_{i}" for i in range(8)
    ]

    package = collect_evidence(ctx, _result("pkg-1")).packages[0]

    assert package.headers_unnamed is False
