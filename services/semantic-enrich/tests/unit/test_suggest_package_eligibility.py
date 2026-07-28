"""Which packages each kind may point at, and the column-browse gate.

The eligibility split is measured, not assumed. On the 11.1 baseline
retrieval was sound on every surrender but the loop had opened a
package on only one of four, so requiring every offer to point at an
*opened* package would have fired on a quarter of surrenders — screening
out turns where the loop stopped one tool call short, not turns with
nothing to offer.
"""
from __future__ import annotations

from typing import Any

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
from semantic_enrich.core.agent.suggest import build_suggestions
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

LISTED = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
RANKED = "1a2b3c4d-5566-7788-99aa-bbccddeeff00"


def _ctx(*, listed: tuple[str, ...], **overrides: Any) -> TurnContext:
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
            conversation_id="c1", history=[], question="q"
        ),
        deps=deps,
    )
    ctx.trace.searches.append({"query": "q", "retrieval_quality": "ok"})
    for i, pid in enumerate(listed):
        ctx.trace.packages_researched.append(pid)
        ctx.state.doc_package[f"doc-{i}"] = pid
        ctx.state.doc_columns[f"doc-{i}"] = ["a", "b"]
    return ctx


def _pkg(
    pid: str, columns: int | None, *, unnamed: bool = False
) -> EvidencePackage:
    return EvidencePackage(
        package_id=pid,
        title=f"Dataset {pid[:4]}",
        column_count=columns,
        headers_unnamed=unnamed,
    )


def _build(ctx: TurnContext, *packages: EvidencePackage) -> list[Any]:
    return build_suggestions(
        ctx,
        ResearchResult(candidate_answer="x", terminal_reason="final_answer"),
        SearchEvidence(
            packages=packages, queries_tried=("q",), truncated=0
        ),
        "no_data",
    )


# ── §3.1 eligibility ──


def test_a_ranked_only_package_yields_a_summary_and_nothing_else() -> None:
    """This is the 75%-of-surrenders case the split exists to serve.
    A summary is honest over a package the loop never opened; the other
    kinds promise contents we do not know."""
    out = _build(_ctx(listed=()), _pkg(RANKED, None))
    assert [s.kind for s in out] == ["summarize_dataset"]
    assert out[0].package_ids == (RANKED,)


def test_a_listed_package_yields_the_contents_promising_kinds_too() -> None:
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, 312))
    assert [s.kind for s in out] == [
        "summarize_dataset",
        "list_columns",
        "sample_rows",
    ]


def test_contents_kinds_skip_past_a_ranked_package_to_a_listed_one() -> None:
    """Ordering puts listed packages first, but a summary may still lead
    with a ranked one; the other kinds must not silently inherit it."""
    out = _build(
        _ctx(listed=(LISTED,)), _pkg(RANKED, None), _pkg(LISTED, 312)
    )
    by_kind = {s.kind: s.package_ids for s in out}
    assert by_kind["summarize_dataset"] == (RANKED,)
    assert by_kind["list_columns"] == (LISTED,)
    assert by_kind["sample_rows"] == (LISTED,)


# ── §3.2 the column-browse gate ──


def test_list_columns_absent_below_the_threshold() -> None:
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, 19))
    assert "list_columns" not in [s.kind for s in out]


def test_list_columns_present_at_the_threshold() -> None:
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, 20))
    assert "list_columns" in [s.kind for s in out]


def test_list_columns_absent_when_the_size_is_unknown() -> None:
    """`None` means the package was never opened, so we cannot say how
    big it is — and never guess."""
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, None))
    assert "list_columns" not in [s.kind for s in out]


def test_the_threshold_is_configurable() -> None:
    out = _build(
        _ctx(listed=(LISTED,), agent_suggest_min_columns=5),
        _pkg(LISTED, 6),
    )
    assert "list_columns" in [s.kind for s in out]


def test_the_browse_chip_carries_no_filter_term() -> None:
    """The regression that would reintroduce the failed query: any term
    available here comes from the searches that just failed, so a chip
    built on one re-runs the failure."""
    ctx = _ctx(listed=(LISTED,))
    ctx.trace.searches.clear()
    ctx.trace.searches.append(
        {
            "query": "housing grant approvals by province since 2020",
            "retrieval_quality": "ok",
        }
    )
    out = _build(ctx, _pkg(LISTED, 312))
    browse = next(s for s in out if s.kind == "list_columns")
    for field in (browse.label, browse.question):
        assert "housing grant approvals" not in field
        assert "matching" not in field
    assert "312 columns" in browse.label
    assert "grouped by theme" in browse.question


def test_no_emitted_text_is_drawn_from_this_turns_searches() -> None:
    ctx = _ctx(listed=(LISTED,))
    ctx.trace.searches.clear()
    ctx.trace.searches.append(
        {"query": "SENTINELQUERY", "retrieval_quality": "ok"}
    )
    for s in _build(ctx, _pkg(LISTED, 312)):
        assert "SENTINELQUERY" not in s.label
        assert "SENTINELQUERY" not in s.question


# ── regressions from the first live pass ──


def test_no_summary_offer_for_a_package_already_in_scope() -> None:
    """Observed in the browser: after accepting "Summarize X" the reply
    carried "Summarize X" again, directly beneath the summary. A scoped
    turn opens and describes its packages by construction, so re-offering
    the summary is an offer to redo what the user just watched."""
    ctx = _ctx(listed=(LISTED,))
    ctx.scope_package_ids = (LISTED,)
    out = _build(ctx, _pkg(LISTED, 312))
    kinds = [s.kind for s in out]
    assert "summarize_dataset" not in kinds
    # The drill-downs are genuine next steps from a summary and stay.
    assert "sample_rows" in kinds


def test_a_scoped_turn_still_offers_a_summary_of_a_different_package() -> None:
    ctx = _ctx(listed=(LISTED,))
    ctx.scope_package_ids = (RANKED,)
    out = _build(ctx, _pkg(LISTED, 312))
    assert "summarize_dataset" in [s.kind for s in out]


def test_browse_chip_declines_a_package_of_generated_headers() -> None:
    """Seen live: a listed document whose header row never parsed
    surfaced as `__col_1 … __col_7`. The chip rests on a human scanning
    the names and recognising the bucket the question lives in — on
    placeholders there is nothing to recognise, and "Show all N
    columns" would promise an inventory and deliver noise. Size alone
    cannot see this."""
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, 30, unnamed=True))
    kinds = [s.kind for s in out]
    assert "list_columns" not in kinds
    # The other kinds still apply — sample rows shows the *values*,
    # which is exactly what rescues a document with unreadable headers.
    assert "sample_rows" in kinds


def test_a_readable_package_is_still_browsable() -> None:
    out = _build(_ctx(listed=(LISTED,)), _pkg(LISTED, 30, unnamed=False))
    assert "list_columns" in [s.kind for s in out]
