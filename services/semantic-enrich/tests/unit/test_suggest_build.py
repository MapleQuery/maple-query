"""The suggestion builder: templates, priority, cap, dedupe.

Composed, never generated — so every label and question here is a pure
function of turn state and can be asserted exactly.
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
from semantic_enrich.core.agent.suggest import (
    MAX_LABEL_CHARS,
    build_suggestions,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

BIG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
SMALL = "1a2b3c4d-5566-7788-99aa-bbccddeeff00"


def _ctx(
    *,
    listed: tuple[str, ...] = (BIG,),
    searched_ok: bool = True,
    turn_records: list[dict[str, Any]] | None = None,
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
            question="air travel spend?",
            turn_records=turn_records or [],
        ),
        deps=deps,
    )
    ctx.trace.searches.append(
        {
            "query": "air travel",
            "retrieval_quality": "ok" if searched_ok else "weak",
        }
    )
    for i, pid in enumerate(listed):
        ctx.trace.packages_researched.append(pid)
        doc = f"doc-{i}"
        ctx.state.doc_package[doc] = pid
        ctx.state.doc_columns[doc] = [f"col_{n}" for n in range(30)]
    return ctx


def _evidence(*packages: EvidencePackage) -> SearchEvidence:
    return SearchEvidence(
        packages=packages, queries_tried=("air travel",), truncated=0
    )


def _big(title: str = "Supplementary Estimates B") -> EvidencePackage:
    return EvidencePackage(package_id=BIG, title=title, column_count=312)


def _result() -> ResearchResult:
    return ResearchResult(
        candidate_answer="no data", terminal_reason="final_answer"
    )


def _build(ctx: TurnContext, evidence: SearchEvidence) -> list[Any]:
    return build_suggestions(ctx, _result(), evidence, "no_data")


def test_full_render_on_a_listed_package() -> None:
    out = _build(_ctx(), _evidence(_big()))
    kinds = [s.kind for s in out]
    assert kinds == ["summarize_dataset", "list_columns", "sample_rows"]
    assert out[0].label == "Summarize Supplementary Estimates B"
    assert out[0].question == (
        "Summarize what data is in Supplementary Estimates B — what it "
        "covers, its time range, and its main columns."
    )
    assert out[1].label == "Show all 312 columns in Supplementary Estimates B"
    assert out[1].question == (
        "List the columns in Supplementary Estimates B, grouped by "
        "theme, so I can see what the dataset actually contains."
    )
    assert all(s.package_ids == (BIG,) for s in out)


def test_priority_order_and_cap_at_three() -> None:
    out = _build(_ctx(agent_suggestions_max=2), _evidence(_big()))
    assert [s.kind for s in out] == ["summarize_dataset", "list_columns"]


def test_labels_stay_within_budget_at_the_worst_case() -> None:
    """Asserted against a title at 11.1's 70-char truncation ceiling —
    the longest string that can reach here — rather than a convenient
    short one."""
    long_title = "Supplementary Estimates " + "B" * 46
    assert len(long_title) == 70
    out = _build(_ctx(), _evidence(_big(title=long_title)))
    assert out
    for s in out:
        assert len(s.label) <= MAX_LABEL_CHARS, (s.kind, s.label)
    # Only the variable part is cut; the fixed words survive intact.
    assert out[1].label.startswith("Show all 312 columns in ")
    assert out[1].label.endswith("…")


def test_untitled_package_falls_back_to_its_id() -> None:
    out = _build(
        _ctx(),
        _evidence(
            EvidencePackage(package_id=BIG, title=None, column_count=312)
        ),
    )
    assert BIG in out[0].label


def test_group_total_absent_without_a_known_monetary_column() -> None:
    out = _build(_ctx(), _evidence(_big()))
    assert "group_total" not in [s.kind for s in out]


def _with_money(ctx: TurnContext) -> TurnContext:
    ctx.state.doc_columns["doc-0"] = ["Department", "total_expenditure"]
    ctx.state.column_metadata["total_expenditure"] = {
        "semantic_type": "currency",
        "description": "Total expenditure in dollars.",
    }
    return ctx


def test_group_total_renders_for_a_monetary_column_on_that_package() -> None:
    out = build_suggestions(
        _with_money(_ctx(agent_suggestions_max=4)),
        _result(),
        _evidence(_big()),
        "no_data",
    )
    totals = [s for s in out if s.kind == "group_total"]
    assert len(totals) == 1
    assert totals[0].label == "Total total_expenditure"
    # No dimension is named at composition time. Hardcoding one
    # ("grouped by department") produced a turn that failed outright on
    # a dataset with no such column — a guess about schema, which is the
    # invention this kind's monetary check exists to prevent.
    assert "department" not in totals[0].question
    assert "pick a column it actually has" in totals[0].question


def test_group_total_is_crowded_out_on_a_large_listed_dataset() -> None:
    """Documents a live consequence of the spec's priority order rather
    than hiding it.

    On a listed package above the column threshold the first three kinds
    always fill the cap, so `group_total` — the only kind that leads to
    a number — can never appear there. It surfaces only on *small*
    listed datasets, which is arguably backwards: a 312-column spending
    file is exactly where "total by department" is worth offering.

    Pinned as-specified. If the ordering is revisited, this test is the
    one that should change and be seen changing.
    """
    out = build_suggestions(
        _with_money(_ctx()), _result(), _evidence(_big()), "no_data"
    )
    assert [s.kind for s in out] == [
        "summarize_dataset",
        "list_columns",
        "sample_rows",
    ]

    # The same package under the column threshold leaves room, and the
    # kind appears.
    small = EvidencePackage(
        package_id=BIG, title="Small Estimates", column_count=4
    )
    out_small = build_suggestions(
        _with_money(_ctx()), _result(), _evidence(small), "no_data"
    )
    assert [s.kind for s in out_small] == [
        "summarize_dataset",
        "sample_rows",
        "group_total",
    ]


def test_a_monetary_column_on_another_package_does_not_qualify() -> None:
    """`column_metadata` is keyed by bare column name, so membership has
    to be resolved through the doc→package map rather than assumed."""
    ctx = _ctx(listed=(BIG,))
    ctx.state.doc_package["doc-other"] = SMALL
    ctx.state.doc_columns["doc-other"] = ["other_amount"]
    ctx.state.column_metadata["other_amount"] = {
        "semantic_type": "currency",
        "description": "dollars",
    }
    out = build_suggestions(ctx, _result(), _evidence(_big()), "no_data")
    assert "group_total" not in [s.kind for s in out]


def test_dedupe_by_kind_and_package() -> None:
    out = _build(_ctx(), _evidence(_big(), _big()))
    keys = [(s.kind, s.package_ids) for s in out]
    assert len(keys) == len(set(keys))


def test_every_question_is_non_empty_and_carries_a_package() -> None:
    for s in _build(_ctx(), _evidence(_big())):
        assert s.question.strip()
        assert s.package_ids
        assert all(pid for pid in s.package_ids)
