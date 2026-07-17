"""The weak/ok retrieval verdict on tool results.

The floor lives in settings and the comparison lives in the tool, so
the model only ever reads `retrieval_quality` and `guidance` — these
tests pin the envelope at the floor boundary, the guidance switching
as the reformulation budget is spent, the duplicate-query guard, and
the empty-after-filter list_documents mapping to the same signal.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,
    )


def _ctx(
    *,
    bq: FakeBqClient | None = None,
    settings: Settings | None = None,
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq or FakeBqClient(),
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        ),
        settings=settings or _settings(),
        state=agent_tools.LoopState(
            conversation_id="c1", turn_id="t1", question="q"
        ),
        emit=events.append,
    )
    return ctx, events


def _package_row(*, package_id: str, distance: float) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "title": f"Title {package_id}",
        "summary": "s",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": distance,
    }


def _search(
    ctx: agent_tools.ToolContext, query: str = "housing"
) -> dict[str, Any]:
    return agent_tools.run_search_datasets(ctx=ctx, args={"query": query})


# ── verdict at the floor boundary ──


def test_similarity_at_floor_is_ok() -> None:
    # distance 0.70 → similarity 0.30 == floor → not weak.
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH", [_package_row(package_id="p1", distance=0.70)]
    )
    ctx, _ = _ctx(bq=bq)
    result = _search(ctx)
    assert result["top_similarity"] == 0.3
    assert result["retrieval_quality"] == "ok"
    assert "guidance" not in result
    assert ctx.state.weak_signal_seen is False


def test_similarity_below_floor_is_weak_with_guidance() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH", [_package_row(package_id="p1", distance=0.76)]
    )
    ctx, _ = _ctx(bq=bq)
    result = _search(ctx)
    assert result["top_similarity"] == 0.24
    assert result["retrieval_quality"] == "weak"
    assert "Reformulate once" in result["guidance"]
    assert ctx.state.weak_signal_seen is True
    assert ctx.state.search_outcomes[-1].weak is True


def test_zero_candidates_is_weak() -> None:
    ctx, _ = _ctx()
    result = _search(ctx)
    assert result["candidates"] == []
    assert result["top_similarity"] is None
    assert result["retrieval_quality"] == "weak"
    assert result["guidance"]


# ── guidance switching across the reformulation budget ──


def test_cap_reached_guidance_steers_to_clarify() -> None:
    ctx, _ = _ctx()
    ctx.state.reformulations_used = 1  # cap (default 1) already spent
    result = _search(ctx)
    assert result["retrieval_quality"] == "weak"
    assert "Do not search again" in result["guidance"]
    assert "ONE clarifying question" in result["guidance"]
    assert ctx.state.clarify_steer_issued is True


def test_prior_clarify_drops_the_clarify_option() -> None:
    ctx, _ = _ctx()
    ctx.state.reformulations_used = 1
    ctx.state.prior_clarify = True
    result = _search(ctx)
    assert "do not ask another clarifying question" in result["guidance"]
    assert ctx.state.clarify_steer_issued is True


# ── duplicate-query guard ──


def test_duplicate_query_replays_cached_result_with_rephrase_guidance() -> None:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH", [_package_row(package_id="p1", distance=0.8)]
    )
    ctx, events = _ctx(bq=bq)
    first = _search(ctx, query="housing  spend")
    # Case-folded, whitespace-collapsed equality → same search.
    second = _search(ctx, query="Housing Spend")
    assert second["candidates"] == first["candidates"]
    assert second["guidance"].startswith("identical query")
    # Replayed from cache: no second embed/retrieval, no repeat events.
    kinds = [e.event_type for e in events]
    assert kinds.count("retrieval_started") == 1
    assert kinds.count("datasets_ranked") == 1
    assert len(ctx.state.search_outcomes) == 1


def test_duplicate_replay_does_not_mutate_cached_entry() -> None:
    ctx, _ = _ctx()
    _search(ctx, query="housing")
    _search(ctx, query="housing")
    cached = ctx.state.search_results[
        agent_tools.normalize_search_query("housing")
    ]
    assert "identical query" not in str(cached.get("guidance", ""))


# ── empty-after-filter list_documents mapped to the same signal ──


def _register_docs(bq: FakeBqClient, columns: list[str]) -> None:
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "Doc One",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-1", "columns": columns}],
    )


def test_unsatisfiable_required_columns_carries_weak_guidance() -> None:
    bq = FakeBqClient()
    _register_docs(bq, ["A", "B"])
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": ["NOPE"]},
    )
    assert result["required_columns_unsatisfiable"] is True
    assert "Reconsider your package choice" in result["guidance"]
    assert ctx.state.weak_signal_seen is True


def test_all_docs_quality_demoted_carries_weak_guidance() -> None:
    bq = FakeBqClient()
    _register_docs(bq, ["__col_1", "__col_2", "__col_3"])
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    assert all("quality" in d for d in result["documents"])
    assert "guidance" in result
    assert ctx.state.weak_signal_seen is True


def test_usable_docs_carry_no_guidance() -> None:
    bq = FakeBqClient()
    _register_docs(bq, ["FISCAL_YEAR", "Amount"])
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    assert "guidance" not in result
    assert ctx.state.weak_signal_seen is False


def test_cap_reached_list_documents_guidance_also_switches() -> None:
    bq = FakeBqClient()
    _register_docs(bq, ["A"])
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    ctx.state.reformulations_used = 1
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": ["NOPE"]},
    )
    assert "Do not search again" in result["guidance"]
