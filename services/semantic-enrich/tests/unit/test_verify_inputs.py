"""Deterministic evidence assembly for the fit checker.

The checker sees evidence, not vibes: everything comes from the turn
trace, SQL ships shape-only (string literals blanked), and a
surrender carries the searches that justify — or fail to justify —
the no-data claim.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent.verify import assemble_inputs
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _ctx() -> TurnContext:
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=Settings(
            gcp_project_id="proj",
            openai_api_key="sk-test",  # type: ignore[arg-type]
        ),
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1", history=[], question="housing grants?"
        ),
        deps=deps,
    )


def _populated_ctx() -> TurnContext:
    ctx = _ctx()
    ctx.trace.searches.append(
        {
            "query": "housing grants",
            "top_similarity": 0.24,
            "retrieval_quality": "weak",
        }
    )
    ctx.trace.searches.append(
        {
            "query": "CMHC grant approvals",
            "top_similarity": 0.55,
            "retrieval_quality": "ok",
        }
    )
    ctx.state.search_results["housing grants"] = {
        "candidates": [
            {"package_id": "pkg-1", "title": "Housing Grants 2020"},
        ],
    }
    return ctx


def _answered_result() -> ResearchResult:
    return ResearchResult(
        candidate_answer="grants totalled $X.",
        terminal_reason="final_answer",
        sql_runs=[
            {
                "sql": (
                    "SELECT SUM(x) FROM r WHERE document_id IN "
                    "('doc-secret-1', 'doc-secret-2') GROUP BY p LIMIT 10"
                ),
                "status": "ok",
                "row_count": 100,
                "null_ratio_warning": {"column": "x", "ratio": 0.9},
            },
            {
                "sql": "SELECT bad FROM r LIMIT 1",
                "status": "execution_error",
                "row_count": None,
                "null_ratio_warning": None,
            },
        ],
        packages_cited=["pkg-1", "pkg-unknown"],
        columns_referenced=["FISCAL_YEAR", "Amount"],
    )


def test_answered_turn_evidence_shape() -> None:
    inputs = assemble_inputs(_populated_ctx(), _answered_result())
    assert inputs["question"] == "housing grants?"
    assert inputs["candidate_answer"] == "grants totalled $X."
    assert inputs["answer_kind"] == "answer"
    assert inputs["columns_referenced"] == ["FISCAL_YEAR", "Amount"]
    assert inputs["question_asks_for"] is None


def test_sql_shapes_blank_literals_and_drop_failed_runs() -> None:
    inputs = assemble_inputs(_populated_ctx(), _answered_result())
    # Only the successful run ships, shape-only: document ids (string
    # literals) are blanked, structure survives.
    assert len(inputs["sql_shapes"]) == 1
    shape = inputs["sql_shapes"][0]
    assert "doc-secret-1" not in shape
    assert "SUM(x)" in shape
    assert "GROUP BY p" in shape


def test_result_summary_carries_rowcount_and_advisory() -> None:
    inputs = assemble_inputs(_populated_ctx(), _answered_result())
    assert inputs["result_summary"]["row_count"] == 100
    assert inputs["result_summary"]["null_ratio_warning"] == {
        "column": "x",
        "ratio": 0.9,
    }


def test_datasets_used_resolves_titles_from_search_results() -> None:
    inputs = assemble_inputs(_populated_ctx(), _answered_result())
    assert inputs["datasets_used"] == [
        {"package_id": "pkg-1", "title": "Housing Grants 2020"},
        {"package_id": "pkg-unknown", "title": None},
    ]


def test_surrender_carries_searches_tried() -> None:
    result = ResearchResult(
        candidate_answer="no data found.",
        terminal_reason="final_answer",
        sql_runs=[],
    )
    inputs = assemble_inputs(_populated_ctx(), result)
    assert inputs["answer_kind"] == "no_data"
    assert inputs["sql_shapes"] == []
    assert inputs["result_summary"]["row_count"] is None
    assert inputs["searches_tried"] == [
        {"query": "housing grants", "top_similarity": 0.24},
        {"query": "CMHC grant approvals", "top_similarity": 0.55},
    ]


def test_empty_trace_assembles_cleanly() -> None:
    result = ResearchResult(
        candidate_answer="", terminal_reason="final_answer"
    )
    inputs: dict[str, Any] = assemble_inputs(_ctx(), result)
    assert inputs["answer_kind"] == "no_data"
    assert inputs["datasets_used"] == []
    assert inputs["searches_tried"] == []
