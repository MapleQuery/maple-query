"""TurnRecord construction and ingest validation.

The record is the deterministic memory primitive: built from the turn
context with no LLM involved, validated defensively on the way back in
— a corrupt client entry is dropped with a log, never an error.
"""
from __future__ import annotations

import json
import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent import records
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _ctx(question: str = "How much airplane spending since 2020?") -> TurnContext:
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
            conversation_id="c1", history=[], question=question
        ),
        deps=deps,
    )


def _result(*, sql_ok: bool = True) -> ResearchResult:
    return ResearchResult(
        candidate_answer="answer.",
        terminal_reason="final_answer",
        sql_runs=(
            [
                {
                    "sql": "SELECT 1 FROM r LIMIT 5",
                    "status": "ok",
                    "row_count": 5,
                    "null_ratio_warning": None,
                }
            ]
            if sql_ok
            else []
        ),
        packages_cited=["pkg-1"],
        columns_referenced=["Airfare"],
    )


def test_question_gist_normalizes() -> None:
    gist = records.question_gist(
        "How much DID Canadian politicians spend, on airplanes?!"
    )
    assert gist == "canadian politicians spend airplanes"


def test_build_answered_turn() -> None:
    ctx = _ctx()
    ctx.state.search_results["q"] = {
        "candidates": [{"package_id": "pkg-1", "title": "Travel"}]
    }
    ctx.state.known_document_ids.update({"doc-2", "doc-1"})
    ctx.trace.searches.append(
        {
            "query": "airplane spending",
            "top_similarity": 0.6,
            "retrieval_quality": "ok",
        }
    )
    record = records.build(
        ctx,
        message="Total was $X." + "x" * 400,
        result=_result(),
        outcome="answered",
    )
    assert record["v"] == records.RECORD_VERSION
    assert record["outcome"] == "answered"
    assert record["packages"] == [
        {"package_id": "pkg-1", "title": "Travel"}
    ]
    assert record["columns_used"] == ["Airfare"]
    assert record["document_ids"] == ["doc-1", "doc-2"]
    assert record["sql"] == "SELECT 1 FROM r LIMIT 5"
    assert record["row_count"] == 5
    assert len(record["answer_digest"]) == 300
    assert record["snapshot_hash"] == "snap-0"
    # Round-trips as JSON and re-validates.
    assert records.validate(json.loads(json.dumps(record))) is not None


def test_build_short_circuit_turn_has_no_research_fields() -> None:
    ctx = _ctx()
    ctx.triage_category = "off_scope"
    record = records.build(
        ctx, message="deflection.", result=None, outcome="deflected"
    )
    assert record["packages"] == []
    assert record["sql"] is None
    assert record["row_count"] is None
    assert record["category"] == "off_scope"


def test_build_unknown_outcome_downgrades_to_error() -> None:
    ctx = _ctx()
    record = records.build(
        ctx, message="m", result=None, outcome="bogus"
    )
    assert record["outcome"] == "error"


def _valid_record() -> dict[str, Any]:
    ctx = _ctx()
    return records.build(
        ctx, message="answer.", result=_result(), outcome="answered"
    )


def test_validate_accepts_built_records_and_tolerates_extras() -> None:
    record = _valid_record()
    record["some_future_field"] = {"nested": True}
    assert records.validate(record) is not None


def test_validate_drops_malformed() -> None:
    base = _valid_record()
    assert records.validate("not a dict") is None
    assert records.validate({**base, "v": 2}) is None
    assert records.validate({**base, "outcome": "great"}) is None
    assert records.validate({**base, "question": 42}) is None
    assert records.validate({**base, "packages": "pkg-1"}) is None
    assert records.validate({**base, "packages": [{"title": "no id"}]}) is None
    assert records.validate({**base, "sql": "x" * 3000}) is None
    assert (
        records.validate({**base, "columns_used": ["a"] * 51}) is None
    )


def test_validate_drops_oversized() -> None:
    base = _valid_record()
    base["question_gist"] = "x" * 1000
    base["padding"] = ["y" * 1000] * 20
    assert records.validate(base) is None


def test_sanitize_incoming_keeps_newest_and_drops_bad() -> None:
    good = _valid_record()
    out = records.sanitize_incoming(
        [{"junk": True}, good, "nope", good],  # type: ignore[list-item]
        max_records=3,
    )
    assert out == [good, good]
