"""Derivation capture wired through the live research phase.

Drives research.run directly (the derivation lives on ResearchResult,
not yet an SSE event) and asserts a scalar aggregate turn produces one
faithful Derivation, while non-numeric turns and the kill switch
produce none.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent import research
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _deps(*, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-test",
        cache=ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60),
        snapshot_hash_provider=lambda: "snap-0",
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def _run_research(ctx: TurnContext) -> ResearchResult:
    gen = research.run(ctx, hints=[])
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]


def _bq_with_dataset_and_doc(row_count: int = 1412) -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "title": "Supplementary Estimates (A), 2020-21",
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
                "package_id": "pkg-1",
                "title": "Supplementary Estimates (A), 2020-21",
                "source_url": "http://x",
                "row_count": row_count,
            }
        ],
    )
    # Column keys for list_documents come from a bounded raw.rows keys
    # job (TO_JSON_STRING(row)); give doc-1 an 'Amount' key so the
    # run_sql doc/column pairing check passes.
    bq.register_bounded_query(
        "TO_JSON_STRING",
        BoundedQueryResult(
            rows=[{"document_id": "doc-1", "row_index": 0, "row_json": '{"Amount": "100"}'}],
            total_bytes_billed=64,
            slot_ms=1,
            elapsed_ms=1,
            timed_out=False,
            error=None,
        ),
    )
    return bq


_SUM_SQL = (
    "SELECT SUM(CAST(JSON_VALUE(r.row, '$.Amount') AS FLOAT64)) AS total "
    "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
)


def _aggregate_script() -> FakeOpenAIClient:
    return FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            _call("s1", "search_datasets", {"query": "spend 2020"}),
            _call("l1", "list_documents", {"package_ids": ["pkg-1"]}),
            _call("r1", "run_sql", {"sql": _SUM_SQL, "rationale": "sum"}),
            {"content": "Total spending 2020-21 was about $8."},
        ],
    )


def _ctx(*, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient) -> TurnContext:
    return TurnContext.begin(
        request=ChatRequest(conversation_id="c1", history=[], question="total spending 2020-21?"),
        deps=_deps(settings=settings, bq=bq, openai=openai),
    )


def test_aggregate_turn_captures_one_derivation() -> None:
    bq = _bq_with_dataset_and_doc(row_count=1412)
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 8.2}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    ctx = _ctx(settings=_settings(), bq=bq, openai=_aggregate_script())
    result = _run_research(ctx)

    assert len(result.derivations) == 1
    deriv = result.derivations[0]
    assert deriv.complete is True
    assert deriv.aggregation == "SUM"
    assert deriv.value_columns == ("Amount",)
    assert deriv.result_value == 8.2
    # The $8 mechanism: a scalar sum (1 output row) drawn from ~1400 rows.
    assert deriv.row_count == 1
    assert deriv.source_row_estimate == 1412
    assert deriv.source_packages == ("pkg-1",)
    assert deriv.dataset_titles == ("Supplementary Estimates (A), 2020-21",)
    # 'Amount' reads as monetary but no scale cue -> unknown, not dollars.
    assert deriv.unit_scale == "unknown"


def test_kill_switch_disables_capture() -> None:
    bq = _bq_with_dataset_and_doc()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 8.2}], total_bytes_billed=1, slot_ms=1, elapsed_ms=1, timed_out=False, error=None
    )
    ctx = _ctx(
        settings=_settings(agent_derivation_capture=False),
        bq=bq,
        openai=_aggregate_script(),
    )
    result = _run_research(ctx)
    assert result.derivations == []


def test_non_numeric_turn_captures_nothing() -> None:
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {"tool_calls": [{"id": "s1", "name": "search_datasets", "arguments": {"query": "nothing"}}]},
            {"content": "I couldn't find a dataset for that."},
        ],
    )
    ctx = _ctx(settings=_settings(), bq=bq, openai=openai)
    result = _run_research(ctx)
    assert result.derivations == []
