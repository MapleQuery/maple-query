"""DerivationEvent emission through the pipeline: one curated event per
value-bearing derivation, in the answer phase, curated (no raw SQL)."""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        agent_verify_mode="off",
        **overrides,
    )


def _deps(*, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60),
        snapshot_hash_provider=lambda: "snap-0",
    )


def _unit_vec(_t: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _call(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def _bq() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [{"package_id": "pkg-1", "title": "Estimates 2020-21", "summary": "s",
          "grain": None, "measures": [], "dimensions": [], "distance": 0.1}],
    )
    bq.register_query(
        "FROM `proj.raw.documents`",
        [{"document_id": "doc-1", "package_id": "pkg-1",
          "title": "Estimates 2020-21", "source_url": "u", "row_count": 1400}],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING",
        BoundedQueryResult(
            rows=[{"document_id": "doc-1", "row_index": 0, "row_json": '{"Amount": "1"}'}],
            total_bytes_billed=1, slot_ms=1, elapsed_ms=1, timed_out=False, error=None,
        ),
    )
    return bq


_SUM = (
    "SELECT SUM(CAST(JSON_VALUE(r.row, '$.Amount') AS FLOAT64)) AS total "
    "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
)


def test_numeric_turn_emits_one_curated_derivation_event() -> None:
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 8.2}], total_bytes_billed=1, slot_ms=1, elapsed_ms=1,
        timed_out=False, error=None,
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            _call("s", "search_datasets", {"query": "spend 2020"}),
            _call("l", "list_documents", {"package_ids": ["pkg-1"]}),
            _call("r", "run_sql", {"sql": _SUM, "rationale": "sum"}),
            {"content": "Total spending 2020-21 was about $8."},
        ],
    )
    outcome = run_turn_collected(
        request=ChatRequest(conversation_id="c1", history=[], question="total 2020-21?"),
        deps=_deps(settings=_settings(), bq=bq, openai=openai),
    )
    kinds = [e.event_type for e in outcome.events]
    assert kinds.count("derivation") == 1
    # Emitted in the answer phase: after the message, before the record.
    di = kinds.index("derivation")
    assert kinds.index("message_delta") < di < kinds.index("turn_record")

    deriv = next(
        e for e in outcome.events if isinstance(e, agent_events.DerivationEvent)
    )
    assert deriv.aggregation == "SUM"
    assert deriv.result_value == 8.2
    assert deriv.source_row_estimate == 1400
    assert "unknown_units" in deriv.flags  # 'Amount' scale unresolved
    # Curated: no raw SQL string or row content in the payload.
    assert "SELECT" not in str(deriv.to_dict())


def test_non_numeric_turn_emits_no_derivation_event() -> None:
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            _call("s", "search_datasets", {"query": "nothing"}),
            {"content": "I could not find a dataset for that."},
        ],
    )
    outcome = run_turn_collected(
        request=ChatRequest(conversation_id="c1", history=[], question="q?"),
        deps=_deps(settings=_settings(), bq=bq, openai=openai),
    )
    assert not any(
        isinstance(e, agent_events.DerivationEvent) for e in outcome.events
    )
