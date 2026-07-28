"""The suggestions event through the pipeline.

Emit-only: nothing consumes it yet, so what matters is that it appears
in the right place with the right payload, that a turn with nothing to
offer emits none at all, and that the contract with the scoped-turn
validator holds — a suggestion whose `package_ids` the next turn would
drop is an offer that silently does nothing when clicked.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent import scope
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import (
    PipelineOutcome,
    run_turn_collected,
)
from semantic_enrich.core.agent.verify import AnswerFitVerifier
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
TITLE = "Supplementary Estimates B, 2025-26"
SURRENDER = (
    "The search did not return columns specific to air travel. I "
    "recommend checking with the relevant departments."
)


def _settings(**overrides: Any) -> Settings:
    overrides.setdefault("agent_verify_mode", "act")
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _deps(
    *, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient
) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-v2-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
        verifier=AnswerFitVerifier.from_settings(settings),
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _bq(*, weak: bool = False, columns: int = 30) -> FakeBqClient:
    bq = FakeBqClient()
    for _ in range(3):
        bq.register_query(
            "VECTOR_SEARCH",
            [
                {
                    "package_id": PKG,
                    "title": TITLE,
                    "summary": "estimates",
                    "grain": None,
                    "measures": [],
                    "dimensions": [],
                    "distance": 0.95 if weak else 0.1,
                }
            ],
        )
    bq.register_query(
        "FROM `proj.raw.documents`",
        [
            {
                "document_id": "doc-1",
                "package_id": PKG,
                "title": TITLE,
                "source_url": "http://x",
                "row_count": 4120,
            }
        ],
    )
    row = {f"c{i}": str(i) for i in range(columns)}
    bq.register_bounded_query(
        "TO_JSON_STRING",
        BoundedQueryResult(
            rows=[
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": __import__("json").dumps(row),
                }
            ],
            total_bytes_billed=64,
            slot_ms=1,
            elapsed_ms=1,
            timed_out=False,
            error=None,
        ),
    )
    return bq


def _call(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def _fits() -> dict[str, Any]:
    return {
        "fits": True,
        "confidence": 0.9,
        "gap": None,
        "action": "answer",
        "retry_hint": None,
    }


def _run(
    *,
    settings: Settings | None = None,
    bq: FakeBqClient | None = None,
    script: list[dict[str, Any]] | None = None,
) -> PipelineOutcome:
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=script
        or [
            _call("c1", "search_datasets", {"query": "air travel"}),
            _call("c2", "list_documents", {"package_ids": [PKG]}),
            {"content": SURRENDER},
        ],
        structured_responses=[_fits()],
    )
    return run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="how much on air travel?",
        ),
        deps=_deps(
            settings=settings or _settings(), bq=bq or _bq(), openai=openai
        ),
    )


def _suggestions(outcome: PipelineOutcome) -> list[agent_events.Suggestions]:
    return [
        e
        for e in outcome.events
        if isinstance(e, agent_events.Suggestions)
    ]


def test_event_ordering_answer_then_trace_then_offers() -> None:
    outcome = _run()
    kinds = [e.event_type for e in outcome.events]
    assert kinds.index("message_delta") < kinds.index("suggestions")
    assert kinds.index("suggestions") < kinds.index("turn_record")
    assert kinds.count("suggestions") == 1


def test_payload_shape() -> None:
    events = _suggestions(_run())
    assert len(events) == 1
    items = events[0].items
    assert 1 <= len(items) <= 3
    for item in items:
        assert set(item) == {"kind", "label", "question", "package_ids"}
        assert item["label"] and item["question"]
        assert item["package_ids"] == [PKG]
        assert len(item["label"]) <= 60


def test_weak_retrieval_emits_no_event_at_all() -> None:
    """Not an empty list — no event. The false-invitation guard."""
    outcome = _run(bq=_bq(weak=True))
    assert _suggestions(outcome) == []
    assert "suggestions" not in [e.event_type for e in outcome.events]


def test_answered_turn_emits_no_event() -> None:
    sql = (
        "SELECT SUM(CAST(JSON_VALUE(r.row, '$.c1') AS FLOAT64)) AS total "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 4_200_000.0}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    outcome = _run(
        bq=bq,
        script=[
            _call("c1", "search_datasets", {"query": "air travel"}),
            _call("c2", "list_documents", {"package_ids": [PKG]}),
            _call("c3", "run_sql", {"sql": sql, "rationale": "sum"}),
            {"content": "It was $4.2M."},
        ],
    )
    assert _suggestions(outcome) == []


def test_kill_switch_removes_the_event() -> None:
    off = _run(settings=_settings(agent_suggestions_enabled=False))
    on = _run()
    assert _suggestions(off) == []
    assert [e.event_type for e in off.events] == [
        e.event_type for e in on.events if e.event_type != "suggestions"
    ]


def test_every_suggestion_round_trips_into_a_scoped_request() -> None:
    """The contract between this PRD and the scoped-turn one, and the
    thing most likely to drift: an offer whose ids the next turn's
    validator drops is an offer that does nothing when clicked."""
    items = _suggestions(_run())[0].items
    assert items
    for item in items:
        ids = tuple(item["package_ids"])
        assert scope.sanitize(list(ids)) == ids
        request = ChatRequest(
            conversation_id="c1",
            history=[],
            question=item["question"],
            scope_package_ids=ids,
        )
        assert request.scope_package_ids == ids


def test_event_survives_the_sse_round_trip() -> None:
    event = _suggestions(_run())[0]
    frame = event.to_sse_frame()
    parsed = agent_events.from_sse_frame(frame)
    assert isinstance(parsed, agent_events.Suggestions)
    assert parsed.items == event.items
