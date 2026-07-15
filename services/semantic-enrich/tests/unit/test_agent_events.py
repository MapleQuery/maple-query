"""Round-trip typed events through SSE frames.

Every event type serialises through `to_sse_frame` and comes back via
`from_sse_frame` bit-for-bit. Guards the event schema; a future field
rename fails here before it reaches the FE.
"""
from __future__ import annotations

import pytest

from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_events import (
    BudgetExceeded,
    CacheHit,
    ColumnsRanked,
    CostUpdate,
    DatasetsRanked,
    DocumentsListed,
    Done,
    ErrorEvent,
    MessageDelta,
    PhaseStart,
    PlanHint,
    Reformulation,
    RetrievalStarted,
    Rows,
    SampleRows,
    SqlExecuted,
    SqlGenerated,
    SqlGuarded,
    ToolError,
    TriageResult,
    TurnRecordEvent,
    TurnStart,
    TurnTimeout,
    Verification,
    from_sse_frame,
)

CASES = [
    TurnStart(conversation_id="c1", turn_id="t1", cached=False),
    CacheHit(cache_key_prefix="abcd"),
    RetrievalStarted(query="housing", k=5),
    DatasetsRanked(candidates=[{"package_id": "pkg-1", "distance": 0.1}]),
    ColumnsRanked(
        package_ids=["pkg-1"],
        candidates=[{"package_id": "pkg-1", "column_name": "TOT_EXP"}],
    ),
    DocumentsListed(
        package_ids=["pkg-1"],
        documents=[{"document_id": "doc-1", "columns": ["Amount"]}],
    ),
    SampleRows(package_id="pkg-1", rows=[{"document_id": "doc-1", "row": {"a": 1}}]),
    SqlGenerated(sql="SELECT 1", rationale="just a test"),
    SqlGuarded(accepted=True, reason=None, sql_final="SELECT 1", dry_run_bytes=1024),
    SqlExecuted(row_count=3, bytes_billed=1024, elapsed_ms=42, sample_rows=[]),
    Rows(sql_call_id="c", rows=[{"a": 1}], is_last=True),
    MessageDelta(delta="hello"),
    CostUpdate(tokens_in_total=100, tokens_out_total=50, dollars_spent=0.0005),
    BudgetExceeded(which="tool_calls", value=6, cap=6),
    TurnTimeout(elapsed_ms=61000, cap_ms=60000),
    ToolError(tool="run_sql", message="bad sql"),
    Done(turn_id="t1", total_tool_calls=3, total_dollars=0.01, elapsed_ms=1200),
    ErrorEvent(message="boom", retryable=False, reason="test"),
    PhaseStart(phase="triage"),
    TriageResult(
        category="in_scope", confidence=0.9, elapsed_ms=12, enforced=False
    ),
    Reformulation(
        original_query="housing",
        reformulated_query="federal housing spending",
        top_similarity_before=0.31,
    ),
    Verification(
        fits=True,
        action="accept",
        confidence=0.8,
        reason="answer grounded in rows",
        enforced=False,
    ),
    PlanHint(
        records_used=[
            {"question_gist": "housing spend", "package_ids": ["pkg-1"]}
        ]
    ),
    TurnRecordEvent(
        record={"turn_id": "t1", "packages": ["pkg-1"], "loop_impl": "v2"}
    ),
]


@pytest.mark.parametrize("event", CASES)
def test_sse_round_trip(event: agent_events.AgentEvent) -> None:
    frame = event.to_sse_frame()
    assert frame.startswith(f"event: {event.event_type}\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
    parsed = from_sse_frame(frame)
    assert parsed == event


def test_from_sse_frame_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        from_sse_frame("event: turn_start\n")


def test_from_sse_frame_rejects_unknown_type() -> None:
    frame = "event: made_up\ndata: {}\n\n"
    with pytest.raises(ValueError):
        from_sse_frame(frame)


def test_to_dict_carries_type_tag() -> None:
    event = MessageDelta(delta="hi")
    payload = event.to_dict()
    assert payload["type"] == "message_delta"
    assert payload["delta"] == "hi"
