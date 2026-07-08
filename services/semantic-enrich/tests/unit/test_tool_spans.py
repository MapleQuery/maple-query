"""Tool-span wrapping in `agent_tools.dispatch` + span digests."""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_tools
from semantic_enrich.core.agent_tools import (
    InvalidToolArgsError,
    LoopState,
    ToolContext,
)
from tests.unit.conftest import FakeBraintrustModule


def _ctx(trace_parent: str | None = "export:turn:1") -> ToolContext:
    return ToolContext(
        bq=object(),
        openai_client=object(),
        settings=Settings(),
        state=LoopState(conversation_id="c", turn_id="t", question="q"),
        emit=lambda event: None,
        trace_parent=trace_parent,
    )


def test_dispatch_wraps_tool_in_named_child_span(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    result_payload = {
        "candidates": [
            {"package_id": "p1", "title": "secret title", "distance": 0.2}
        ]
    }
    monkeypatch.setitem(
        agent_tools._IMPLS,
        "search_datasets",
        lambda *, ctx, args: result_payload,
    )

    args = {"query": "housing", "k": 5}
    result = agent_tools.dispatch(
        ctx=_ctx(), tool_name="search_datasets", args=args
    )

    assert result is result_payload
    assert len(fake_braintrust.spans) == 1
    span = fake_braintrust.spans[0]
    assert span.name == "tool.search_datasets"
    assert span.ended is True
    assert span.kwargs["parent"] == "export:turn:1"
    assert span.kwargs["input"] == args

    output = span.logs[-1]["output"]
    assert output["status"] == "ok"
    assert output["candidate_count"] == 1
    # Candidate lists reduce to identifiers + distance — no summaries,
    # titles, or row payloads on spans.
    assert output["candidates"] == [{"package_id": "p1", "distance": 0.2}]


def test_tool_exception_still_closes_span_with_error_status(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    def exploding(*, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        raise InvalidToolArgsError("missing_or_empty: 'query'")

    monkeypatch.setitem(agent_tools._IMPLS, "search_datasets", exploding)

    with pytest.raises(InvalidToolArgsError):
        agent_tools.dispatch(
            ctx=_ctx(), tool_name="search_datasets", args={}
        )

    span = fake_braintrust.spans[0]
    assert span.ended is True
    assert span.logs[-1]["output"]["status"] == "tool_error"
    assert "query" in span.logs[-1]["output"]["message"]


def test_dispatch_untraced_when_tracing_disabled(
    monkeypatch, tracing_disabled
) -> None:
    monkeypatch.setitem(
        agent_tools._IMPLS, "search_datasets", lambda *, ctx, args: {"ok": 1}
    )
    result = agent_tools.dispatch(
        ctx=_ctx(), tool_name="search_datasets", args={"query": "x"}
    )
    assert result == {"ok": 1}


def test_dispatch_untraced_when_master_switch_off(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    monkeypatch.setitem(
        agent_tools._IMPLS, "search_datasets", lambda *, ctx, args: {"ok": 1}
    )
    ctx = _ctx()
    ctx.settings = Settings().model_copy(
        update={"agent_trace_sessions": False}
    )
    agent_tools.dispatch(ctx=ctx, tool_name="search_datasets", args={})
    assert fake_braintrust.spans == []


def test_unknown_tool_raises_without_span(
    fake_braintrust: FakeBraintrustModule,
) -> None:
    with pytest.raises(InvalidToolArgsError, match="unknown_tool"):
        agent_tools.dispatch(ctx=_ctx(), tool_name="nope", args={})
    assert fake_braintrust.spans == []


def test_run_sql_span_carries_full_sql_and_truncated_rows(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    rows = [{"n": i} for i in range(7)]
    monkeypatch.setitem(
        agent_tools._IMPLS,
        "run_sql",
        lambda *, ctx, args: {
            "status": "ok",
            "row_count": 7,
            "bytes_billed": 1024,
            "elapsed_ms": 12,
            "rows": rows,
            "truncated": False,
        },
    )

    args = {"sql": "SELECT 1 LIMIT 100", "rationale": "count things"}
    agent_tools.dispatch(ctx=_ctx(), tool_name="run_sql", args=args)

    span = fake_braintrust.spans[0]
    assert span.kwargs["input"] == {
        "sql": "SELECT 1 LIMIT 100",
        "rationale": "count things",
    }
    output = span.logs[-1]["output"]
    assert output["status"] == "ok"
    assert output["row_count"] == 7
    assert output["rows"] == rows[:3]


# ── Digest units ──


def test_sample_rows_digest_truncates_rows() -> None:
    rows = [{"a": i} for i in range(10)]
    digest = agent_tools._result_digest("sample_rows", {"rows": rows})
    assert digest["rows"] == rows[:3]
    assert digest["row_count"] == 10


def test_list_documents_digest_drops_column_lists() -> None:
    digest = agent_tools._result_digest(
        "list_documents",
        {
            "documents": [
                {
                    "document_id": "d1",
                    "package_id": "p1",
                    "row_count": 9,
                    "title": "t",
                    "columns": ["a", "b", "c"],
                }
            ]
        },
    )
    assert digest["documents"] == [
        {
            "document_id": "d1",
            "package_id": "p1",
            "row_count": 9,
            "column_count": 3,
        }
    ]
    assert digest["document_count"] == 1


def test_guard_rejected_run_sql_digest_keeps_reason() -> None:
    digest = agent_tools._result_digest(
        "run_sql",
        {"status": "guard_rejected", "reason": "not_single_select", "sql_final": "…"},
    )
    assert digest == {"reason": "not_single_select"}
