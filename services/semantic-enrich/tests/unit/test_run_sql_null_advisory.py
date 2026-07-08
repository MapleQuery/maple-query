"""NULL-ratio advisory on run_sql results.

The query succeeded — status stays "ok" — but mostly-NULL columns get
flagged so the model re-examines its column/document choice instead of
reporting "no data".
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent_tools import compute_null_ratio_warning
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


_PLAIN_SQL = (
    "SELECT JSON_VALUE(r.row, '$.x') AS x FROM `proj.raw.rows` AS r "
    "WHERE r.document_id IN ('d') LIMIT 100"
)


def test_mostly_null_column_warns() -> None:
    """The trace evidence: 99/100 NULL."""
    rows: list[dict[str, Any]] = [
        {"expenditures_2020_21": None, "org": "x"} for _ in range(99)
    ]
    rows.append({"expenditures_2020_21": "1", "org": "y"})
    warning = compute_null_ratio_warning(
        sql=_PLAIN_SQL, rows=rows, settings=_settings()
    )
    assert warning is not None
    assert warning["columns"] == {"expenditures_2020_21": 0.99}
    assert "mostly NULL" in warning["message"]


def test_document_id_excluded() -> None:
    rows: list[dict[str, Any]] = [
        {"document_id": None, "x": "v"} for _ in range(10)
    ]
    warning = compute_null_ratio_warning(
        sql=_PLAIN_SQL, rows=rows, settings=_settings()
    )
    assert warning is None


def test_below_threshold_is_silent() -> None:
    rows: list[dict[str, Any]] = [{"x": None} for _ in range(7)] + [
        {"x": "v"} for _ in range(3)
    ]
    warning = compute_null_ratio_warning(
        sql=_PLAIN_SQL, rows=rows, settings=_settings()
    )
    assert warning is None


def test_zero_rows_do_not_warn() -> None:
    """A zero-row result is a different signal, not a NULL-ratio one."""
    warning = compute_null_ratio_warning(
        sql=_PLAIN_SQL, rows=[], settings=_settings()
    )
    assert warning is None


def test_ungrouped_aggregate_columns_excluded() -> None:
    """SUM over zero matching rows returns one all-NULL row — visible
    to the model already, so the advisory stays quiet."""
    sql = (
        "SELECT SUM(SAFE_CAST(JSON_VALUE(r.row, '$.amt') AS FLOAT64)) "
        "AS total FROM `proj.raw.rows` AS r "
        "WHERE r.document_id IN ('d') LIMIT 100"
    )
    warning = compute_null_ratio_warning(
        sql=sql, rows=[{"total": None}], settings=_settings()
    )
    assert warning is None


def test_grouped_aggregate_columns_still_warn() -> None:
    """The turn-5 shape: GROUP BY with an all-NULL aggregate column is
    exactly the silent failure the advisory exists for."""
    sql = (
        "SELECT JSON_VALUE(r.row, '$.fy') AS fy, "
        "SUM(SAFE_CAST(JSON_VALUE(r.row, '$.exp') AS FLOAT64)) AS total "
        "FROM `proj.raw.rows` AS r WHERE r.document_id IN ('d') "
        "GROUP BY fy LIMIT 100"
    )
    rows: list[dict[str, Any]] = [
        {"fy": str(2000 + i), "total": None} for i in range(10)
    ]
    warning = compute_null_ratio_warning(
        sql=sql, rows=rows, settings=_settings()
    )
    assert warning is not None
    assert warning["columns"] == {"total": 1.0}


def test_unparseable_sql_still_computes_ratios() -> None:
    rows: list[dict[str, Any]] = [{"x": None} for _ in range(10)]
    warning = compute_null_ratio_warning(
        sql="not really sql (", rows=rows, settings=_settings()
    )
    assert warning is not None
    assert warning["columns"] == {"x": 1.0}


def test_scalar_aggregate_columns_empty_parse() -> None:
    assert agent_tools._scalar_aggregate_columns("") == set()


def test_scalar_aggregate_columns_none_tree(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        agent_tools.sqlglot, "parse_one", lambda *_a, **_k: None
    )
    assert agent_tools._scalar_aggregate_columns("SELECT 1") == set()


def test_execution_error_result_still_carries_normalizations() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[], total_bytes_billed=0, slot_ms=0,
        elapsed_ms=5001, timed_out=True, error=None,
    )
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=_settings(),
        state=state,
        emit=lambda _e: None,
    )
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": (
                "SELECT JSON_VALUE(r.row, '$.a-b') AS x FROM raw.rows AS r "
                "WHERE r.document_id IN ('d') LIMIT 100"
            ),
            "rationale": "test",
        },
    )
    assert result["status"] == "execution_error"
    assert result["reason"] == "timed_out"
    assert result["normalizations"]["json_paths_quoted"] == ["$.a-b"]
    assert result["normalizations"]["tables_rewritten"] == ["raw.rows"]


def test_run_sql_carries_warning_in_result_and_event() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"x": None} for _ in range(9)] + [{"x": "v"}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=_settings(),
        state=state,
        emit=events.append,
    )
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={"sql": _PLAIN_SQL, "rationale": "test"},
    )
    # The query succeeded; the advisory is a hint, not a failure.
    assert result["status"] == "ok"
    assert result["null_ratio_warning"]["columns"] == {"x": 0.9}
    executed = [e for e in events if e.event_type == "sql_executed"]
    assert len(executed) == 1
    assert isinstance(executed[0], agent_events.SqlExecuted)
    assert executed[0].null_ratio_warning == result["null_ratio_warning"]


def test_run_sql_clean_result_has_no_warning_key() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"x": "v"} for _ in range(5)],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=_settings(),
        state=state,
        emit=lambda _e: None,
    )
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={"sql": _PLAIN_SQL, "rationale": "test"},
    )
    assert result["status"] == "ok"
    assert "null_ratio_warning" not in result
