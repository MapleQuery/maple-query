"""JSONPath auto-quoting inside run_sql.

Table-driven over the prompt's original example inventory. The tool
rewrites what it can rewrite unambiguously and leaves everything else
for the guard.
"""
from __future__ import annotations

import math

import pytest

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent_tools import autoquote_json_paths
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Identifier-shaped keys stay bare.
        ("$.Amount_Montant", None),
        ("$.__col_3", None),
        # The prompt's inventory of must-quote shapes.
        ("$.2020-21_Expenditures", '$."2020-21_Expenditures"'),
        ("$.Quantity_(kgs)", '$."Quantity_(kgs)"'),
        ("$.Major_Transfers(1)", '$."Major_Transfers(1)"'),
        (
            "$.Sector,_Canada,_1990_to_2024",
            '$."Sector,_Canada,_1990_to_2024"',
        ),
        # Already-quoted segments pass through untouched (idempotent).
        ('$."2020-21_Expenditures"', None),
        # Mixed: only the bare non-identifier segment gets quoted.
        ("$.a.2020-21", '$.a."2020-21"'),
        # Unparseable / unsupported shapes are left alone.
        ("$", None),
        ("$.arr[0]", None),
        ("$.", None),
        ("$x", None),
        ('$.a"b', None),
        ("$..b", None),
        ("no_dollar", None),
        ('$."unterminated', None),
    ],
)
def test_quote_json_path_table(path: str, expected: str | None) -> None:
    assert agent_tools._quote_json_path(path) == expected


def test_second_arg_span_malformed_calls() -> None:
    for sql in (
        # Unterminated string literal.
        "SELECT JSON_VALUE(r.row, '$.a-b",
        # Call runs off the end of the SQL.
        "SELECT JSON_VALUE(r.row, '$.a-b'",
    ):
        out, changed = autoquote_json_paths(sql)
        assert out == sql
        assert changed == []


def test_autoquote_rewrites_all_json_functions() -> None:
    for func in (
        "JSON_VALUE",
        "JSON_QUERY",
        "JSON_EXTRACT",
        "JSON_EXTRACT_SCALAR",
    ):
        sql = f"SELECT {func}(r.row, '$.2020-21_Exp') AS e FROM t"
        out, changed = autoquote_json_paths(sql)
        assert f"{func}(r.row, '$.\"2020-21_Exp\"')" in out
        assert changed == ["$.2020-21_Exp"]


def test_autoquote_is_idempotent() -> None:
    sql = 'SELECT JSON_VALUE(r.row, \'$."2020-21_Exp"\') AS e FROM t'
    out, changed = autoquote_json_paths(sql)
    assert out == sql
    assert changed == []


def test_autoquote_skips_non_literal_second_arg() -> None:
    for sql in (
        "SELECT JSON_VALUE(r.row, @path) AS e FROM t",
        "SELECT JSON_VALUE(r.row, CONCAT('$.', col)) AS e FROM t",
        "SELECT JSON_VALUE(r.row) AS e FROM t",
        "SELECT JSON_VALUE(r.row, '$.a-b', 'x') AS e FROM t",
        "SELECT JSON_VALUE(r.row, '$.a\\'b') AS e FROM t",
    ):
        out, changed = autoquote_json_paths(sql)
        assert out == sql
        assert changed == []


def test_autoquote_handles_nested_call_first_arg() -> None:
    sql = (
        "SELECT JSON_VALUE(JSON_QUERY(r.row, '$.outer-key'), "
        "'$.2020-21_Exp') AS e FROM t"
    )
    out, changed = autoquote_json_paths(sql)
    assert "'$.\"outer-key\"'" in out
    assert "'$.\"2020-21_Exp\"'" in out
    assert changed == ["$.outer-key", "$.2020-21_Exp"]


def test_autoquote_ignores_function_name_inside_string_literal() -> None:
    sql = "SELECT 'JSON_VALUE(x, ' AS s, '$.2020-21' AS t FROM t"
    out, changed = autoquote_json_paths(sql)
    assert out == sql
    assert changed == []


def test_run_sql_reports_json_paths_quoted_and_events_carry_rewrite() -> None:
    """Acceptance #1 shape: a bare hyphenated path executes with the
    quoted path, returns real values, and reports the correction."""
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"fiscal_year": "2020", "expenditures": "123"}],
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
        args={
            "sql": (
                "SELECT JSON_VALUE(r.row, '$.FISCAL_YEAR') AS fiscal_year, "
                "JSON_VALUE(r.row, '$.2020-21_Expenditures') AS expenditures "
                "FROM `proj.raw.rows` AS r "
                "WHERE r.document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "turn-5 housing failure replay",
        },
    )
    assert result["status"] == "ok"
    assert result["normalizations"]["json_paths_quoted"] == [
        "$.2020-21_Expenditures"
    ]
    guarded = [e for e in events if e.event_type == "sql_guarded"]
    assert len(guarded) == 1
    assert isinstance(guarded[0], agent_events.SqlGuarded)
    assert "'$.\"2020-21_Expenditures\"'" in guarded[0].sql_final
    # The executed SQL is the rewritten SQL.
    assert "'$.\"2020-21_Expenditures\"'" in bq.bounded_calls[0]


def test_run_sql_without_corrections_omits_normalizations() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"n": 1}],
        total_bytes_billed=0,
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
        args={
            "sql": (
                "SELECT 1 AS n FROM `proj.raw.rows` "
                "WHERE document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "clean sql",
        },
    )
    assert result["status"] == "ok"
    assert "normalizations" not in result
