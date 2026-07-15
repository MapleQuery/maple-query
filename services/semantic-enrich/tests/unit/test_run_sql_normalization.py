"""Table-reference normalization inside run_sql.

Bare and placeholder `raw.rows` references get rewritten to the fully
qualified form before the guard runs; string literals are never
touched; the guard stays the authority on which tables are allowed.
"""
from __future__ import annotations

import math

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent_tools import normalize_table_references
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def _ctx(
    *, bq: FakeBqClient | None = None
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    bq = bq or FakeBqClient()
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
    return ctx, events


def test_bare_raw_rows_rewritten() -> None:
    sql = "SELECT 1 AS n FROM raw.rows WHERE document_id IN ('d') LIMIT 10"
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert "FROM `proj.raw.rows`" in out
    assert rewritten == ["raw.rows"]


def test_backticked_raw_rows_rewritten() -> None:
    sql = "SELECT 1 AS n FROM `raw.rows` WHERE document_id IN ('d') LIMIT 10"
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert "FROM `proj.raw.rows`" in out
    assert rewritten == ["`raw.rows`"]


def test_placeholder_project_rewritten() -> None:
    for ref in (
        "`<project>.raw.rows`",
        "`<project_id>.raw.rows`",
        "`PROJECT_ID.raw.rows`",
        "PROJECT_ID.raw.rows",
        "`project_id`.raw.rows",
    ):
        sql = (
            f"SELECT 1 AS n FROM {ref} "
            "WHERE document_id IN ('d') LIMIT 10"
        )
        out, rewritten = normalize_table_references(sql, settings=_settings())
        assert "FROM `proj.raw.rows`" in out, ref
        assert rewritten == [ref]


def test_join_target_rewritten() -> None:
    sql = (
        "SELECT 1 AS n FROM `proj.raw.rows` AS a JOIN raw.rows AS b "
        "ON a.document_id = b.document_id "
        "WHERE a.document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert "JOIN `proj.raw.rows` AS b" in out
    assert rewritten == ["raw.rows"]


def test_correctly_qualified_reference_untouched() -> None:
    sql = (
        "SELECT 1 AS n FROM `proj.raw.rows` "
        "WHERE document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert out == sql
    assert rewritten == []


def test_string_literal_containing_raw_rows_untouched() -> None:
    sql = (
        "SELECT 1 AS n FROM `proj.raw.rows` "
        "WHERE JSON_VALUE(row, '$.name') = 'FROM raw.rows' "
        "AND document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert out == sql
    assert rewritten == []


def test_no_project_configured_is_a_noop() -> None:
    settings = Settings(
        gcp_project_id=None,
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    sql = "SELECT 1 AS n FROM raw.rows WHERE document_id IN ('d') LIMIT 10"
    out, rewritten = normalize_table_references(sql, settings=settings)
    assert out == sql
    assert rewritten == []


def test_run_sql_reports_tables_rewritten_and_executes_rewritten() -> None:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"n": 1}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    ctx, events = _ctx(bq=bq)
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": (
                "SELECT 1 AS n FROM raw.rows "
                "WHERE document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "test",
        },
    )
    assert result["status"] == "ok"
    assert result["normalizations"]["tables_rewritten"] == ["raw.rows"]
    guarded = [e for e in events if e.event_type == "sql_guarded"]
    assert len(guarded) == 1
    assert isinstance(guarded[0], agent_events.SqlGuarded)
    # The evidence rail must show the SQL that actually ran.
    assert "`proj.raw.rows`" in guarded[0].sql_final
    assert "`proj.raw.rows`" in bq.bounded_calls[0]


def test_guard_still_rejects_disallowed_tables_after_normalization() -> None:
    """Normalization never widens what the guard accepts."""
    ctx, _events = _ctx()
    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": (
                "SELECT 1 AS n FROM raw.rows AS r "
                "JOIN `proj.other.table` AS o ON r.document_id = o.id "
                "WHERE r.document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "test",
        },
    )
    assert result["status"] == "guard_rejected"
    assert "other" in str(result["reason"])
    # The normalization note still rides along so the model learns the
    # corrected table form even on a rejection.
    assert result["normalizations"]["tables_rewritten"] == ["raw.rows"]


def test_masking_handles_escaped_quotes() -> None:
    masked = agent_tools._mask_string_literals(
        r"SELECT 'it\'s raw.rows' FROM raw.rows"
    )
    assert "FROM raw.rows" in masked
    assert "it" not in masked.split("FROM")[0]


def test_comment_apostrophe_does_not_corrupt_string_literal() -> None:
    """An apostrophe inside a `--` comment must not flip the masker's
    quote parity: the string literal after it stays untouched."""
    sql = (
        "SELECT 1 AS n FROM `proj.raw.rows` -- don't scan too much\n"
        "WHERE JSON_VALUE(row, '$.note') = 'FROM raw.rows' "
        "AND document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert out == sql
    assert rewritten == []


def test_bare_ref_after_comment_apostrophe_still_rewritten() -> None:
    sql = (
        "SELECT 1 AS n -- what's the total\n"
        "FROM raw.rows WHERE document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert "FROM `proj.raw.rows`" in out
    assert rewritten == ["raw.rows"]
    # The comment itself is preserved verbatim.
    assert "-- what's the total" in out


def test_ref_inside_comments_untouched() -> None:
    for sql in (
        "SELECT 1 AS n FROM `proj.raw.rows` "
        "/* was: FROM raw.rows */ WHERE document_id IN ('d') LIMIT 10",
        "SELECT 1 AS n FROM `proj.raw.rows` -- was: FROM raw.rows\n"
        "WHERE document_id IN ('d') LIMIT 10",
        "SELECT 1 AS n FROM `proj.raw.rows` # was: FROM raw.rows\n"
        "WHERE document_id IN ('d') LIMIT 10",
    ):
        out, rewritten = normalize_table_references(sql, settings=_settings())
        assert out == sql, sql
        assert rewritten == [], sql


def test_masking_blanks_comment_contents() -> None:
    masked = agent_tools._mask_string_literals(
        "SELECT 1 -- it's a note\nFROM raw.rows /* don't */ WHERE x = 'y'"
    )
    assert "it" not in masked
    assert "don" not in masked
    assert "FROM raw.rows" in masked
    # Length-preserving so spans map back to the original.
    assert len(masked) == len(
        "SELECT 1 -- it's a note\nFROM raw.rows /* don't */ WHERE x = 'y'"
    )


def test_backtick_identifier_with_apostrophe_does_not_break_scan() -> None:
    """A quote inside a backtick identifier must not open a phantom
    string literal; refs after it are still rewritten and backticked
    table refs stay matchable in the masked text."""
    masked = agent_tools._mask_string_literals(
        "SELECT `it's odd` FROM `raw.rows` WHERE x = 'y'"
    )
    assert "`raw.rows`" in masked
    sql = (
        "SELECT `it's odd` AS o FROM raw.rows "
        "WHERE document_id IN ('d') LIMIT 10"
    )
    out, rewritten = normalize_table_references(sql, settings=_settings())
    assert "FROM `proj.raw.rows`" in out
    assert rewritten == ["raw.rows"]
