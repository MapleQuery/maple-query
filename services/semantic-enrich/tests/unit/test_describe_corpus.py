"""describe_corpus: metadata-only corpus stats with an in-process TTL
cache. The rows count comes from table metadata — `raw.rows` is never
scanned."""
from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    agent_tools.reset_corpus_stats_cache()


def _settings(**overrides: object) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    ).model_copy(update=overrides)


def _ctx(
    *, bq: FakeBqClient, settings: Settings | None = None
) -> agent_tools.ToolContext:
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    events: list[agent_events.AgentEvent] = []
    return agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=settings or _settings(),
        state=state,
        emit=events.append,
    )


def _bq() -> FakeBqClient:
    bq = FakeBqClient()
    bq.table_num_rows_by_ref["proj.raw.rows"] = 198_000_000
    bq.register_query(
        "AS packages",
        [
            {
                "packages": 1234,
                "documents_loaded": 5678,
                "latest_load_at": datetime(
                    2026, 7, 1, 12, 34, 56, tzinfo=UTC
                ),
            }
        ],
    )
    return bq


def test_schema_stable() -> None:
    schemas = {
        s["function"]["name"]: s["function"] for s in agent_tools.tool_schemas()
    }
    corpus = schemas["describe_corpus"]
    assert corpus["parameters"]["properties"] == {}
    assert corpus["parameters"]["additionalProperties"] is False


def test_result_shape_and_no_rows_table_scan() -> None:
    bq = _bq()
    ctx = _ctx(bq=bq)
    result = agent_tools.run_describe_corpus(ctx=ctx, args={})
    assert result["packages"] == 1234
    assert result["documents_loaded"] == 5678
    assert result["rows_total"] == 198_000_000
    assert result["latest_load_at"] == "2026-07-01T12:34:56+00:00"
    assert "open.canada.ca" in result["corpus_description"]
    # rows_total must come from table metadata; no issued query may
    # target raw.rows.
    assert bq.table_num_rows_calls == ["proj.raw.rows"]
    for call in bq.calls:
        assert "raw.rows" not in call["sql"]
    assert bq.bounded_calls == []
    assert bq.dry_run_calls == []


def test_cache_honored_within_ttl() -> None:
    bq = _bq()
    ctx = _ctx(bq=bq)
    first = agent_tools.run_describe_corpus(ctx=ctx, args={})
    second = agent_tools.run_describe_corpus(ctx=ctx, args={})
    assert first == second
    # One metadata read + one stats query total.
    assert len(bq.table_num_rows_calls) == 1
    assert len(bq.calls) == 1


def test_cache_expires_after_ttl() -> None:
    bq = _bq()
    bq.register_query(
        "AS packages",
        [{"packages": 1, "documents_loaded": 2, "latest_load_at": None}],
    )
    ctx = _ctx(bq=bq, settings=_settings(agent_snapshot_refresh_seconds=0))
    agent_tools.run_describe_corpus(ctx=ctx, args={})
    second = agent_tools.run_describe_corpus(ctx=ctx, args={})
    assert len(bq.calls) == 2
    assert second["latest_load_at"] is None


def test_requires_project_id() -> None:
    settings = Settings(
        gcp_project_id=None,
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    ctx = _ctx(bq=FakeBqClient(), settings=settings)
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_describe_corpus(ctx=ctx, args={})


def test_empty_stats_row_degrades_to_zeroes() -> None:
    bq = FakeBqClient()
    ctx = _ctx(bq=bq)
    result = agent_tools.run_describe_corpus(ctx=ctx, args={})
    assert result["packages"] == 0
    assert result["documents_loaded"] == 0
    assert result["rows_total"] == 0
    assert result["latest_load_at"] is None


def test_dispatch_routes_describe_corpus() -> None:
    bq = _bq()
    ctx = _ctx(bq=bq)
    result = agent_tools.dispatch(
        ctx=ctx, tool_name="describe_corpus", args={}
    )
    assert result["packages"] == 1234


def test_span_digest_keeps_counts_drops_description() -> None:
    digest = agent_tools._result_digest(
        "describe_corpus",
        {
            "packages": 1,
            "documents_loaded": 2,
            "rows_total": 3,
            "latest_load_at": None,
            "corpus_description": "long text",
        },
    )
    assert digest == {
        "packages": 1,
        "documents_loaded": 2,
        "rows_total": 3,
        "latest_load_at": None,
    }
