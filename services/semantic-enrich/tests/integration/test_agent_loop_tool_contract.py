"""End-to-end 6.2 contract through `run_turn`: a scripted FakeOpenAI
turn replays the turn-5 housing failure — a bare hyphenated JSONPath
over a mostly-NULL column — and asserts the model-facing tool message
carries the auto-quote correction and the NULL-ratio advisory."""
from __future__ import annotations

import json
import math
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import (
    ChatRequest,
    LoopDeps,
    load_system_prompt,
    run_turn_collected,
)
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
    )


def _deps(
    *, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient
) -> LoopDeps:
    prompt, prompt_hash = load_system_prompt(
        settings.agent_system_prompt_path, settings
    )
    return LoopDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt=prompt,
        prompt_hash=prompt_hash,
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )


def test_autoquote_and_advisory_reach_the_model() -> None:
    settings = _settings()
    bq = FakeBqClient()
    # 9 of 10 rows NULL — above the 0.8 advisory threshold.
    bq.bounded_default = BoundedQueryResult(
        rows=[{"fy": str(2010 + i), "exp": None} for i in range(9)]
        + [{"fy": "2020", "exp": "1"}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "run_sql",
                        "arguments": {
                            "sql": (
                                "SELECT JSON_VALUE(r.row, '$.FY') AS fy, "
                                "JSON_VALUE(r.row, "
                                "'$.2020-21_Expenditures') AS exp "
                                "FROM raw.rows AS r "
                                "WHERE r.document_id IN ('doc-1') LIMIT 10"
                            ),
                            "rationale": "housing spend by fiscal year",
                        },
                    }
                ]
            },
            {"content": "Mostly-NULL result; re-examining."},
        ],
    )
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="how much did we spend on housing?",
        ),
        deps=deps,
    )

    # The tool message the model saw on its second call.
    second_call_messages: list[dict[str, Any]] = openai.chat_calls[1][
        "messages"
    ]
    tool_messages = [
        m for m in second_call_messages if m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["status"] == "ok"
    assert payload["normalizations"]["json_paths_quoted"] == [
        "$.2020-21_Expenditures"
    ]
    assert payload["normalizations"]["tables_rewritten"] == ["raw.rows"]
    assert payload["null_ratio_warning"]["columns"] == {"exp": 0.9}

    # The evidence rail (sql_guarded / executed SQL) shows the SQL that
    # actually ran: quoted path + qualified table.
    guarded = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.SqlGuarded)
    ]
    assert len(guarded) == 1
    assert "'$.\"2020-21_Expenditures\"'" in guarded[0].sql_final
    assert "`proj.raw.rows`" in guarded[0].sql_final
    executed = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.SqlExecuted)
    ]
    assert len(executed) == 1
    assert executed[0].null_ratio_warning is not None
    assert outcome.final_message.startswith("Mostly-NULL")


def test_describe_corpus_counts_against_tool_budget() -> None:
    settings = _settings()
    bq = FakeBqClient()
    bq.table_num_rows_by_ref["proj.raw.rows"] = 42
    bq.register_query(
        "AS packages",
        [{"packages": 3, "documents_loaded": 7, "latest_load_at": None}],
    )
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "describe_corpus",
                        "arguments": {},
                    }
                ]
            },
            {"content": "The corpus has 42 rows across 3 datasets."},
        ],
    )
    from semantic_enrich.core import agent_tools

    agent_tools.reset_corpus_stats_cache()
    deps = _deps(settings=settings, bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="how many rows do you have?",
        ),
        deps=deps,
    )
    assert outcome.tool_call_count == 1
    tool_messages = [
        m
        for m in openai.chat_calls[1]["messages"]
        if m.get("role") == "tool"
    ]
    payload = json.loads(tool_messages[0]["content"])
    assert payload["rows_total"] == 42
    assert payload["packages"] == 3
    assert outcome.final_message.startswith("The corpus has")
