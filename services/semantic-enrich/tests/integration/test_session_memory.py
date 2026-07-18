"""Session memory through the v2 pipeline.

The two evidence scenarios: identical questions replay from the digest
cache (one live run, then sub-second replays with zero model/warehouse
calls), and a related follow-up carrying the prior turn's record gets
a plan hint. Plus the eligibility guard (surrenders never cached) and
the no-summarization contract of compaction v2.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.memory import ReplayCacheV2, SessionMemory
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

QUESTION = "How much airplane spending by federal officials?"

_SQL = (
    "SELECT JSON_VALUE(r.row, '$.Airfare') AS airfare "
    "FROM `proj.raw.rows` AS r "
    "WHERE r.document_id IN ('doc-1') LIMIT 10"
)


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _bq_full_turn() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-travel",
                "title": "Travel expenses",
                "summary": "s",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.1,
            }
        ],
    )
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-travel",
                "title": "Travel 2024",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)", [{"document_id": "doc-1", "columns": ["Airfare"]}]
    )
    bq.bounded_default = BoundedQueryResult(
        rows=[{"airfare": "100"}],
        total_bytes_billed=10,
        slot_ms=1,
        elapsed_ms=1,
        timed_out=False,
        error=None,
    )
    return bq


def _full_turn_script() -> list[dict[str, Any]]:
    return [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "search_datasets",
                    "arguments": {"query": "airplane spending"},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "c2",
                    "name": "list_documents",
                    "arguments": {"package_ids": ["pkg-travel"]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "c3",
                    "name": "run_sql",
                    "arguments": {"sql": _SQL, "rationale": "sum airfare"},
                }
            ]
        },
        {
            "content": (
                "Airfare was $100 "
                "([Travel expenses](/datasets/pkg-travel))."
            )
        },
    ]


def _deps(
    *,
    settings: Settings,
    bq: FakeBqClient,
    openai: FakeOpenAIClient,
    cache: ReplayCacheV2,
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
        memory=SessionMemory(cache=cache),
    )


def _request(
    question: str = QUESTION,
    *,
    history: list[dict[str, Any]] | None = None,
    turn_records: list[dict[str, Any]] | None = None,
) -> ChatRequest:
    return ChatRequest(
        conversation_id="c1",
        history=history or [],
        question=question,
        turn_records=turn_records or [],
    )


def test_identical_questions_replay_from_digest_cache() -> None:
    settings = _settings()
    cache = ReplayCacheV2(max_entries=10, ttl_seconds=600)
    bq = _bq_full_turn()
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=_full_turn_script(),
    )
    deps = _deps(settings=settings, bq=bq, openai=openai, cache=cache)

    live = run_turn_collected(request=_request(), deps=deps)
    assert live.cache_hit is False
    assert live.final_message.startswith("Airfare was $100")
    live_model_calls = len(openai.chat_calls)
    live_bq_calls = len(bq.calls)

    # The 05 repro: three identical follow-ups, all replays.
    for _ in range(3):
        replayed = run_turn_collected(request=_request(), deps=deps)
        assert replayed.cache_hit is True
        assert replayed.final_message == live.final_message
        # Fresh turn ids, fresh record.
        assert replayed.turn_id != live.turn_id
        records = [
            e
            for e in replayed.events
            if isinstance(e, agent_events.TurnRecordEvent)
        ]
        assert records[0].record["turn_id"] == replayed.turn_id
    # Zero additional model or warehouse traffic across all replays.
    assert len(openai.chat_calls) == live_model_calls
    assert len(bq.calls) == live_bq_calls


def test_surrendered_turns_are_never_cached() -> None:
    settings = _settings()
    cache = ReplayCacheV2(max_entries=10, ttl_seconds=600)
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query("VECTOR_SEARCH", [])
    script = [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "search_datasets",
                    "arguments": {"query": "unfindable"},
                }
            ]
        },
        {"content": "I could not find data; I tried 'unfindable'."},
    ]
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=script + script,
    )
    deps = _deps(settings=settings, bq=bq, openai=openai, cache=cache)

    first = run_turn_collected(request=_request("q?"), deps=deps)
    assert first.cache_hit is False
    assert len(cache) == 0
    second = run_turn_collected(request=_request("q?"), deps=deps)
    assert second.cache_hit is False  # re-ran live, no frozen failure


def test_followup_with_record_gets_plan_hint_and_no_summarization() -> None:
    settings = _settings()
    cache = ReplayCacheV2(max_entries=10, ttl_seconds=600)
    bq = _bq_full_turn()
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=_full_turn_script(),
    )
    deps = _deps(settings=settings, bq=bq, openai=openai, cache=cache)
    first = run_turn_collected(request=_request(), deps=deps)
    record = next(
        e.record
        for e in first.events
        if isinstance(e, agent_events.TurnRecordEvent)
    )
    assert record["outcome"] == "answered"

    # Related follow-up, fresh deps (cold cache — the hint, not the
    # replay, must carry the plan). Long prior history exercises
    # compaction v2's keep-window.
    bq2 = _bq_full_turn()
    # The scripted model obeys the hint: straight to list_documents on
    # the plan's package, no dataset/column search. The recall step
    # must have admitted the package to the tool whitelist for this
    # to succeed.
    openai2 = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "list_documents",
                        "arguments": {"package_ids": ["pkg-travel"]},
                    }
                ]
            },
            {"content": "still $100."},
        ],
    )
    cache2 = ReplayCacheV2(max_entries=10, ttl_seconds=600)
    deps2 = _deps(settings=settings, bq=bq2, openai=openai2, cache=cache2)
    history: list[dict[str, Any]] = []
    for i in range(6):
        history.append({"role": "user", "content": f"old question {i}"})
        history.append({"role": "assistant", "content": f"old answer {i}"})
    outcome = run_turn_collected(
        request=_request(
            "airplane spending by officials in 2024?",
            history=history,
            turn_records=[record],
        ),
        deps=deps2,
    )

    hints = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.PlanHint)
    ]
    assert len(hints) == 1
    assert hints[0].records_used[0]["question"] == QUESTION
    # The hint-driven retrieval skip worked end to end: no search, no
    # tool error, documents listed from the plan's package.
    kinds = [e.event_type for e in outcome.events]
    assert "retrieval_started" not in kinds
    assert "tool_error" not in kinds
    assert "documents_listed" in kinds
    assert outcome.final_message == "still $100."
    assert outcome.tool_call_count == 1
    # The hint reached the model as a system message naming the plan.
    first_call_messages = openai2.chat_calls[0]["messages"]
    hint_msgs = [
        m
        for m in first_call_messages
        if m.get("role") == "system"
        and "Prior resolved plans" in str(m.get("content", ""))
    ]
    assert len(hint_msgs) == 1
    assert "pkg-travel" in hint_msgs[0]["content"]
    # Compaction v2: only the keep-window turns reach the model, and
    # no summarization call was ever made.
    user_msgs = [
        m for m in first_call_messages if m.get("role") == "user"
    ]
    assert [m["content"] for m in user_msgs] == [
        "old question 4",
        "old question 5",
        "airplane spending by officials in 2024?",
    ]
    assert openai2.structured_calls == []
