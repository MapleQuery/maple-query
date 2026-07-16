"""Triage wired through the v2 orchestrator end-to-end.

Deflected turns must be cheap by construction: the FakeBqClient
records every call, so the off_scope case asserts literally zero
warehouse traffic and zero research-model calls, while the meta case
allows exactly the describe_corpus metadata queries."""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent.triage import QueryTriage
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


@pytest.fixture(autouse=True)
def _fresh_corpus_cache() -> None:
    agent_tools.reset_corpus_stats_cache()


def _settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {
        "gcp_project_id": "proj",
        "openai_api_key": "sk-test",
        "agent_triage_mode": "act",
        "agent_cache_replay_delay_ms": 0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _classifier_output(category: str, **fields: Any) -> dict[str, Any]:
    return {
        "category": category,
        "confidence": fields.get("confidence", 0.95),
        "reason": "test",
        "off_scope_reason": fields.get("off_scope_reason"),
        "deflection_hint": fields.get("deflection_hint"),
        "clarify_question": fields.get("clarify_question"),
    }


def _deps(
    *, settings: Settings, openai: FakeOpenAIClient, bq: FakeBqClient
) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,  # type: ignore[arg-type]
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-v2-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
        triage=QueryTriage.from_settings(settings),
    )


def _request(question: str) -> ChatRequest:
    return ChatRequest(conversation_id="c1", history=[], question=question)


def test_off_scope_turn_deflects_with_zero_warehouse_traffic() -> None:
    bq = FakeBqClient()
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output("off_scope", off_scope_reason="provincial")
        ],
        chat_script=[{"content": "research should never run"}],
    )
    outcome = run_turn_collected(
        request=_request("what has doug ford bought"),
        deps=_deps(settings=_settings(), openai=openai, bq=bq),
    )

    types = [e.event_type for e in outcome.events]
    # The full deflection stream, in order.
    assert [
        t
        for t in types
        if t
        in (
            "triage_result",
            "turn_start",
            "message_delta",
            "turn_record",
            "done",
        )
    ] == ["triage_result", "turn_start", "message_delta", "turn_record", "done"]
    assert outcome.final_message.startswith("MapleQuery answers")

    # Zero warehouse, retrieval, or research-model work.
    assert bq.calls == []
    assert bq.bounded_calls == []
    assert bq.dry_run_calls == []
    assert bq.table_num_rows_calls == []
    assert openai.calls == []  # no embeddings
    assert openai.chat_calls == []  # no research loop

    record = next(
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ).record
    assert record["triage_category"] == "off_scope"
    assert record["terminal_reason"] == "triage_short_circuit"


def test_meta_turn_touches_only_the_corpus_stats_surfaces() -> None:
    bq = FakeBqClient()
    bq.table_num_rows_by_ref["proj.raw.rows"] = 205_000_000
    bq.register_query(
        "AS packages",
        [
            {
                "packages": 210,
                "documents_loaded": 950,
                "latest_load_at": "2026-07-01T00:00:00",
            }
        ],
    )
    openai = FakeOpenAIClient(
        structured_responses=[_classifier_output("meta")],
        chat_script=[{"content": "research should never run"}],
    )
    outcome = run_turn_collected(
        request=_request("how many rows of data do you have access to?"),
        deps=_deps(settings=_settings(), openai=openai, bq=bq),
    )

    assert "205,000,000" in outcome.final_message
    # Only describe_corpus's metadata surfaces — nothing else.
    assert bq.table_num_rows_calls == ["proj.raw.rows"]
    assert all("AS packages" in c["sql"] for c in bq.calls)
    assert bq.bounded_calls == []
    assert openai.calls == []
    assert openai.chat_calls == []
    assert outcome.events[-1].event_type == "done"


def test_log_mode_runs_research_despite_an_off_scope_verdict() -> None:
    bq = FakeBqClient()
    openai = FakeOpenAIClient(
        structured_responses=[
            _classifier_output("off_scope", off_scope_reason="news")
        ],
        chat_script=[{"content": "a researched answer."}],
    )
    outcome = run_turn_collected(
        request=_request("what's the latest on tariffs"),
        deps=_deps(
            settings=_settings(agent_triage_mode="log"),
            openai=openai,
            bq=bq,
        ),
    )

    assert outcome.final_message == "a researched answer."
    assert len(openai.chat_calls) == 1
    triage_results = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TriageResult)
    ]
    assert len(triage_results) == 1
    assert triage_results[0].category == "off_scope"
    assert triage_results[0].enforced is False
