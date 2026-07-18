"""The surrender contract: a no-data conclusion must say what was
tried and name the closest candidate.

The live wording is the model's job (the prompt-side contract is
pinned in test_prompt_v2); these scripted turns pin the pipeline-side
inputs that make the contract checkable downstream — the turn record
carries every phrasing tried with its verdict, and a surrender is
recorded as an answer, not a clarify.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
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
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


SURRENDER = (
    "I could not find a confident dataset match. I searched "
    '"visa processing times" and "temporary resident visa wait", and '
    "the closest candidate was [Visa Office Statistics](/datasets/p1), "
    "which covers application volumes but not processing times."
)


def test_surrender_turn_record_names_every_phrasing_tried() -> None:
    bq = FakeBqClient()
    for _ in range(2):
        bq.register_query(
            "VECTOR_SEARCH",
            [
                {
                    "package_id": "p1",
                    "title": "Visa Office Statistics",
                    "summary": "s",
                    "grain": None,
                    "measures": [],
                    "dimensions": [],
                    "distance": 0.85,
                }
            ],
        )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "visa processing times"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "search_datasets",
                        "arguments": {
                            "query": "temporary resident visa wait"
                        },
                    }
                ]
            },
            {"content": SURRENDER},
        ],
    )
    deps = _deps(settings=_settings(), bq=bq, openai=openai)
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="How long do visas take?",
        ),
        deps=deps,
    )

    records = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    record = records[0].record
    # A statement-shaped surrender records as a no-data claim — never
    # a clarify (that tag needs a question), never an answer (no SQL
    # evidence behind it).
    assert record["outcome"] == "no_data"
    # The record carries the fit-check inputs: every phrasing tried,
    # each with its retrieval verdict. A surrender with zero searches
    # is downstream's retry signal.
    tried = [s["query"] for s in record["searches_tried"]]
    assert tried == [
        "visa processing times",
        "temporary resident visa wait",
    ]
    assert all(
        s["retrieval_quality"] == "weak" for s in record["searches_tried"]
    )
    # The scripted message honours the contract the prompt demands:
    # phrasings tried + closest candidate by title.
    for phrase in tried:
        assert phrase in outcome.final_message
    assert "Visa Office Statistics" in outcome.final_message
