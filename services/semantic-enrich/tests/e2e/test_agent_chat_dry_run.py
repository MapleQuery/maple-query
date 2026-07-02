"""E2E dry-run for the agent loop.

Drives the loop directly (not through the CliRunner — which pollutes
structlog's cached stdout with a captured file handle and breaks
sibling e2e tests). The CLI subcommand is a thin wrapper around
`run_turn` + a canned client, so exercising `run_turn` end-to-end
covers the same ground the CLI test would.
"""
from __future__ import annotations

import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_events import Done, MessageDelta, TurnStart
from semantic_enrich.core.agent_loop import (
    ChatRequest,
    LoopDeps,
    load_system_prompt,
    run_turn,
)
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

_SERVICE_ROOT = Path(__file__).resolve().parents[2]


def test_scripted_conversation_streams_expected_events() -> None:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        agent_system_prompt_path=(
            _SERVICE_ROOT / "agent" / "prompts" / "system.j2"
        ),
    )
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [])
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[{"content": "[canned] no data available."}],
    )
    prompt, prompt_hash = load_system_prompt(
        settings.agent_system_prompt_path, settings
    )
    deps = LoopDeps(
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
    events = list(
        run_turn(
            request=ChatRequest(
                conversation_id="test-conv",
                history=[],
                question="hello agent",
            ),
            deps=deps,
        )
    )
    assert isinstance(events[0], TurnStart)
    assert isinstance(events[-1], Done)
    assert any(
        isinstance(e, MessageDelta) and "[canned]" in e.delta for e in events
    )
