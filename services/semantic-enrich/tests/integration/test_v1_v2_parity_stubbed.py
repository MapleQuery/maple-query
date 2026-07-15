"""v1 ↔ v2-with-stubs event parity on scripted fixtures.

With stub phases, the v2 pipeline must emit the same event sequence
as the v1 loop — the extraction changed no behaviour. "Same" modulo:

- the additive v2-only event types (phase_start, turn_record …),
  which v1 consumers ignore by contract and the comparison strips;
- per-turn identifiers and clocks (turn_id, elapsed_ms), normalized.

Both loops run each scenario on identical scripts and fresh fakes.
The prompt swap case pins that switching v2 to prompt v2 changes the
system message and nothing else observable in the event stream.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_loop
from semantic_enrich.core.agent import phases, pipeline
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

V2_ONLY_TYPES = {
    "phase_start",
    "triage_result",
    "reformulation",
    "verification",
    "plan_hint",
    "turn_record",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _v1_deps(
    *,
    settings: Settings,
    bq: FakeBqClient,
    openai: FakeOpenAIClient,
    prompt: str,
) -> agent_loop.LoopDeps:
    return agent_loop.LoopDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt=prompt,
        prompt_hash="hash-shared",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )


def _v2_deps(
    *,
    settings: Settings,
    bq: FakeBqClient,
    openai: FakeOpenAIClient,
    prompt: str,
) -> phases.PipelineDeps:
    return phases.PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt=prompt,
        prompt_hash="hash-shared",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )


def _normalized(
    events: list[agent_events.AgentEvent], *, strip_v2: bool
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        if strip_v2 and event.event_type in V2_ONLY_TYPES:
            continue
        payload = event.to_dict()
        payload.pop("turn_id", None)
        payload.pop("elapsed_ms", None)
        out.append(payload)
    return out


def _happy_path_bq() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "VECTOR_SEARCH",
        [
            {
                "package_id": "pkg-1",
                "title": "Housing Starts",
                "summary": "housing",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "distance": 0.1,
            }
        ],
    )
    return bq


def _happy_path_script() -> list[dict[str, Any]]:
    return [
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_datasets",
                    "arguments": {"query": "housing spend", "k": 3},
                }
            ],
        },
        {"content": "Housing spend was $X ([Housing Starts](/datasets/pkg-1))."},
    ]


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _request(question: str = "how much housing spend?") -> ChatRequest:
    return ChatRequest(conversation_id="c1", history=[], question=question)


def _run_v1(
    *, settings: Settings, bq: FakeBqClient, script: list[dict[str, Any]],
    prompt: str, request: ChatRequest, runs: int = 1,
) -> list[list[agent_events.AgentEvent]]:
    openai = FakeOpenAIClient(vector_factory=_unit_vec, chat_script=script)
    deps = _v1_deps(settings=settings, bq=bq, openai=openai, prompt=prompt)
    return [
        list(agent_loop.run_turn(request=request, deps=deps))
        for _ in range(runs)
    ]


def _run_v2(
    *, settings: Settings, bq: FakeBqClient, script: list[dict[str, Any]],
    prompt: str, request: ChatRequest, runs: int = 1,
) -> tuple[list[list[agent_events.AgentEvent]], FakeOpenAIClient]:
    openai = FakeOpenAIClient(vector_factory=_unit_vec, chat_script=script)
    deps = _v2_deps(settings=settings, bq=bq, openai=openai, prompt=prompt)
    return [
        list(pipeline.run_turn(request=request, deps=deps))
        for _ in range(runs)
    ], openai


def test_happy_path_parity_with_shared_prompt() -> None:
    settings = _settings()
    prompt = "shared test prompt"
    v1 = _run_v1(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt=prompt, request=_request(),
    )[0]
    v2, _ = _run_v2(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt=prompt, request=_request(),
    )
    assert _normalized(v2[0], strip_v2=True) == _normalized(
        v1, strip_v2=False
    )


def test_budget_forced_parity() -> None:
    settings = _settings(agent_max_tool_calls=1)

    def script() -> list[dict[str, Any]]:
        return [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "x"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "search_datasets",
                        "arguments": {"query": "y"},
                    }
                ]
            },
            {"content": "best effort."},
        ]

    def bq() -> FakeBqClient:
        client = FakeBqClient()
        for _ in range(2):
            client.register_query("VECTOR_SEARCH", [])
        return client

    v1 = _run_v1(
        settings=settings, bq=bq(), script=script(),
        prompt="p", request=_request("pick"),
    )[0]
    v2, _ = _run_v2(
        settings=settings, bq=bq(), script=script(),
        prompt="p", request=_request("pick"),
    )
    assert _normalized(v2[0], strip_v2=True) == _normalized(
        v1, strip_v2=False
    )


def test_invalid_history_parity() -> None:
    settings = _settings()
    request = ChatRequest(
        conversation_id="c1",
        history=[{"role": "wizard", "content": "bad"}],
        question="anything",
    )
    v1 = _run_v1(
        settings=settings, bq=FakeBqClient(), script=[],
        prompt="p", request=request,
    )[0]
    v2, _ = _run_v2(
        settings=settings, bq=FakeBqClient(), script=[],
        prompt="p", request=request,
    )
    assert _normalized(v2[0], strip_v2=True) == _normalized(
        v1, strip_v2=False
    )


def test_cache_replay_parity_including_the_known_quirks() -> None:
    """Second identical question replays from cache on both loops —
    including v1's recorded-turn_start-replays-after-the-fresh-one
    quirk, which NoopMemory preserves bug-for-bug."""
    settings = _settings()
    _v1_first, v1_second = _run_v1(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt="p",
        request=_request(), runs=2,
    )
    (_v2_first, v2_second), _ = _run_v2(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt="p",
        request=_request(), runs=2,
    )
    assert _normalized(v1_second, strip_v2=False)[0]["cached"] is True
    assert _normalized(v2_second, strip_v2=True) == _normalized(
        v1_second, strip_v2=False
    )


def test_prompt_swap_changes_only_the_system_message() -> None:
    """v2 with prompt A vs prompt B: identical event streams, different
    system message — the only prompt-dependent divergence."""
    settings = _settings()
    events_a, openai_a = _run_v2(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt="prompt A",
        request=_request(),
    )
    events_b, openai_b = _run_v2(
        settings=settings, bq=_happy_path_bq(),
        script=_happy_path_script(), prompt="prompt B",
        request=_request(),
    )
    assert _normalized(events_a[0], strip_v2=True) == _normalized(
        events_b[0], strip_v2=True
    )
    sys_a = openai_a.chat_calls[0]["messages"][0]
    sys_b = openai_b.chat_calls[0]["messages"][0]
    assert sys_a["role"] == "system" and sys_b["role"] == "system"
    assert sys_a["content"] == "prompt A"
    assert sys_b["content"] == "prompt B"
    rest_a = openai_a.chat_calls[0]["messages"][1:]
    rest_b = openai_b.chat_calls[0]["messages"][1:]
    assert rest_a == rest_b
