"""`agent_loop_impl` flag dispatch at the service boundary.

The flag selects which orchestrator `AppState.run_turn` drives, and
`/chat` accepts the `turn_records` field under both implementations.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic import SecretStr
from semantic_enrich.clients.openai import ChatCompletionResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import LoopDeps

from agent_service.app import create_app
from agent_service.config import AgentServiceSettings
from agent_service.deps import AppState
from tests.conftest import FIXED_TOKEN, FakeBqClient, FakeOpenAIClient


def _service_settings() -> AgentServiceSettings:
    return AgentServiceSettings(
        api_token=SecretStr(FIXED_TOKEN),
        cors_origins="http://localhost:3000",
        openai_api_key=SecretStr("test-key"),
        gcp_project_id="test-project",
    )


def _loop_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        gcp_project_id="test-project",
        openai_api_key=SecretStr("test-key"),
    )


def _scripted_openai(answer: str) -> FakeOpenAIClient:
    openai = FakeOpenAIClient()
    openai.chat_responses = [
        ChatCompletionResult(
            content=answer,
            tool_calls=[],
            tokens_in=20,
            tokens_out=5,
            finish_reason="stop",
        )
    ]
    return openai


def _build_app(loop_impl: str):  # type: ignore[no-untyped-def]
    service_settings = _service_settings()
    loop_settings = _loop_settings()
    bq = FakeBqClient()
    openai = _scripted_openai(f"served by {loop_impl}")
    cache = ResponseCache(
        max_entries=8, max_value_bytes=1_000_000, ttl_seconds=60
    )
    common = {
        "bq": bq,
        "openai_client": openai,
        "settings": loop_settings,
        "system_prompt": "test system prompt",
        "prompt_hash": f"hash-{loop_impl}",
        "cache": cache,
        "snapshot_hash_provider": lambda: "snapshot-hash",
    }
    loop_deps = (
        PipelineDeps(**common)  # type: ignore[arg-type]
        if loop_impl == "v2"
        else LoopDeps(**common)  # type: ignore[arg-type]
    )
    state = AppState(
        service_settings=service_settings,
        loop_settings=loop_settings,
        loop_deps=loop_deps,
        bq=bq,
        openai_client=openai,
        loop_impl=loop_impl,  # type: ignore[arg-type]
    )
    return create_app(
        service_settings=service_settings, app_state=state
    )


def _stream_body(loop_impl: str, body: dict[str, object]) -> str:
    with TestClient(_build_app(loop_impl)) as client, client.stream(
        "POST",
        "/chat",
        json=body,
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    ) as r:
        assert r.status_code == 200
        return r.read().decode("utf-8")


def _stream_types(loop_impl: str, body: dict[str, object]) -> list[str]:
    types: list[str] = []
    for block in _stream_body(loop_impl, body).split("\n\n"):
        for line in block.strip().splitlines():
            if line.startswith("event:"):
                types.append(line[len("event:") :].strip())
    return types


def test_v2_flag_serves_chat_with_phase_events() -> None:
    types = _stream_types(
        "v2",
        {"conversation_id": "conv-1", "question": "meaning of life"},
    )
    assert "phase_start" in types
    assert "turn_record" in types
    assert types[-1] == "done"


def test_v1_default_emits_no_pipeline_events() -> None:
    types = _stream_types(
        "v1",
        {"conversation_id": "conv-1", "question": "meaning of life"},
    )
    assert "phase_start" not in types
    assert "turn_record" not in types
    assert types[0] == "turn_start"
    assert types[-1] == "done"


def test_turn_records_accepted_by_both_impls() -> None:
    body = {
        "conversation_id": "conv-1",
        "question": "meaning of life",
        "turn_records": [
            {"turn_id": "t-prior", "packages": ["pkg-1"]}
        ],
    }
    for impl in ("v1", "v2"):
        types = _stream_types(impl, body)
        assert types[-1] == "done", f"turn_records broke impl {impl}"


def test_turn_record_frame_is_valid_json_with_v2_payload() -> None:
    text = _stream_body(
        "v2", {"conversation_id": "conv-1", "question": "q?"}
    )
    record_payload = None
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        if any(line.strip() == "event: turn_record" for line in lines):
            data_line = next(
                line for line in lines if line.startswith("data:")
            )
            record_payload = json.loads(data_line[len("data:") :])
    assert record_payload is not None
    assert record_payload["record"]["loop_impl"] == "v2"
