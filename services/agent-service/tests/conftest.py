"""Shared fixtures for agent-service tests.

The FastAPI TestClient runs against an in-process app; the sibling
`semantic_enrich` package is imported directly. All tests use fake BQ +
OpenAI clients so no cloud credentials are needed.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from semantic_enrich.clients.openai import ChatCompletionResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import LoopDeps

from agent_service.app import create_app
from agent_service.config import AgentServiceSettings
from agent_service.deps import AppState

FIXED_TOKEN = "test-token-abc"


class FakeBqClient:
    """Minimal BqClient stand-in.

    Tests populate `queries` with `(match_substring, rows_or_exception)`
    pairs; the fake matches queries against them in insertion order and
    returns the first hit. Missing hits raise so a regression that adds
    an unexpected query fails loudly rather than silently returning [].
    """

    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []
        self.executed: list[str] = []
        self.dry_run_bytes_calls: list[str] = []
        self.dry_run_return: int = 1_000
        self.bounded_return: Any = None

    def query_rows(self, sql: str, *, params: Iterable[Any] = ()) -> Iterator[dict[str, Any]]:
        self.executed.append(sql)
        for needle, result in self.queries:
            if needle in sql:
                if isinstance(result, BaseException):
                    raise result
                return iter(result)
        raise AssertionError(f"unexpected query: {sql}")

    def execute(self, sql: str) -> None:  # pragma: no cover - unused in tests
        self.executed.append(sql)

    def execute_with_params(self, sql: str, *, params: Iterable[Any] = ()) -> None:  # pragma: no cover
        self.executed.append(sql)

    def append_jsonl_file(self, **kwargs: Any) -> int:  # pragma: no cover
        return 0

    def create_staging_table(self, **kwargs: Any) -> None:  # pragma: no cover
        return None

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None:  # pragma: no cover
        return None

    def dry_run_bytes(self, sql: str, *, params: Iterable[Any] = (), timeout_ms: int) -> int:
        self.dry_run_bytes_calls.append(sql)
        return self.dry_run_return

    def run_bounded_query(
        self,
        sql: str,
        *,
        params: Iterable[Any] = (),
        timeout_ms: int,
        max_bytes_billed: int,
        row_limit: int,
    ) -> Any:
        if self.bounded_return is None:
            from semantic_enrich.clients.bq import BoundedQueryResult

            return BoundedQueryResult(
                rows=[{"n": 1}],
                total_bytes_billed=500,
                slot_ms=10,
                elapsed_ms=20,
                timed_out=False,
                error=None,
            )
        return self.bounded_return


class FakeOpenAIClient:
    """Scripted OpenAI client. Tests set `chat_responses` to a list of
    `ChatCompletionResult`s; `chat_with_tools` returns them in order.

    `embed` returns a fixed unit vector at the configured dim so shape
    checks in downstream code (`embed_question`) don't reject the stub.
    """

    def __init__(self, *, dim: int = 1536) -> None:
        self.dim = dim
        self.chat_responses: list[ChatCompletionResult] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[0.0] * self.dim for _ in texts]

    def generate_structured(self, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("not used")

    def chat_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        parallel_tool_calls: bool = True,
    ) -> ChatCompletionResult:
        self.chat_calls.append(
            {"messages": list(messages), "tools": list(tools)}
        )
        if not self.chat_responses:
            return ChatCompletionResult(
                content="fallback", tool_calls=[], tokens_in=1,
                tokens_out=1, finish_reason="stop",
            )
        return self.chat_responses.pop(0)


@pytest.fixture
def service_settings() -> AgentServiceSettings:
    return AgentServiceSettings(
        api_token=SecretStr(FIXED_TOKEN),
        cors_origins="http://localhost:3000,https://maplequery.vercel.app",
        openai_api_key=SecretStr("test-key"),
        gcp_project_id="test-project",
    )


@pytest.fixture
def loop_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        gcp_project_id="test-project",
        openai_api_key=SecretStr("test-key"),
    )


@pytest.fixture
def fake_bq() -> FakeBqClient:
    return FakeBqClient()


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def app_state(
    service_settings: AgentServiceSettings,
    loop_settings: Settings,
    fake_bq: FakeBqClient,
    fake_openai: FakeOpenAIClient,
) -> AppState:
    cache = ResponseCache(
        max_entries=8, max_value_bytes=1_000_000, ttl_seconds=60
    )
    loop_deps = LoopDeps(
        bq=fake_bq,
        openai_client=fake_openai,
        settings=loop_settings,
        system_prompt="test system prompt",
        prompt_hash="hash-abc",
        cache=cache,
        snapshot_hash_provider=lambda: "snapshot-hash",
    )
    return AppState(
        service_settings=service_settings,
        loop_settings=loop_settings,
        loop_deps=loop_deps,
        bq=fake_bq,
        openai_client=fake_openai,
    )


@pytest.fixture
def client(
    service_settings: AgentServiceSettings,
    app_state: AppState,
) -> Iterator[TestClient]:
    app = create_app(service_settings=service_settings, app_state=app_state)
    with TestClient(app) as c:
        yield c
