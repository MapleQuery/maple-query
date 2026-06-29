"""Shared fixtures for the validate_round_trip unit tests.

The tests deliberately avoid talking HTTP: they hand-roll fakes
that match the surface of `openai.OpenAI` and `httpx.Client` for
the two endpoints the gate touches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from validate_round_trip import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    GateConfig,
)


@dataclass
class _FakeResponse:
    status: int
    body: dict[str, Any]

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> dict[str, Any]:
        return self.body


class FakeHttpClient:
    """Stands in for `httpx.Client` — only `.get(url, timeout=...)`."""

    def __init__(self, model_id: str = GENERATION_MODEL_NAME,
                 raise_on_get: Exception | None = None) -> None:
        self.model_id = model_id
        self.raise_on_get = raise_on_get
        self.calls: list[str] = []

    def get(self, url: str, timeout: float = 10.0) -> _FakeResponse:
        self.calls.append(url)
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return _FakeResponse(
            status=200,
            body={"data": [{"id": self.model_id}]},
        )


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice]


@dataclass
class _FakeEmbeddingDatum:
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingDatum]


class _FakeChat:
    def __init__(self, content: str | Exception, *, sleep_ms: int = 0) -> None:
        self._content = content
        self._sleep_ms = sleep_ms

    @property
    def completions(self) -> _FakeChat:
        return self

    def create(self, **_: Any) -> _FakeChatCompletion:
        if isinstance(self._content, Exception):
            raise self._content
        if self._sleep_ms > 0:
            import time
            time.sleep(self._sleep_ms / 1000.0)
        return _FakeChatCompletion(
            choices=[_FakeChoice(message=_FakeMessage(content=self._content))]
        )


class _FakeEmbeddings:
    def __init__(self, vector: list[float] | Exception) -> None:
        self._vector = vector

    def create(self, **_: Any) -> _FakeEmbeddingResponse:
        if isinstance(self._vector, Exception):
            raise self._vector
        return _FakeEmbeddingResponse(
            data=[_FakeEmbeddingDatum(embedding=list(self._vector))]
        )


class FakeOpenAI:
    """Mimics the openai.OpenAI surface used by validate_round_trip."""

    def __init__(
        self,
        *,
        chat_content: str | Exception = "",
        embedding_vector: list[float] | Exception | None = None,
        chat_sleep_ms: int = 0,
    ) -> None:
        self.chat = _FakeChat(chat_content, sleep_ms=chat_sleep_ms)
        if embedding_vector is None:
            embedding_vector = list(np.zeros(EMBEDDING_DIM))
        self.embeddings = _FakeEmbeddings(embedding_vector)


def unit_vector(dim: int = EMBEDDING_DIM, seed: int = 1) -> list[float]:
    """Deterministic L2-normalised vector of the given dimension."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


def valid_chat_payload() -> str:
    return json.dumps({
        "package_id": "test-pkg-123",
        "summary": "A quarterly CPI dataset.",
    })


@pytest.fixture
def gate_config() -> GateConfig:
    return GateConfig(
        generation_base_url="http://127.0.0.1:8001",
        embedding_base_url="http://127.0.0.1:8002",
        generation_model=GENERATION_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
    )
