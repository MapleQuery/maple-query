"""Unit-level assertions on the OpenAI client + batching helper.

The real `RealOpenAIClient` is not exercised end-to-end here (that
would need network); we test the batching + validation shape via the
`OpenAIClient` Protocol using `FakeOpenAIClient`, and pin the client
Protocol surface.
"""
from __future__ import annotations

import math

import pytest
import structlog

from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.core.openai_embed import embed_texts_in_batches
from tests.integration.openai_fakes import FakeOpenAIClient


def _unit_vec(dim: int) -> list[float]:
    return [1.0 / math.sqrt(dim)] * dim


def test_fake_client_is_openai_client_protocol() -> None:
    """FakeOpenAIClient must satisfy the runtime-checkable Protocol."""
    client = FakeOpenAIClient()
    assert isinstance(client, OpenAIClient)


def test_embed_batches_in_chunks() -> None:
    """Batch size 2 → 3 texts becomes 2 batches (2 + 1)."""
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(1536))
    results = embed_texts_in_batches(
        client=client,
        texts=["a", "b", "c"],
        batch_size=2,
        expected_dim=1536,
        log=structlog.get_logger("test"),
        log_event_prefix="test",
    )
    assert len(results) == 3
    assert client.calls == [["a", "b"], ["c"]]
    assert all(r.vector is not None and len(r.vector) == 1536 for r in results)


def test_embed_flags_wrong_dim() -> None:
    """Vectors of the wrong dim are marked failed, not written."""
    client = FakeOpenAIClient(vector_factory=lambda _: [0.1] * 8)
    results = embed_texts_in_batches(
        client=client,
        texts=["x"],
        batch_size=32,
        expected_dim=1536,
        log=structlog.get_logger("test"),
        log_event_prefix="test",
    )
    assert results[0].vector is None
    assert results[0].failure_reason == "wrong_dim"


def test_embed_flags_nan() -> None:
    client = FakeOpenAIClient(
        vector_factory=lambda _: [float("nan")] * 1536,
    )
    results = embed_texts_in_batches(
        client=client,
        texts=["x"],
        batch_size=32,
        expected_dim=1536,
        log=structlog.get_logger("test"),
        log_event_prefix="test",
    )
    assert results[0].failure_reason == "has_nan"


def test_embed_raises_on_length_mismatch() -> None:
    """A vendor bug that returns fewer vectors than inputs fails loud."""

    class ShortClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [_unit_vec(1536)]  # always exactly 1

    with pytest.raises(RuntimeError, match="length mismatch"):
        embed_texts_in_batches(
            client=ShortClient(),
            texts=["a", "b"],
            batch_size=8,
            expected_dim=1536,
            log=structlog.get_logger("test"),
            log_event_prefix="test",
        )
