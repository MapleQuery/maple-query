"""OpenAI client surface: `OpenAIClient` Protocol + `RealOpenAIClient` impl.

Ships the embedding surface only. Query-time generation
(`generate_structured`) lands here in 4.6 — one client, one vendor,
one key, one timeout.

Vectors returned by `embed()` carry the model's native dimensionality;
callers assert against `Settings.openai_embedding_dim` so a
model/config mismatch fails loudly at the batch boundary rather than
silently corrupting the warehouse.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from openai import OpenAI
from tenacity import Retrying

from semantic_enrich.providers.openai_retry import openai_retry_policy


@runtime_checkable
class OpenAIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input.

        All vectors are the model's native dimensionality; the caller
        asserts against `Settings.openai_embedding_dim`.
        """


class RealOpenAIClient:
    """Concrete `OpenAIClient` backed by `openai.OpenAI`.

    Constructed once at process start (`for_settings`) and passed
    through the core runners as a parameter so tests can substitute
    `FakeOpenAIClient` at the Protocol boundary.
    """

    def __init__(
        self,
        *,
        api_key: str,
        embedding_model: str,
        request_timeout_s: float,
        max_retries: int,
        retry: Retrying | None = None,
    ) -> None:
        # `openai.OpenAI` itself supports `max_retries`, but we
        # centralise retry policy in tenacity so the shape matches
        # `bq_retry_policy`. Client-level max_retries=0 keeps the SDK
        # from double-retrying underneath us.
        self._client = OpenAI(
            api_key=api_key,
            timeout=request_timeout_s,
            max_retries=0,
        )
        self._embedding_model = embedding_model
        self._retry = retry if retry is not None else openai_retry_policy(
            max_attempts=max_retries,
        )

    @classmethod
    def for_settings(
        cls,
        *,
        api_key: str,
        embedding_model: str,
        request_timeout_s: float,
        max_retries: int,
    ) -> RealOpenAIClient:
        return cls(
            api_key=api_key,
            embedding_model=embedding_model,
            request_timeout_s=request_timeout_s,
            max_retries=max_retries,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for attempt in self._retry:
            with attempt:
                response = self._client.embeddings.create(
                    model=self._embedding_model,
                    input=texts,
                )
                vectors = [list(datum.embedding) for datum in response.data]
        return vectors
