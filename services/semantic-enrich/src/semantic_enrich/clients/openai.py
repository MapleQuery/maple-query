"""OpenAI client surface: `OpenAIClient` Protocol + `RealOpenAIClient` impl.

Two production surfaces:

- `embed(texts)` — text-embedding-3-small vectors. Owned by 4.7; used by
  the embed / reembed pipelines and by the 4.6 harness at question time.
- `generate_structured(prompt, schema, ...)` — Structured Outputs
  (strict JSON Schema) via chat.completions. Added by 4.6 for
  single-shot SQL generation.

One vendor, one key, one timeout, one retry policy for both surfaces.
Vectors returned by `embed()` carry the model's native dimensionality;
callers assert against `Settings.openai_embedding_dim` so a
model/config mismatch fails loudly at the batch boundary rather than
silently corrupting the warehouse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI
from tenacity import Retrying

from semantic_enrich.providers.openai_retry import openai_retry_policy


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Return shape of `generate_structured`.

    `parsed` is the schema-conforming dict; `tokens_in` / `tokens_out`
    come straight off the OpenAI usage block and feed the eval report's
    aggregate cost accounting.
    """

    parsed: dict[str, Any]
    tokens_in: int
    tokens_out: int


@runtime_checkable
class OpenAIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input.

        All vectors are the model's native dimensionality; the caller
        asserts against `Settings.openai_embedding_dim`.
        """

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> StructuredGenerationResult:
        """Single-shot Structured Outputs call.

        `strict: true` at the vendor makes the response schema-conforming
        by construction. A schema violation despite that is a vendor
        regression, not a caller bug; the caller surfaces it as an
        `structured_output_violation` event and grades the question
        `sql_not_generated`.
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

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> StructuredGenerationResult:
        parsed: dict[str, Any] = {}
        tokens_in = 0
        tokens_out = 0
        for attempt in self._retry:
            with attempt:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                content = response.choices[0].message.content or ""
                # `strict: true` means the response is schema-conforming;
                # a parse error here is a vendor regression, surfaced up
                # so the runner can log `structured_output_violation`.
                parsed = json.loads(content)
                usage = response.usage
                tokens_in = int(usage.prompt_tokens) if usage else 0
                tokens_out = int(usage.completion_tokens) if usage else 0
        return StructuredGenerationResult(
            parsed=parsed, tokens_in=tokens_in, tokens_out=tokens_out
        )
