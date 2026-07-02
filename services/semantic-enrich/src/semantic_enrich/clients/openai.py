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


@dataclass(frozen=True)
class ChatToolCall:
    """One tool call in an assistant response.

    `arguments` is the parsed JSON object; the OpenAI SDK returns it
    as a string but the client parses it once at the boundary so
    downstream code doesn't repeat the work.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatCompletionResult:
    """Return shape of `chat_with_tools`.

    Either `tool_calls` is non-empty (model wants another loop
    iteration) or `content` is non-empty (final assistant message).
    Exactly one is populated by construction of an OpenAI chat
    completion, but the caller checks both anyway.
    """

    content: str
    tool_calls: list[ChatToolCall]
    tokens_in: int
    tokens_out: int
    finish_reason: str


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
        """One chat.completions call with tool calling enabled.

        Non-streaming for now — the loop wraps the eventual assistant
        text in a synthetic `message_delta` event so the FE UX is the
        same either way. Streaming is a follow-up when the CLI /
        HTTP surfaces prove they want per-token deltas.
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
        content = ""
        tool_calls: list[ChatToolCall] = []
        tokens_in = 0
        tokens_out = 0
        finish_reason = ""
        for attempt in self._retry:
            with attempt:
                # OpenAI SDK types messages/tools with strict TypedDicts;
                # the loop constructs them dynamically from tool-call
                # round-trips, which is broader than those TypedDicts by
                # necessity. The runtime shape matches — we round-trip
                # OpenAI's own output back to it.
                response = self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=parallel_tool_calls,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                tool_calls = []
                for call in message.tool_calls or []:
                    fn = getattr(call, "function", None)
                    if fn is None:
                        continue
                    raw_args = getattr(fn, "arguments", "") or "{}"
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    tool_calls.append(
                        ChatToolCall(
                            id=str(call.id),
                            name=str(fn.name),
                            arguments=parsed_args,
                        )
                    )
                usage = response.usage
                tokens_in = int(usage.prompt_tokens) if usage else 0
                tokens_out = int(usage.completion_tokens) if usage else 0
                finish_reason = str(choice.finish_reason or "")
        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            finish_reason=finish_reason,
        )
