"""Shared OpenAI-client fake used across the embed / reembed / eval tests.

Implements the `OpenAIClient` Protocol so runners see the same surface
they would in prod, with hooks:

- `vector_factory(text) -> list[float]` decides what vector to return
  per input string. Defaults to a deterministic 1536-dim unit vector.
- `calls` records every batch sent to `embed()` so tests can assert
  batching semantics.
- `structured_responses` is a list of `parsed` dicts that
  `generate_structured` pops FIFO. `structured_calls` records every
  call so eval tests can assert the SQL-gen prompt shape without
  round-tripping the vendor.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

from semantic_enrich.clients.openai import (
    ChatCompletionResult,
    ChatToolCall,
    StructuredGenerationResult,
)


def _default_1536(text: str) -> list[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    raw = [((seed >> (i % 60)) & 0xFF) / 255.0 + 0.001 for i in range(1536)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class FakeOpenAIClient:
    def __init__(
        self,
        *,
        vector_factory: Callable[[str], list[float]] = _default_1536,
        structured_responses: list[dict[str, Any]] | None = None,
        structured_tokens: tuple[int, int] = (100, 50),
        chat_script: list[dict[str, Any]] | None = None,
        chat_tokens: tuple[int, int] = (100, 50),
    ) -> None:
        self._vector_factory = vector_factory
        self.calls: list[list[str]] = []
        self.structured_responses: list[dict[str, Any]] = list(
            structured_responses or []
        )
        self.structured_tokens = structured_tokens
        self.structured_calls: list[dict[str, Any]] = []
        # 5.1 agent-loop hook. Each entry is one scripted assistant
        # response — either `{"content": "..."}` for a terminal message
        # or `{"tool_calls": [{"id", "name", "arguments"}]}` for tool
        # calls. `chat_calls` records every request the loop made.
        self.chat_script: list[dict[str, Any]] = list(chat_script or [])
        self.chat_tokens = chat_tokens
        self.chat_calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_factory(t) for t in texts]

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
        self.structured_calls.append(
            {
                "prompt": prompt,
                "schema_name": schema_name,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        # Canary preflight is handled independently of the per-question
        # response queue so tests don't need to prepend an "ok" entry.
        if schema_name == "canary":
            parsed: dict[str, Any] = {"ok": "yes"}
        elif self.structured_responses:
            parsed = self.structured_responses.pop(0)
        else:
            parsed = {
                "sql": "SELECT 1 AS n FROM `proj.raw.rows` LIMIT 10",
                "rationale": "canned",
                "answer_summary": "one row",
            }
        return StructuredGenerationResult(
            parsed=parsed,
            tokens_in=self.structured_tokens[0],
            tokens_out=self.structured_tokens[1],
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
        self.chat_calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
        if not self.chat_script:
            return ChatCompletionResult(
                content="[fake] no scripted response.",
                tool_calls=[],
                tokens_in=self.chat_tokens[0],
                tokens_out=self.chat_tokens[1],
                finish_reason="stop",
            )
        step = self.chat_script.pop(0)
        content = str(step.get("content", ""))
        tool_calls_payload = step.get("tool_calls") or []
        tool_calls = [
            ChatToolCall(
                id=str(tc.get("id", f"call_{i}")),
                name=str(tc["name"]),
                arguments=dict(tc.get("arguments", {})),
            )
            for i, tc in enumerate(tool_calls_payload)
        ]
        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            tokens_in=int(step.get("tokens_in", self.chat_tokens[0])),
            tokens_out=int(step.get("tokens_out", self.chat_tokens[1])),
            finish_reason="tool_calls" if tool_calls else "stop",
        )
