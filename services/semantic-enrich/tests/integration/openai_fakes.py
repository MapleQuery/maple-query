"""Shared OpenAI-client fake used across the embed / reembed tests.

Implements the `OpenAIClient` Protocol so runners see the same surface
they would in prod, with two hooks:

- `vector_factory(text) -> list[float]` decides what vector to return
  per input string. Defaults to a deterministic 1536-dim unit vector.
- `calls` records every batch sent to `embed()` so tests can assert
  batching semantics.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable


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
    ) -> None:
        self._vector_factory = vector_factory
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_factory(t) for t in texts]
