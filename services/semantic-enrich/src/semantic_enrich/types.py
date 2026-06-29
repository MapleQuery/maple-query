"""Shared type aliases and dataclasses.

Pure data shapes only. No model SDKs imported here so it can sit at
the bottom of the layer stack and import-free into the smoke-test
result path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Outlines exposes the model object through a factory rather than a
# stable public class. Treat it as opaque at our boundary — the
# function signatures document intent without locking us to a private
# import path that could shift between minor releases.
GenerationModel = Any
EmbeddingModel = Any


class MaxTokensExceededError(RuntimeError):
    """Raised when constrained-JSON generation produced malformed output.

    Outlines enforces JSON-Schema conformance per-token; a parse or
    schema-validation failure means the decoder hit `max_tokens` before
    closing the structure. Callers should re-run with a larger budget,
    not retry blindly.
    """


@dataclass(frozen=True)
class SmokeResult:
    """End-of-smoke roll-up. Maps to exit codes in the CLI:

    - `ok=True`                                  → exit 0
    - `ok=False` and `precondition_failure`      → exit 2
    - any uncaught exception in the runner       → exit 1 (handled by CLI)
    """

    ok: bool
    precondition_failure: str | None
    generation_output: dict[str, Any] | None
    embedding_dim: int | None
    embedding_norm: float | None
    duration_ms: int
