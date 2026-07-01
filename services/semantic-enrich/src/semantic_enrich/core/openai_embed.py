"""OpenAI embedding batching helper.

Shared by:
  - `embedding_pass.run_embed` / `run_columns_embed` (post-4.7 the
    stage-JSONL embed passes call OpenAI instead of the local Qwen
    model),
  - `reembed` (one-off warehouse reembed from BQ source text).

Both callers batch a list of texts against the same client Protocol,
so the batching + dim-validation loop lives here.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import structlog

from semantic_enrich.clients.openai import OpenAIClient

VectorFailureReason = Literal["wrong_dim", "has_nan", "has_inf"]


@dataclass(frozen=True)
class EmbedVectorResult:
    """One row-shaped outcome of `embed_texts_in_batches`.

    `vector` is populated on success; `failure_reason` on validation
    failure. The two are mutually exclusive.
    """

    vector: list[float] | None
    failure_reason: VectorFailureReason | None


def embed_texts_in_batches(
    *,
    client: OpenAIClient,
    texts: list[str],
    batch_size: int,
    expected_dim: int,
    log: structlog.BoundLogger,
    log_event_prefix: str,
) -> list[EmbedVectorResult]:
    """Batch-embed `texts`; validate dim + finiteness per vector.

    A batch that raises (rate-limit exhaustion, 5xx after retry, etc.)
    is not caught here — the caller decides whether one poisoned batch
    should fail the whole run (the reembed contract: yes) or degrade to
    per-row failure accounting (the stage-embed contract: yes, but
    counted as `embeddings_failed`).
    """
    results: list[EmbedVectorResult] = [
        EmbedVectorResult(vector=None, failure_reason=None) for _ in texts
    ]
    for batch_index, batch_start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[batch_start : batch_start + batch_size]
        t0 = time.monotonic()
        vectors = client.embed(batch)
        log.info(
            f"{log_event_prefix}_batch_done",
            batch_index=batch_index,
            batch_size=len(batch),
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"openai embedding response length mismatch: expected "
                f"{len(batch)} vectors, got {len(vectors)}"
            )
        for offset, vec in enumerate(vectors):
            row_index = batch_start + offset
            reason = _validate_vector(vec, expected_dim=expected_dim)
            if reason is not None:
                results[row_index] = EmbedVectorResult(
                    vector=None, failure_reason=reason
                )
            else:
                results[row_index] = EmbedVectorResult(
                    vector=vec, failure_reason=None
                )
    return results


def _validate_vector(
    vec: list[float], *, expected_dim: int
) -> VectorFailureReason | None:
    if len(vec) != expected_dim:
        return "wrong_dim"
    for x in vec:
        if math.isnan(x):
            return "has_nan"
        if math.isinf(x):
            return "has_inf"
    return None
