"""Smoke-test runner.

One generation round-trip, drop generation model from VRAM, one
embedding round-trip. Asserts shape + content on both halves; returns
a `SmokeResult` whose fields the CLI translates to exit codes.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.embed import embed_batch, load_embedding_model
from semantic_enrich.core.generate import generate_json, load_generation_model
from semantic_enrich.core.schemas import SmokeOutput
from semantic_enrich.types import (
    EmbeddingModel,
    GenerationModel,
    MaxTokensExceededError,
    SmokeResult,
)

SMOKE_PROMPT = (
    "Return a JSON object describing a fictional Canadian government "
    "open dataset. The object must have exactly two fields: "
    "`package_id` (a short identifier string, e.g. 'pkg-001'), and "
    "`summary` (a one-sentence description of the dataset, no more "
    "than 500 characters)."
)

SMOKE_EMBED_INPUT = "Quarterly federal employment statistics for the Canadian public service."

EXPECTED_EMBED_DIM = 1024
EMBED_NORM_TOLERANCE = 0.01


def _drop_cuda_cache() -> None:
    """Free the generation model's VRAM before loading the embedder.

    `del gen_model` only releases the Python reference; the CUDA
    allocator holds the workspace until `empty_cache()` runs. A `gc`
    pass first is belt-and-suspenders for any cyclic references the
    transformers model wires up.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_smoke_test(
    *,
    settings: Settings,
    load_generation_model_fn: Callable[..., GenerationModel] = load_generation_model,
    generate_json_fn: Callable[..., dict[str, Any]] = generate_json,
    load_embedding_model_fn: Callable[..., EmbeddingModel] = load_embedding_model,
    embed_batch_fn: Callable[..., list[list[float]]] = embed_batch,
    drop_cuda_cache_fn: Callable[[], None] = _drop_cuda_cache,
) -> SmokeResult:
    """Run the two-pass smoke test. Returns a `SmokeResult`.

    Precondition failures (model load error, schema violation,
    dimension or L2-norm drift) populate `precondition_failure` and
    set `ok=False`. Internal errors (bugs in this runner) are NOT
    caught — they propagate so the CLI can map them to exit 1.
    """
    started = time.monotonic()

    cache_dir = str(settings.hf_cache_dir) if settings.hf_cache_dir else None

    # Generation pass.
    try:
        gen_model = load_generation_model_fn(
            settings.generation_model,
            device=settings.device,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        return _failure(
            f"generation_model_load_failed: {exc}",
            duration_ms=_elapsed_ms(started),
        )

    try:
        gen_output = generate_json_fn(
            SMOKE_PROMPT,
            SmokeOutput,
            model=gen_model,
        )
    except MaxTokensExceededError as exc:
        return _failure(
            f"generation_truncated: {exc}",
            duration_ms=_elapsed_ms(started),
        )

    if not isinstance(gen_output, dict):
        return _failure(
            f"generation_not_dict: returned {type(gen_output).__name__}",
            duration_ms=_elapsed_ms(started),
        )

    try:
        validated = SmokeOutput.model_validate(gen_output)
    except pydantic.ValidationError as exc:
        return _failure(
            f"generation_schema_violation: {exc.errors()}",
            duration_ms=_elapsed_ms(started),
        )

    if not validated.package_id.strip() or not validated.summary.strip():
        return _failure(
            "generation_empty_after_strip",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
        )

    # Drop the generation model before loading the embedder. The 4.4
    # pipeline does the same — keeping both co-resident would blow
    # the 48 GB A6000 budget.
    del gen_model
    drop_cuda_cache_fn()

    # Embedding pass.
    try:
        emb_model = load_embedding_model_fn(
            settings.embedding_model,
            device=settings.device,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        return _failure(
            f"embedding_model_load_failed: {exc}",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
        )

    try:
        vectors = embed_batch_fn([SMOKE_EMBED_INPUT], model=emb_model)
    except Exception as exc:
        return _failure(
            f"embedding_encode_failed: {exc}",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
        )

    if not vectors or len(vectors) != 1:
        return _failure(
            f"embedding_batch_size_mismatch: got {len(vectors)} expected 1",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
        )

    vector = vectors[0]
    dim = len(vector)
    if dim != EXPECTED_EMBED_DIM:
        return _failure(
            f"embedding_dim_drift: got {dim} expected {EXPECTED_EMBED_DIM}",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
            embedding_dim=dim,
        )

    # All-zeros check runs before the L2-norm check so the distinct
    # failure mode surfaces with its own reason. A zero vector also
    # fails norm (sqrt(0) = 0, far from 1.0), but reporting
    # `embedding_all_zeros` points at "the encoder returned nothing"
    # instead of the vaguer "norm drifted".
    if all(component == 0.0 for component in vector):
        return _failure(
            "embedding_all_zeros",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
            embedding_dim=dim,
            embedding_norm=0.0,
        )

    norm = math.sqrt(sum(component * component for component in vector))
    if abs(norm - 1.0) > EMBED_NORM_TOLERANCE:
        return _failure(
            f"embedding_norm_drift: got {norm} expected 1.0 +/- {EMBED_NORM_TOLERANCE}",
            duration_ms=_elapsed_ms(started),
            generation_output=gen_output,
            embedding_dim=dim,
            embedding_norm=norm,
        )

    return SmokeResult(
        ok=True,
        precondition_failure=None,
        generation_output=gen_output,
        embedding_dim=dim,
        embedding_norm=norm,
        duration_ms=_elapsed_ms(started),
    )


def write_models_lock(
    path: Path,
    *,
    generation_repo: str,
    embedding_repo: str,
) -> dict[str, Any]:
    """Capture resolved package versions + HF commit SHAs to MODELS.lock.

    Called from `smoke-test --write-lock` on first successful run.
    Operator commits the resulting file so future runs are reproducible.
    """
    from importlib import metadata

    from huggingface_hub import HfApi

    packages: dict[str, str | None] = {}
    for pkg in (
        "transformers",
        "torch",
        "accelerate",
        "outlines",
        "sentence-transformers",
        "pydantic",
        "huggingface-hub",
    ):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None

    api = HfApi()
    payload: dict[str, Any] = {
        "generation": {
            "repo": generation_repo,
            "commit": api.model_info(generation_repo).sha,
        },
        "embedding": {
            "repo": embedding_repo,
            "commit": api.model_info(embedding_repo).sha,
        },
        "packages": packages,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _failure(
    reason: str,
    *,
    duration_ms: int,
    generation_output: dict[str, Any] | None = None,
    embedding_dim: int | None = None,
    embedding_norm: float | None = None,
) -> SmokeResult:
    return SmokeResult(
        ok=False,
        precondition_failure=reason,
        generation_output=generation_output,
        embedding_dim=embedding_dim,
        embedding_norm=embedding_norm,
        duration_ms=duration_ms,
    )
