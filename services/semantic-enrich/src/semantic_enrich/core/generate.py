"""Guided JSON generation via outlines + transformers.

Two functions: `load_generation_model` (owned by the caller, lifetime
spans the generation pass) and `generate_json` (one constrained-decode
call per invocation). The downstream pipeline keeps the model loaded
across its per-package loop and drops it before the embedding pass.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pydantic

from semantic_enrich.types import GenerationModel, MaxTokensExceededError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def load_generation_model(
    repo: str = "Qwen/Qwen2.5-14B-Instruct",
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    cache_dir: str | None = None,
) -> GenerationModel:
    """Load the generation model once. Caller owns the lifetime.

    `dtype` is a torch dtype name (`"bfloat16"`, `"float16"`, `"float32"`)
    rather than the torch object so the loader can be invoked without
    a torch import on the call-site.
    """
    import torch
    from outlines import models

    torch_dtype = getattr(torch, dtype)
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if cache_dir is not None:
        model_kwargs["cache_dir"] = cache_dir

    return models.transformers(
        repo,
        device=device,
        model_kwargs=model_kwargs,
    )


def generate_json(
    prompt: str,
    schema: dict[str, Any] | type[pydantic.BaseModel],
    *,
    model: GenerationModel,
    max_tokens: int = 1500,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run one constrained-JSON generation. Returns a parsed dict.

    `schema` is either a JSON Schema dict or a pydantic model class.
    Outlines constrains the decoder so the returned object always
    parses and conforms to the schema; the only failure modes are
    `max_tokens` truncation (raises `MaxTokensExceededError`) and OOM
    (raises whatever torch raises).
    """
    from outlines import generate

    generator = generate.json(model, schema)
    try:
        result = generator(prompt, max_tokens=max_tokens, temperature=temperature)
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        raise MaxTokensExceededError(
            f"constrained JSON generation produced malformed output; "
            f"likely truncated at max_tokens={max_tokens}",
        ) from exc

    if isinstance(result, pydantic.BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    # outlines guarantees one of the two above when given a schema,
    # but the runtime type is `Any` so we narrow defensively.
    raise TypeError(
        f"outlines returned unexpected type {type(result).__name__}; "
        "expected dict or pydantic.BaseModel",
    )
