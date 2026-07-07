"""Guided JSON generation via outlines (>=1.0) + transformers.

outlines 1.x removed the old `outlines.models.transformers(repo, ...)`
factory. Loaders now pass a pre-constructed HF model + tokenizer
through `outlines.from_transformers(...)`, and generation is a direct
call on the returned model — `model(prompt, schema, max_new_tokens=N)`
— rather than the older `outlines.generate.json(...)` two-step.
"""
from __future__ import annotations

import json
from typing import Any

import pydantic

from semantic_enrich.types import GenerationModel, MaxTokensExceededError


def load_generation_model(
    repo: str = "Qwen/Qwen2.5-14B-Instruct",
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    cache_dir: str | None = None,
) -> GenerationModel:
    """Load the generation model once. Caller owns the lifetime.

    `dtype` is a torch dtype name (`"bfloat16"`, `"float16"`, `"float32"`)
    rather than the torch object so the loader can be invoked without a
    torch import on the call-site. `device` is forwarded as
    `device_map` to `AutoModelForCausalLM.from_pretrained` — passing
    `"cuda"` lets accelerate place the weights, and `"cuda:0"` /
    `"cuda:1"` pins to a specific card when the box has more than one.
    """
    try:
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies not installed. "
            "GPU box (conda): pip install -e '.[gpu]'  "
            "GPU box (uv): uv sync --extra gpu"
        ) from exc

    torch_dtype = getattr(torch, dtype)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device,
    }
    tokenizer_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        model_kwargs["cache_dir"] = cache_dir
        tokenizer_kwargs["cache_dir"] = cache_dir

    tf_model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(repo, **tokenizer_kwargs)  # type: ignore[no-untyped-call]

    return outlines.from_transformers(tf_model, tokenizer)


def _as_output_type(schema: Any) -> Any:
    """Normalize a `schema` argument into an outlines 1.x output type.

    outlines 1.x does not accept a bare JSON-Schema dict as an output
    type — `python_types_to_terms` rejects it (and a top-level `array`
    schema raises outright). A dict must be wrapped in
    `outlines.types.JsonSchema`; pydantic model classes and typing
    generics (`list[ColumnOutput]`) are already valid output types and
    pass through untouched.
    """
    if isinstance(schema, dict):
        from outlines.types import JsonSchema

        return JsonSchema(schema)
    return schema


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
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature == 0.0:
        # Greedy decoding — silences the transformers `temperature=0`
        # warning and matches the PRD's "deterministic by default".
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    try:
        result = model(prompt, _as_output_type(schema), **gen_kwargs)
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        raise MaxTokensExceededError(
            f"constrained JSON generation produced malformed output; "
            f"likely truncated at max_tokens={max_tokens}",
        ) from exc

    return _coerce_to_dict(result, max_tokens=max_tokens)


def generate_json_list(
    prompt: str,
    schema: dict[str, Any] | type[pydantic.BaseModel] | Any,
    *,
    model: GenerationModel,
    max_tokens: int = 1500,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Run one constrained-JSON generation that returns a JSON array.

    Same shape as `generate_json`, but the response is a `list[dict]`
    rather than a `dict`. Used by 4.5's columns generator where the
    prompt asks the model for one JSON array per chunk.

    `schema` may be a JSON-Schema dict (e.g. `COLUMNS_GUIDED_JSON_SCHEMA`)
    or a Python generic like `list[ColumnOutput]`. A dict is wrapped via
    `_as_output_type` into `outlines.types.JsonSchema`, since outlines 1.x
    rejects a bare dict as an output type.
    """
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    try:
        result = model(prompt, _as_output_type(schema), **gen_kwargs)
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        raise MaxTokensExceededError(
            f"constrained JSON-array generation produced malformed output; "
            f"likely truncated at max_tokens={max_tokens}",
        ) from exc

    return _coerce_to_list(result, max_tokens=max_tokens)


def get_tokenizer(model: GenerationModel) -> Any:
    """Return the HF tokenizer backing the outlines model.

    Outlines 1.x wraps the HF tokenizer in its own `TransformerTokenizer`
    adapter at `model.tokenizer`; the original (with the
    `apply_chat_template` method) is one level deeper at
    `model.tokenizer.tokenizer`. This helper probes the common paths
    and returns the first one that actually exposes
    `apply_chat_template`, so call sites stay free of
    outlines-internal field names.
    """
    candidates: tuple[tuple[str, ...], ...] = (
        ("tokenizer", "tokenizer"),   # outlines 1.x adapter -> underlying HF
        ("tokenizer",),
        ("hf_tokenizer",),
        ("model", "tokenizer"),
        ("transformers", "tokenizer"),
    )
    for attr_chain in candidates:
        obj: Any = model
        ok = True
        for attr in attr_chain:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None and hasattr(obj, "apply_chat_template"):
            return obj
    raise RuntimeError(
        "could not locate HF tokenizer with apply_chat_template on "
        "outlines GenerationModel; outlines internals may have shifted "
        "— update get_tokenizer()."
    )


def _coerce_to_dict(result: Any, *, max_tokens: int) -> dict[str, Any]:
    """Normalise outlines's return shape into a plain dict.

    outlines 1.x returns a JSON string when handed a pydantic class or
    dict schema. Earlier 0.x dialects returned a pydantic instance or a
    dict directly. Accept all three so the wrapper survives a future
    library flip without churning the call sites.
    """
    if isinstance(result, pydantic.BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise MaxTokensExceededError(
                f"outlines returned non-JSON string at max_tokens={max_tokens}: "
                f"{result[:200]!r}",
            ) from exc
        if not isinstance(parsed, dict):
            raise TypeError(
                f"outlines returned JSON that decoded to "
                f"{type(parsed).__name__}; expected an object",
            )
        return parsed
    raise TypeError(
        f"outlines returned unexpected type {type(result).__name__}; "
        "expected dict, str, or pydantic.BaseModel",
    )


def _coerce_to_list(result: Any, *, max_tokens: int) -> list[dict[str, Any]]:
    """Normalise outlines's return shape into a list of dicts.

    Mirrors `_coerce_to_dict` but for array-shaped schemas. Accepts a
    Python list of dicts (or pydantic models) and a JSON string.
    """
    if isinstance(result, list):
        out: list[dict[str, Any]] = []
        for entry in result:
            if isinstance(entry, pydantic.BaseModel):
                out.append(entry.model_dump())
            elif isinstance(entry, dict):
                out.append(entry)
            else:
                raise TypeError(
                    f"outlines list entry has unexpected type "
                    f"{type(entry).__name__}; expected dict or "
                    "pydantic.BaseModel",
                )
        return out
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise MaxTokensExceededError(
                f"outlines returned non-JSON string at max_tokens={max_tokens}: "
                f"{result[:200]!r}",
            ) from exc
        if not isinstance(parsed, list):
            raise TypeError(
                f"outlines returned JSON that decoded to "
                f"{type(parsed).__name__}; expected an array",
            )
        return [dict(e) for e in parsed]
    raise TypeError(
        f"outlines returned unexpected type {type(result).__name__}; "
        "expected list or str for a list-schema response",
    )
