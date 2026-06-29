"""`core.generate.generate_json` happy-path + truncation paths.

The runtime deps (`outlines`, `torch`) are real package installs (main
deps, not `[dev]`). Tests mock the call boundaries — `outlines.models.transformers`
and `outlines.generate.json` — so no real model load happens in CI.
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pydantic
import pytest

from semantic_enrich.core.generate import generate_json, load_generation_model
from semantic_enrich.core.schemas import SmokeOutput
from semantic_enrich.types import MaxTokensExceededError


def _install_fake_outlines(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generator_return: Any = None,
    generator_raises: Exception | None = None,
    factory: MagicMock | None = None,
) -> MagicMock:
    """Wire a fake `outlines` package into `sys.modules`.

    `from outlines import generate` resolves against `sys.modules` —
    populating it here lets the unit tests run on a host where outlines
    isn't installed (laptop dev) and on the GPU box (real install) alike.
    """
    generator_fn = MagicMock(name="outlines-generator")
    if generator_raises is not None:
        generator_fn.side_effect = generator_raises
    else:
        generator_fn.return_value = generator_return

    factory = factory or MagicMock(return_value=generator_fn)

    fake_generate = types.ModuleType("outlines.generate")
    fake_generate.json = factory  # type: ignore[attr-defined]

    fake_models = types.ModuleType("outlines.models")
    fake_models.transformers = MagicMock(return_value="fake-loaded-model")  # type: ignore[attr-defined]

    fake_outlines = types.ModuleType("outlines")
    fake_outlines.generate = fake_generate  # type: ignore[attr-defined]
    fake_outlines.models = fake_models  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.generate", fake_generate)
    monkeypatch.setitem(sys.modules, "outlines.models", fake_models)
    return generator_fn


def test_generate_json_returns_dict_from_pydantic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = SmokeOutput(package_id="pkg-1", summary="A fictional dataset.")
    _install_fake_outlines(monkeypatch, generator_return=instance)

    result = generate_json(
        "ignored prompt",
        SmokeOutput,
        model=MagicMock(),
    )

    assert result == {"package_id": "pkg-1", "summary": "A fictional dataset."}


def test_generate_json_returns_dict_from_dict_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_outlines(
        monkeypatch,
        generator_return={"package_id": "pkg-2", "summary": "Another."},
    )

    result = generate_json(
        "ignored",
        {"type": "object"},
        model=MagicMock(),
    )

    assert result == {"package_id": "pkg-2", "summary": "Another."}


def test_generate_json_passes_max_tokens_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_fn = _install_fake_outlines(
        monkeypatch,
        generator_return={"package_id": "p", "summary": "s"},
    )

    generate_json(
        "prompt",
        SmokeOutput,
        model=MagicMock(),
        max_tokens=42,
        temperature=0.5,
    )

    generator_fn.assert_called_once_with("prompt", max_tokens=42, temperature=0.5)


def test_generate_json_raises_max_tokens_on_jsondecodeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_outlines(
        monkeypatch,
        generator_raises=json.JSONDecodeError("expected ','", "{}", 0),
    )

    with pytest.raises(MaxTokensExceededError, match="max_tokens"):
        generate_json("p", SmokeOutput, model=MagicMock(), max_tokens=10)


def test_generate_json_raises_max_tokens_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        SmokeOutput.model_validate({})
    except pydantic.ValidationError as exc:
        validation_exc: Exception = exc
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValidationError")

    _install_fake_outlines(monkeypatch, generator_raises=validation_exc)

    with pytest.raises(MaxTokensExceededError):
        generate_json("p", SmokeOutput, model=MagicMock())


def test_generate_json_rejects_unexpected_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_outlines(monkeypatch, generator_return="not a dict or model")

    with pytest.raises(TypeError, match="unexpected type"):
        generate_json("p", SmokeOutput, model=MagicMock())


def test_load_generation_model_invokes_outlines_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16-sentinel"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_models = types.ModuleType("outlines.models")
    transformers_factory = MagicMock(return_value="loaded-model")
    fake_models.transformers = transformers_factory  # type: ignore[attr-defined]

    fake_outlines = types.ModuleType("outlines")
    fake_outlines.models = fake_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.models", fake_models)

    result = load_generation_model(
        "Qwen/Qwen2.5-14B-Instruct",
        dtype="bfloat16",
        device="cuda",
        cache_dir="/tmp/hf",
    )

    assert result == "loaded-model"
    transformers_factory.assert_called_once_with(
        "Qwen/Qwen2.5-14B-Instruct",
        device="cuda",
        model_kwargs={"torch_dtype": "bf16-sentinel", "cache_dir": "/tmp/hf"},
    )


def test_load_generation_model_omits_cache_dir_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_models = types.ModuleType("outlines.models")
    factory = MagicMock(return_value="m")
    fake_models.transformers = factory  # type: ignore[attr-defined]
    fake_outlines = types.ModuleType("outlines")
    fake_outlines.models = fake_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.models", fake_models)

    load_generation_model("repo", dtype="bfloat16", device="cuda", cache_dir=None)

    assert "cache_dir" not in factory.call_args.kwargs["model_kwargs"]
