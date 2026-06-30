"""`core.generate.generate_json` happy-path + truncation paths.

Tests mock the model object's `__call__` (the outlines 1.x boundary)
and inject fake `transformers` / `outlines` / `torch` modules via
`sys.modules` so the suite runs on hosts where the heavy deps aren't
installed.
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


def test_generate_json_returns_dict_from_pydantic_result() -> None:
    instance = SmokeOutput(package_id="pkg-1", summary="A fictional dataset.")
    model = MagicMock(return_value=instance)

    result = generate_json("prompt", SmokeOutput, model=model)

    assert result == {"package_id": "pkg-1", "summary": "A fictional dataset."}


def test_generate_json_returns_dict_from_string_result() -> None:
    model = MagicMock(return_value='{"package_id": "pkg-2", "summary": "Another."}')

    result = generate_json("p", SmokeOutput, model=model)

    assert result == {"package_id": "pkg-2", "summary": "Another."}


def test_generate_json_returns_dict_from_dict_result() -> None:
    model = MagicMock(return_value={"package_id": "pkg-3", "summary": "Third."})

    result = generate_json("p", {"type": "object"}, model=model)

    assert result == {"package_id": "pkg-3", "summary": "Third."}


def test_generate_json_greedy_decoding_for_zero_temperature() -> None:
    model = MagicMock(return_value={"package_id": "p", "summary": "s"})

    generate_json("prompt", SmokeOutput, model=model, max_tokens=100, temperature=0.0)

    call = model.call_args
    assert call.args == ("prompt", SmokeOutput)
    assert call.kwargs["max_new_tokens"] == 100
    assert call.kwargs["do_sample"] is False
    # `temperature` must not be passed when do_sample=False — the HF
    # generation path warns otherwise.
    assert "temperature" not in call.kwargs


def test_generate_json_samples_for_nonzero_temperature() -> None:
    model = MagicMock(return_value={"package_id": "p", "summary": "s"})

    generate_json("prompt", SmokeOutput, model=model, max_tokens=42, temperature=0.7)

    call = model.call_args
    assert call.kwargs == {"max_new_tokens": 42, "do_sample": True, "temperature": 0.7}


def test_generate_json_raises_max_tokens_on_jsondecodeerror_from_model() -> None:
    model = MagicMock(side_effect=json.JSONDecodeError("expected ','", "{}", 0))

    with pytest.raises(MaxTokensExceededError, match="max_tokens"):
        generate_json("p", SmokeOutput, model=model, max_tokens=10)


def test_generate_json_raises_max_tokens_on_validation_error_from_model() -> None:
    try:
        SmokeOutput.model_validate({})
    except pydantic.ValidationError as exc:
        validation_exc: Exception = exc
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValidationError")

    model = MagicMock(side_effect=validation_exc)

    with pytest.raises(MaxTokensExceededError):
        generate_json("p", SmokeOutput, model=model)


def test_generate_json_raises_max_tokens_on_unparseable_string() -> None:
    model = MagicMock(return_value="not valid json")

    with pytest.raises(MaxTokensExceededError, match="non-JSON"):
        generate_json("p", SmokeOutput, model=model)


def test_generate_json_rejects_non_object_json() -> None:
    model = MagicMock(return_value="[1, 2, 3]")

    with pytest.raises(TypeError, match="expected an object"):
        generate_json("p", SmokeOutput, model=model)


def test_generate_json_rejects_unexpected_type() -> None:
    model = MagicMock(return_value=12345)

    with pytest.raises(TypeError, match="unexpected type"):
        generate_json("p", SmokeOutput, model=model)


def _install_fake_heavy_deps(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub torch, transformers, and outlines in sys.modules.

    Returns the mocks so individual tests can assert against them. The
    real packages stay uninstalled in the test env — `load_generation_model`
    only touches them through these three symbols.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16-sentinel"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    auto_model_cls = MagicMock(name="AutoModelForCausalLM")
    auto_model_cls.from_pretrained.return_value = "tf-model-handle"
    auto_tokenizer_cls = MagicMock(name="AutoTokenizer")
    auto_tokenizer_cls.from_pretrained.return_value = "tokenizer-handle"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = auto_model_cls  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = auto_tokenizer_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from_transformers = MagicMock(return_value="wrapped-outlines-model")
    fake_outlines = types.ModuleType("outlines")
    fake_outlines.from_transformers = from_transformers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)

    return {
        "auto_model_cls": auto_model_cls,
        "auto_tokenizer_cls": auto_tokenizer_cls,
        "from_transformers": from_transformers,
    }


def test_load_generation_model_wires_transformers_into_outlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_fake_heavy_deps(monkeypatch)

    result = load_generation_model(
        "Qwen/Qwen2.5-14B-Instruct",
        dtype="bfloat16",
        device="cuda:0",
        cache_dir="/tmp/hf",
    )

    assert result == "wrapped-outlines-model"
    mocks["auto_model_cls"].from_pretrained.assert_called_once_with(
        "Qwen/Qwen2.5-14B-Instruct",
        torch_dtype="bf16-sentinel",
        device_map="cuda:0",
        cache_dir="/tmp/hf",
    )
    mocks["auto_tokenizer_cls"].from_pretrained.assert_called_once_with(
        "Qwen/Qwen2.5-14B-Instruct",
        cache_dir="/tmp/hf",
    )
    mocks["from_transformers"].assert_called_once_with(
        "tf-model-handle",
        "tokenizer-handle",
    )


def test_load_generation_model_omits_cache_dir_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_fake_heavy_deps(monkeypatch)

    load_generation_model("repo", dtype="bfloat16", device="cuda", cache_dir=None)

    model_kwargs = mocks["auto_model_cls"].from_pretrained.call_args.kwargs
    tokenizer_kwargs = mocks["auto_tokenizer_cls"].from_pretrained.call_args.kwargs
    assert "cache_dir" not in model_kwargs
    assert "cache_dir" not in tokenizer_kwargs
