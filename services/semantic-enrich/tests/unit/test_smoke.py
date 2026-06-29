"""Smoke-test runner — one assertion per exit-code path.

Each test pumps a fake load/generate/embed callable into
`run_smoke_test` instead of patching at the module boundary; that
makes the assertions point at the runner's branches, not at the
mock-wiring scaffolding.
"""
from __future__ import annotations

import math
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import smoke as smoke_mod
from semantic_enrich.core.smoke import (
    EMBED_NORM_TOLERANCE,
    EXPECTED_EMBED_DIM,
    run_smoke_test,
    write_models_lock,
)
from semantic_enrich.types import MaxTokensExceededError


def _unit_vector(dim: int = EXPECTED_EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    vec[0] = 1.0
    return vec


def _settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def _kwargs(
    *,
    load_gen: Callable[..., Any] | None = None,
    gen_json: Callable[..., Any] | None = None,
    load_emb: Callable[..., Any] | None = None,
    emb_batch: Callable[..., Any] | None = None,
    drop_cache: Callable[[], None] | None = None,
) -> dict[str, Any]:
    return {
        "load_generation_model_fn": load_gen
        or (lambda *_a, **_k: MagicMock(name="gen-model")),
        "generate_json_fn": gen_json
        or (lambda *_a, **_k: {"package_id": "pkg-1", "summary": "ok"}),
        "load_embedding_model_fn": load_emb
        or (lambda *_a, **_k: MagicMock(name="emb-model")),
        "embed_batch_fn": emb_batch or (lambda *_a, **_k: [_unit_vector()]),
        "drop_cuda_cache_fn": drop_cache or (lambda: None),
    }


def test_smoke_happy_path_returns_ok() -> None:
    result = run_smoke_test(settings=_settings(), **_kwargs())

    assert result.ok is True
    assert result.precondition_failure is None
    assert result.generation_output == {"package_id": "pkg-1", "summary": "ok"}
    assert result.embedding_dim == EXPECTED_EMBED_DIM
    assert math.isclose(result.embedding_norm or 0.0, 1.0, abs_tol=EMBED_NORM_TOLERANCE)
    assert result.duration_ms >= 0


def test_smoke_generation_load_failure_precondition() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA OOM on weight upload")

    result = run_smoke_test(settings=_settings(), **_kwargs(load_gen=boom))

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "generation_model_load_failed" in result.precondition_failure


def test_smoke_generation_truncation_precondition() -> None:
    def truncated(*_a: Any, **_k: Any) -> Any:
        raise MaxTokensExceededError("simulated truncation")

    result = run_smoke_test(settings=_settings(), **_kwargs(gen_json=truncated))

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "generation_truncated" in result.precondition_failure


def test_smoke_generation_schema_violation_precondition() -> None:
    # summary > 500 chars violates SmokeOutput's max_length constraint.
    bad_payload = {"package_id": "pkg-1", "summary": "x" * 600}

    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(gen_json=lambda *_a, **_k: bad_payload),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "generation_schema_violation" in result.precondition_failure


def test_smoke_generation_empty_after_strip_precondition() -> None:
    # min_length=1 passes whitespace; the explicit strip check catches it.
    blank_payload = {"package_id": "   ", "summary": "real text"}

    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(gen_json=lambda *_a, **_k: blank_payload),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "generation_empty_after_strip" in result.precondition_failure


def test_smoke_generation_non_dict_precondition() -> None:
    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(gen_json=lambda *_a, **_k: "not a dict"),  # type: ignore[arg-type]
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "generation_not_dict" in result.precondition_failure


def test_smoke_embedding_load_failure_precondition() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("disk full")

    result = run_smoke_test(settings=_settings(), **_kwargs(load_emb=boom))

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_model_load_failed" in result.precondition_failure


def test_smoke_embedding_encode_failure_precondition() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("encode crashed")

    result = run_smoke_test(settings=_settings(), **_kwargs(emb_batch=boom))

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_encode_failed" in result.precondition_failure


def test_smoke_embedding_dim_drift_precondition() -> None:
    short_vec = [_unit_vector(dim=512)]
    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(emb_batch=lambda *_a, **_k: short_vec),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_dim_drift" in result.precondition_failure
    assert result.embedding_dim == 512


def test_smoke_embedding_norm_drift_precondition() -> None:
    drift = [2.0, *([0.0] * (EXPECTED_EMBED_DIM - 1))]
    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(emb_batch=lambda *_a, **_k: [drift]),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_norm_drift" in result.precondition_failure


def test_smoke_embedding_all_zeros_precondition() -> None:
    zeros = [[0.0] * EXPECTED_EMBED_DIM]
    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(emb_batch=lambda *_a, **_k: zeros),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_all_zeros" in result.precondition_failure


def test_smoke_embedding_empty_batch_precondition() -> None:
    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(emb_batch=lambda *_a, **_k: []),
    )

    assert result.ok is False
    assert result.precondition_failure is not None
    assert "embedding_batch_size_mismatch" in result.precondition_failure


def test_smoke_drop_cuda_cache_invoked_between_passes() -> None:
    drop_calls: list[int] = []

    result = run_smoke_test(
        settings=_settings(),
        **_kwargs(drop_cache=lambda: drop_calls.append(1)),
    )
    assert result.ok is True
    assert drop_calls == [1]


def test_smoke_internal_error_propagates() -> None:
    """Internal errors (bugs in the runner path) are NOT caught — the
    CLI maps them to exit 1 separately."""

    def faulty_drop() -> None:
        raise AssertionError("invariant broken in runner")

    with pytest.raises(AssertionError, match="invariant broken"):
        run_smoke_test(settings=_settings(), **_kwargs(drop_cache=faulty_drop))


def test_drop_cuda_cache_skips_empty_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=MagicMock(),
    )
    fake_torch.cuda = cuda_ns  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    smoke_mod._drop_cuda_cache()

    cuda_ns.empty_cache.assert_not_called()


def test_drop_cuda_cache_calls_empty_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_calls = MagicMock()
    fake_torch = types.ModuleType("torch")
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=empty_calls,
    )
    fake_torch.cuda = cuda_ns  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    smoke_mod._drop_cuda_cache()

    empty_calls.assert_called_once_with()


def test_write_models_lock_writes_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Fake huggingface_hub.HfApi so the test doesn't touch the network.
    fake_hf = types.ModuleType("huggingface_hub")

    class _FakeApi:
        def model_info(self, repo: str) -> Any:
            return types.SimpleNamespace(sha=f"sha-{repo}")

    fake_hf.HfApi = _FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    lock_path = tmp_path / "MODELS.lock"
    payload = write_models_lock(
        lock_path,
        generation_repo="gen/repo",
        embedding_repo="emb/repo",
    )

    assert lock_path.exists()
    assert payload["generation"] == {"repo": "gen/repo", "commit": "sha-gen/repo"}
    assert payload["embedding"] == {"repo": "emb/repo", "commit": "sha-emb/repo"}
    assert "packages" in payload
    # We don't assert specific versions — just that the dict is populated.
    assert isinstance(payload["packages"], dict)
