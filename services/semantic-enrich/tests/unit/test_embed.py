"""`core.embed.embed_batch` and loader behaviour."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from semantic_enrich.core.embed import embed_batch, load_embedding_model


class _FakeNdarray:
    """Mimics the minimum surface of a numpy ndarray that `embed_batch`
    touches — an iterable of iterables of floats."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __iter__(self) -> object:
        return iter(self._rows)


def test_embed_batch_calls_encode_with_required_kwargs() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = _FakeNdarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )

    result = embed_batch(["text-a", "text-b"], model=fake_model)

    fake_model.encode.assert_called_once_with(
        ["text-a", "text-b"],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_batch_respects_batch_size_override() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = _FakeNdarray([[1.0]])

    embed_batch(["only-one"], model=fake_model, batch_size=32)

    assert fake_model.encode.call_args.kwargs["batch_size"] == 32
    # The L2-normalisation flag is not exposed for override and must
    # always be True — the embedding columns commit to unit vectors.
    assert fake_model.encode.call_args.kwargs["normalize_embeddings"] is True


def test_embed_batch_returns_python_floats() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = _FakeNdarray([[1, 2, 3]])

    result = embed_batch(["x"], model=fake_model)

    assert all(isinstance(component, float) for component in result[0])


def test_load_embedding_model_invokes_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls = MagicMock(return_value="loaded-embedder")
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    result = load_embedding_model(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        cache_dir="/tmp/hf",
    )

    assert result == "loaded-embedder"
    fake_cls.assert_called_once_with(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        cache_folder="/tmp/hf",
    )
