"""Shared test fixtures.

Real model loads only happen on the operator's GPU box during
`smoke-test --write-lock`. CI stubs `outlines.from_transformers` +
the model's `__call__`, and `SentenceTransformer.encode`, so the test
suite stays GPU-free.
"""
from __future__ import annotations

import pytest

from semantic_enrich.config.settings import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A minimal Settings populated from the WHENRICH_ defaults.

    Tests assert against `Settings()` rather than patching individual
    fields so a regression in the default surfaces here too.
    """
    monkeypatch.delenv("WHENRICH_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("WHENRICH_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("WHENRICH_DEVICE", raising=False)
    monkeypatch.delenv("WHENRICH_HF_CACHE_DIR", raising=False)
    return Settings()  # type: ignore[call-arg]
