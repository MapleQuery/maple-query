"""Settings env parsing + defaults."""
from __future__ import annotations

import pytest

from semantic_enrich.config.settings import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "WHENRICH_GENERATION_MODEL",
        "WHENRICH_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.generation_model == "Qwen/Qwen2.5-14B-Instruct"
    assert s.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert s.bq_dataset_raw == "raw"
    assert s.bq_dataset_semantic == "semantic"
    assert s.embedding_dim == 1024
    assert s.embedding_batch_size == 64
    assert s.sample_rows_per_package == 10
    assert s.flush_every_n_packages == 500
    # `gcp_project_id` may be populated from a repo-level `.env`; we
    # don't assert its value here — just that the field exists and is
    # either None or a string.
    assert s.gcp_project_id is None or isinstance(s.gcp_project_id, str)


def test_alias_choices_prefers_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHENRICH_GCP_PROJECT_ID wins over the bare GCP_PROJECT_ID."""
    monkeypatch.setenv("GCP_PROJECT_ID", "bare-project")
    monkeypatch.setenv("WHENRICH_GCP_PROJECT_ID", "prefixed-project")
    s = Settings()
    assert s.gcp_project_id == "prefixed-project"


def test_alias_choices_bare_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHENRICH_GCP_PROJECT_ID", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "bare-project")
    s = Settings()
    assert s.gcp_project_id == "bare-project"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHENRICH_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("WHENRICH_FLUSH_EVERY_N_PACKAGES", "100")
    s = Settings()
    assert s.embedding_batch_size == 32
    assert s.flush_every_n_packages == 100
