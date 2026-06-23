"""Settings env-var parsing + defaults."""
from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_load.config.settings import Settings


def test_settings_reads_whload_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "my-project")
    s = Settings()  # type: ignore[call-arg]
    assert s.gcp_project_id == "my-project"


def test_settings_falls_back_to_unprefixed_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHLOAD_GCP_PROJECT_ID", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "fallback-project")
    s = Settings()  # type: ignore[call-arg]
    assert s.gcp_project_id == "fallback-project"


def test_settings_default_bq_dataset_and_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    s = Settings()  # type: ignore[call-arg]
    assert s.bq_dataset_raw == "raw"
    assert s.bq_documents_table == "documents"


def test_settings_runlog_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.setenv("WHLOAD_RUNLOG_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("WHLOAD_RUNLOG_GCS_PREFIX", "gs://bucket/runlog/")
    s = Settings()  # type: ignore[call-arg]
    assert s.runlog_local_dir == tmp_path
    assert s.runlog_gcs_prefix == "gs://bucket/runlog/"


def test_settings_run_id_is_uuid_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    a = Settings().run_id  # type: ignore[call-arg]
    b = Settings().run_id  # type: ignore[call-arg]
    assert a != b
    assert len(a) == 36  # canonical uuid4 length


def test_settings_bucket_prefix_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.delenv("WHLOAD_BUCKET_PREFIX", raising=False)
    s = Settings()  # type: ignore[call-arg]
    assert s.bucket_prefix == "gs://maplequery-raw/raw/"


def test_settings_bucket_prefix_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.setenv("WHLOAD_BUCKET_PREFIX", "gs://other-bucket/data/")
    s = Settings()  # type: ignore[call-arg]
    assert s.bucket_prefix == "gs://other-bucket/data/"


def test_settings_bucket_prefix_requires_gs_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.setenv("WHLOAD_BUCKET_PREFIX", "https://example.org/raw/")
    with pytest.raises(ValueError, match="gs://"):
        Settings()  # type: ignore[call-arg]


def test_settings_bucket_prefix_requires_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.setenv("WHLOAD_BUCKET_PREFIX", "gs://bucket/raw")
    with pytest.raises(ValueError, match="end with"):
        Settings()  # type: ignore[call-arg]


def test_settings_bucket_prefix_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHLOAD_GCP_PROJECT_ID", "p")
    monkeypatch.setenv("WHLOAD_BUCKET_PREFIX", "gs:///raw/")
    with pytest.raises(ValueError, match="missing bucket"):
        Settings()  # type: ignore[call-arg]
