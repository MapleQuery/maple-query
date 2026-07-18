"""CachedSnapshotHash: one provider call per refresh window, timestamp
canonicalization, and invalidation exactly once on change."""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.memory import (
    CachedSnapshotHash,
    canonicalize_timestamp,
    make_snapshot_hash_provider_v2,
)
from tests.integration.conftest import FakeBqClient


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_provider_called_once_per_refresh_window() -> None:
    calls = {"n": 0}

    def provider() -> str:
        calls["n"] += 1
        return "hash-a"

    clock = _Clock()
    cached = CachedSnapshotHash(
        provider=provider, refresh_seconds=3600, clock=clock
    )
    for _ in range(50):
        assert cached() == "hash-a"
    assert calls["n"] == 1
    # Window lapses → exactly one more provider call.
    clock.now = 3601.0
    for _ in range(50):
        assert cached() == "hash-a"
    assert calls["n"] == 2


def test_changed_hash_fires_invalidation_exactly_once() -> None:
    values = iter(["hash-a", "hash-b", "hash-b"])
    invalidations: list[str] = []
    clock = _Clock()
    cached = CachedSnapshotHash(
        provider=lambda: next(values),
        refresh_seconds=100,
        on_change=invalidations.append,
        clock=clock,
    )
    assert cached() == "hash-a"
    assert invalidations == []  # first fetch is not a change
    clock.now = 101.0
    assert cached() == "hash-b"
    assert invalidations == ["hash-b"]
    clock.now = 202.0
    assert cached() == "hash-b"
    assert invalidations == ["hash-b"]  # unchanged → no second firing


def test_canonicalize_equalizes_string_cast_formats() -> None:
    a = canonicalize_timestamp("2026-07-01 12:00:00+00:00")
    b = canonicalize_timestamp("2026-07-01T12:00:00+00:00")
    naive = canonicalize_timestamp("2026-07-01 12:00:00")
    assert a == b == naive
    assert canonicalize_timestamp(None) == "none"
    # Unparseable stays stable rather than raising.
    assert canonicalize_timestamp("not a time") == "not a time"


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def test_provider_v2_hashes_canonicalized_timestamps() -> None:
    def bq_with(ts_datasets: Any, ts_columns: Any) -> FakeBqClient:
        bq = FakeBqClient()
        bq.register_query("semantic", [{"max_ts": ts_datasets}])
        bq.register_query("semantic", [{"max_ts": ts_columns}])
        return bq

    settings = _settings()
    space = make_snapshot_hash_provider_v2(
        bq_with("2026-07-01 12:00:00+00:00", "2026-07-01 13:00:00+00:00"),
        settings,
    )()
    tee = make_snapshot_hash_provider_v2(
        bq_with("2026-07-01T12:00:00+00:00", "2026-07-01T13:00:00+00:00"),
        settings,
    )()
    assert space == tee

    different = make_snapshot_hash_provider_v2(
        bq_with("2026-07-02 12:00:00+00:00", "2026-07-01 13:00:00+00:00"),
        settings,
    )()
    assert different != space


def test_provider_v2_without_project_is_stable() -> None:
    settings = Settings(
        gcp_project_id="",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    provider = make_snapshot_hash_provider_v2(FakeBqClient(), settings)
    assert provider() == "no-snapshot"
