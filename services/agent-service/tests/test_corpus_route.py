"""`GET /corpus/stats` — landing page counts.

Covers:
  - happy path returns datasets/documents/rows from the fake BQ.
  - a second call inside the TTL is served from the process cache and
    doesn't re-hit BQ.
  - bearer auth is required.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import agent_service.routes.corpus as corpus_mod
from tests.conftest import FIXED_TOKEN, FakeBqClient


def _reset_cache() -> None:
    corpus_mod._cache = None
    corpus_mod._cache_expires_at = 0.0


def test_corpus_stats_returns_counts(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    _reset_cache()
    fake_bq.queries = [
        (
            "COUNT(DISTINCT package_id)",
            [{"datasets": 3742, "documents": 14318, "rows": 42_193_051}],
        )
    ]
    r = client.get(
        "/corpus/stats",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json() == {
        "datasets": 3742,
        "documents": 14318,
        "rows": 42_193_051,
    }


def test_corpus_stats_caches_within_ttl(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    _reset_cache()
    fake_bq.queries = [
        (
            "COUNT(DISTINCT package_id)",
            [{"datasets": 3742, "documents": 14318, "rows": 42_193_051}],
        )
    ]
    headers = {"Authorization": f"Bearer {FIXED_TOKEN}"}
    first = client.get("/corpus/stats", headers=headers)
    assert first.status_code == 200
    executed_after_first = list(fake_bq.executed)

    second = client.get("/corpus/stats", headers=headers)
    assert second.status_code == 200
    assert second.json() == first.json()
    # Cache serves the second call — no new BQ execution.
    assert fake_bq.executed == executed_after_first


def test_corpus_stats_requires_auth(client: TestClient) -> None:
    _reset_cache()
    r = client.get("/corpus/stats")
    assert r.status_code in (401, 403)
