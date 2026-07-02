"""`/healthz` always 200; `/readyz` 200 when canaries pass, 503 when any
one fails.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import FakeBqClient, FakeOpenAIClient


def test_healthz_always_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200


def test_readyz_ok(
    client: TestClient,
    fake_bq: FakeBqClient,
    fake_openai: FakeOpenAIClient,
) -> None:
    fake_bq.queries = [("SELECT 1", [{"ok": 1}])]
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["openai"]["ok"] is True
    assert body["checks"]["bq"]["ok"] is True
    assert body["checks"]["semantic_snapshot"]["ok"] is True


def test_readyz_reports_bq_failure(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [("SELECT 1", RuntimeError("bq unreachable"))]
    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["bq"]["ok"] is False


def test_readyz_reports_openai_failure(
    client: TestClient,
    fake_bq: FakeBqClient,
    fake_openai: FakeOpenAIClient,
    monkeypatch: Any,
) -> None:
    fake_bq.queries = [("SELECT 1", [{"ok": 1}])]

    def raising(_: list[str]) -> list[list[float]]:
        raise RuntimeError("openai unreachable")

    monkeypatch.setattr(fake_openai, "embed", raising)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["openai"]["ok"] is False
