"""Bearer-token auth. Every non-health route must reject
missing / malformed / wrong tokens with 401.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FIXED_TOKEN


def test_healthz_bypasses_auth(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_missing_token_rejected(client: TestClient) -> None:
    r = client.post("/chat", json={"conversation_id": "c", "question": "q"})
    assert r.status_code == 401
    assert r.json()["detail"] == "missing_bearer"


def test_malformed_token_rejected(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"conversation_id": "c", "question": "q"},
        headers={"Authorization": "Basic user:pass"},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "malformed_bearer"


def test_wrong_token_rejected(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"conversation_id": "c", "question": "q"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_bearer"


def test_correct_token_accepted(client: TestClient) -> None:
    # /chat returns SSE; a 200 is enough to confirm auth passed. We
    # intentionally don't drain the stream — that's covered elsewhere.
    r = client.post(
        "/chat",
        json={"conversation_id": "c", "question": "q"},
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
