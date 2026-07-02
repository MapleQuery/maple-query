"""`POST /sql/run` — guardrails identical to the loop's `run_sql` tool.

Asserts:
  - a valid `SELECT 1` passes guard + executes and returns `status: ok`.
  - a DDL statement is rejected by the guard (200 with
    `status: guard_rejected` — DDL never touches BQ, per PRD §10).
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from semantic_enrich.clients.bq import BoundedQueryResult

from tests.conftest import FIXED_TOKEN, FakeBqClient


def test_sql_run_ok(client: TestClient, fake_bq: FakeBqClient) -> None:
    fake_bq.dry_run_return = 1_000
    fake_bq.bounded_return = BoundedQueryResult(
        rows=[{"n": 1}],
        total_bytes_billed=500,
        slot_ms=10,
        elapsed_ms=42,
        timed_out=False,
        error=None,
    )
    r = client.post(
        "/sql/run",
        json={
            "sql": (
                "SELECT r.document_id FROM `test-project.raw.rows` r "
                "WHERE r.document_id IN ('doc-a') LIMIT 5"
            ),
            "rationale": "smoke",
        },
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["row_count"] == 1
    assert body["bytes_billed"] == 500
    assert body["rows"] == [{"n": 1}]


def test_sql_run_rejects_insert(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    r = client.post(
        "/sql/run",
        json={
            "sql": "INSERT INTO `test-project.raw.rows` VALUES ('x')",
            "rationale": "should be rejected",
        },
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "guard_rejected"
    assert "INSERT" in (body.get("reason") or "")
    # DDL must never reach BQ — no dry-run call should have fired.
    assert fake_bq.dry_run_bytes_calls == []


def test_sql_run_missing_token_rejected(client: TestClient) -> None:
    r = client.post("/sql/run", json={"sql": "SELECT 1"})
    assert r.status_code == 401
