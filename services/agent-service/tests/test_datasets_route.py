"""`GET /datasets` and `/datasets/{package_id}/columns`.

Covers:
  - straight scan when `q` is absent (ORDER BY generated_at DESC path).
  - VECTOR_SEARCH path when `q` is present.
  - column listing per package.
  - 404 on unknown package_id.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FIXED_TOKEN, FakeBqClient


def test_list_datasets_scan(client: TestClient, fake_bq: FakeBqClient) -> None:
    fake_bq.queries = [
        (
            "COUNT(*)",
            [{"n": 42}],
        ),
        (
            "ORDER BY generated_at DESC",
            [
                {
                    "package_id": "pkg-a",
                    "summary": "housing spend",
                    "grain": "monthly",
                    "measures": ["amount"],
                    "dimensions": ["province"],
                    "date_range_start": "2019-01",
                    "date_range_end": "2020-12",
                }
            ],
        ),
    ]
    r = client.get(
        "/datasets",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 42
    assert len(body["datasets"]) == 1
    card = body["datasets"][0]
    assert card["package_id"] == "pkg-a"
    assert card["distance"] is None


def test_list_datasets_vector_search(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        ("COUNT(*)", [{"n": 42}]),
        (
            "VECTOR_SEARCH",
            [
                {
                    "package_id": "pkg-b",
                    "summary": "match",
                    "grain": "yearly",
                    "measures": ["gdp"],
                    "dimensions": ["province"],
                    "date_range_start": None,
                    "date_range_end": None,
                    "distance": 0.12,
                }
            ],
        ),
    ]
    r = client.get(
        "/datasets",
        params={"q": "gdp by province"},
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    card = r.json()["datasets"][0]
    assert card["package_id"] == "pkg-b"
    assert card["distance"] == 0.12


def test_list_columns(client: TestClient, fake_bq: FakeBqClient) -> None:
    fake_bq.queries = [
        (
            "FROM `test-project.semantic.columns`",
            [
                {
                    "column_name": "TOT_EXP",
                    "semantic_type": "measure",
                    "description": "total expenditure",
                    "sample_values": ["100", "200"],
                }
            ],
        ),
    ]
    r = client.get(
        "/datasets/pkg-a/columns",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["package_id"] == "pkg-a"
    assert body["columns"][0]["column_name"] == "TOT_EXP"


def test_list_columns_unknown_package(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        ("FROM `test-project.semantic.columns`", []),
        ("FROM `test-project.semantic.datasets`", []),
    ]
    r = client.get(
        "/datasets/pkg-missing/columns",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "package_not_found"
