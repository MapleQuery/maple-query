"""`GET /datasets`, `/datasets/{package_id}/columns`, and
`/datasets/{package_id}/documents`.

Covers:
  - straight scan when `q` is absent (ORDER BY generated_at DESC path).
  - VECTOR_SEARCH path when `q` is present.
  - column listing per package.
  - source-document listing (representative flag, empty list, 404).
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
                    "title": "Housing spending by province",
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
    assert card["title"] == "Housing spending by province"
    assert card["distance"] is None
    # The scan SELECT must ask for title now that the column exists.
    scan_sql = next(s for s in fake_bq.executed if "ORDER BY generated_at" in s)
    assert "title" in scan_sql


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
                    "title": "GDP by province",
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
    assert card["title"] == "GDP by province"
    assert card["distance"] == 0.12


def test_get_dataset_by_id(client: TestClient, fake_bq: FakeBqClient) -> None:
    fake_bq.queries = [
        (
            "WHERE package_id = @pkg LIMIT 1",
            [
                {
                    "package_id": "pkg-a",
                    "title": "Housing spending by province",
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
        "/datasets/pkg-a",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["package_id"] == "pkg-a"
    assert body["title"] == "Housing spending by province"
    assert body["measures"] == ["amount"]
    assert body["dimensions"] == ["province"]


def test_get_dataset_by_id_unknown(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [("WHERE package_id = @pkg LIMIT 1", [])]
    r = client.get(
        "/datasets/pkg-missing",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "package_not_found"


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


def test_list_documents_marks_representative(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        (
            "FROM `test-project.raw.documents`",
            [
                {
                    "document_id": "doc-data",
                    "title": "Prospecting Permits (Data)",
                    "source_url": "https://open.canada.ca/data.csv",
                    "file_format": "csv",
                    "language": "en",
                    "row_count": 12384,
                    "published_date": "2024-11-01",
                },
                {
                    "document_id": "doc-dict",
                    "title": None,
                    "source_url": "https://open.canada.ca/schema.csv",
                    "file_format": "csv",
                    "language": "en",
                    "row_count": 24,
                    "published_date": None,
                },
            ],
        ),
        (
            "SELECT representative_document_id",
            [{"representative_document_id": "doc-data"}],
        ),
    ]
    r = client.get(
        "/datasets/pkg-a/documents",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["package_id"] == "pkg-a"
    assert len(body["documents"]) == 2
    data, dictionary = body["documents"]
    assert data["document_id"] == "doc-data"
    assert data["is_representative"] is True
    assert data["source_url"] == "https://open.canada.ca/data.csv"
    assert data["published_date"] == "2024-11-01"
    assert dictionary["is_representative"] is False
    assert dictionary["title"] is None
    # Loaded-docs filter + biggest-first ordering.
    docs_sql = next(
        s for s in fake_bq.executed if "raw.documents" in s
    )
    assert "load_status = 'loaded'" in docs_sql
    assert "ORDER BY row_count DESC" in docs_sql


def test_list_documents_no_representative_stamped(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        (
            "FROM `test-project.raw.documents`",
            [
                {
                    "document_id": "doc-a",
                    "title": "T",
                    "source_url": "https://open.canada.ca/a.csv",
                    "file_format": "csv",
                    "language": "fr",
                    "row_count": 10,
                    "published_date": None,
                }
            ],
        ),
        ("SELECT representative_document_id", []),
    ]
    r = client.get(
        "/datasets/pkg-a/documents",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["documents"][0]["is_representative"] is False


def test_list_documents_empty_for_known_package(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        ("FROM `test-project.raw.documents`", []),
        # _package_exists peek: the package is enriched, just has no
        # loaded documents (edge case) → 200 with an empty list.
        ("FROM `test-project.semantic.datasets`", [{"1": 1}]),
    ]
    r = client.get(
        "/datasets/pkg-a/documents",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["documents"] == []


def test_list_documents_unknown_package(
    client: TestClient, fake_bq: FakeBqClient
) -> None:
    fake_bq.queries = [
        ("FROM `test-project.raw.documents`", []),
        ("FROM `test-project.semantic.datasets`", []),
    ]
    r = client.get(
        "/datasets/pkg-missing/documents",
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "package_not_found"
