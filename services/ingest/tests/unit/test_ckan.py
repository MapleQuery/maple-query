from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pytest_httpserver import HTTPServer

from ingest.clients.ckan import CkanClient, CkanError, Dataset, Resource
from ingest.clients.http import HttpClient

# --- fq builder -----------------------------------------------------------

def test_fq_subject_only() -> None:
    fq = CkanClient._build_fq(
        subject="government_and_politics", formats=None, organization=None, since=None
    )
    assert fq == "subject:government_and_politics"


def test_fq_subject_and_single_format_uppercases() -> None:
    fq = CkanClient._build_fq(
        subject="government_and_politics", formats=["csv"], organization=None, since=None
    )
    assert fq == "subject:government_and_politics AND res_format:CSV"


def test_fq_subject_and_multiple_formats_or_clause() -> None:
    fq = CkanClient._build_fq(
        subject="government_and_politics",
        formats=["csv", "xlsx"],
        organization=None,
        since=None,
    )
    assert (
        fq
        == "subject:government_and_politics AND (res_format:CSV OR res_format:XLSX)"
    )


def test_fq_with_organization() -> None:
    fq = CkanClient._build_fq(
        subject="economy", formats=None, organization="fin", since=None
    )
    assert fq == "subject:economy AND organization:fin"


def test_fq_with_since() -> None:
    fq = CkanClient._build_fq(
        subject="economy",
        formats=None,
        organization=None,
        since=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
    )
    assert fq == "subject:economy AND metadata_modified:[2026-05-01T12:00:00Z TO *]"


def test_fq_all_dimensions() -> None:
    fq = CkanClient._build_fq(
        subject="economy",
        formats=["csv", "xlsx"],
        organization="fin",
        since=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert fq == (
        "subject:economy AND (res_format:CSV OR res_format:XLSX) "
        "AND organization:fin "
        "AND metadata_modified:[2026-05-01T00:00:00Z TO *]"
    )


def test_fq_naive_since_treated_as_utc() -> None:
    fq = CkanClient._build_fq(
        subject="economy",
        formats=None,
        organization=None,
        since=datetime(2026, 5, 1, 0, 0),  # no tzinfo
    )
    assert "2026-05-01T00:00:00Z" in fq


# --- response parsing -----------------------------------------------------

# A realistic-ish payload modeled on open.canada.ca's package_search shape.
SAMPLE_PKG = {
    "id": "98abeb62-7c76-4dfb-a134-1551f55ede55",
    "name": "some-dataset-slug",
    "title": "Some Dataset",
    "organization": {"name": "tbs-sct", "title": "Treasury Board"},
    "metadata_modified": "2026-05-25T00:20:45.366217",
    "subject": ["government_and_politics", "information_and_communications"],
    "resources": [
        {
            "id": "1b977865-6f74-4548-b024-d6ca1a6161a3",
            "url": "https://open.canada.ca/data/file.json",
            "name": "Data Schema",
            "format": "JSON",
            "mimetype": None,
            "size": 50079,
            "language": ["en", "fr"],
            "last_modified": "2022-02-18T22:08:05.765183",
        },
    ],
}


def test_dataset_parses_real_shape() -> None:
    d = Dataset.model_validate(SAMPLE_PKG)
    assert d.id == "98abeb62-7c76-4dfb-a134-1551f55ede55"
    assert d.organization_code == "tbs-sct"
    assert d.subjects == ["government_and_politics", "information_and_communications"]
    assert d.metadata_modified.tzinfo is not None
    assert len(d.resources) == 1


def test_resource_languages_as_list() -> None:
    r = Resource.model_validate(SAMPLE_PKG["resources"][0])
    assert r.languages_declared == ["en", "fr"]


def test_resource_languages_as_string_is_coerced() -> None:
    raw = {**SAMPLE_PKG["resources"][0], "language": "en"}
    r = Resource.model_validate(raw)
    assert r.languages_declared == ["en"]


def test_resource_languages_missing_defaults_to_empty() -> None:
    raw = {**SAMPLE_PKG["resources"][0]}
    raw.pop("language")
    r = Resource.model_validate(raw)
    assert r.languages_declared == []


def test_dataset_subject_missing_defaults_to_empty() -> None:
    raw = {**SAMPLE_PKG}
    raw.pop("subject")
    d = Dataset.model_validate(raw)
    assert d.subjects == []


def test_dataset_organization_as_string_also_works() -> None:
    raw = {**SAMPLE_PKG, "organization": "fin"}
    d = Dataset.model_validate(raw)
    assert d.organization_code == "fin"


# --- search() pagination --------------------------------------------------

@pytest.fixture
def http_client() -> HttpClient:
    c = HttpClient(user_agent="test/1.0", request_timeout_seconds=5.0)
    yield c
    c.close()


def _ckan_payload(*, results: list[dict], count: int) -> dict:
    return {"success": True, "result": {"count": count, "results": results}}


def test_search_yields_datasets_from_single_page(
    http_client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/package_search").respond_with_json(
        _ckan_payload(results=[SAMPLE_PKG], count=1)
    )
    ckan = CkanClient(
        http=http_client,
        api_base=httpserver.url_for(""),
        inter_request_delay_seconds=0,
    )
    datasets = list(ckan.search(subject="government_and_politics"))
    assert len(datasets) == 1
    assert datasets[0].organization_code == "tbs-sct"


def test_search_paginates_until_partial_page(
    http_client: HttpClient, httpserver: HTTPServer
) -> None:
    # 3 total, page_size 2 ⇒ page 1 has 2 results, page 2 has 1.
    httpserver.expect_ordered_request("/package_search").respond_with_json(
        _ckan_payload(results=[SAMPLE_PKG, SAMPLE_PKG], count=3)
    )
    httpserver.expect_ordered_request("/package_search").respond_with_json(
        _ckan_payload(results=[SAMPLE_PKG], count=3)
    )

    ckan = CkanClient(
        http=http_client,
        api_base=httpserver.url_for(""),
        inter_request_delay_seconds=0,
    )
    datasets = list(ckan.search(subject="government_and_politics", page_size=2))

    assert len(datasets) == 3
    assert len(httpserver.log) == 2
    # Second call should have start=2
    second_request, _ = httpserver.log[1]
    assert second_request.args["start"] == "2"


def test_search_stops_when_start_meets_count(
    http_client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/package_search").respond_with_json(
        _ckan_payload(results=[SAMPLE_PKG, SAMPLE_PKG], count=2)
    )

    ckan = CkanClient(
        http=http_client,
        api_base=httpserver.url_for(""),
        inter_request_delay_seconds=0,
    )
    datasets = list(ckan.search(subject="x", page_size=2))

    assert len(datasets) == 2
    assert len(httpserver.log) == 1


def test_search_raises_on_success_false(
    http_client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/package_search").respond_with_json(
        {"success": False, "error": {"message": "boom"}}
    )
    ckan = CkanClient(
        http=http_client,
        api_base=httpserver.url_for(""),
        inter_request_delay_seconds=0,
    )
    with pytest.raises(CkanError):
        list(ckan.search(subject="x"))


def test_search_sends_built_fq(
    http_client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/package_search").respond_with_json(
        _ckan_payload(results=[], count=0)
    )
    ckan = CkanClient(
        http=http_client,
        api_base=httpserver.url_for(""),
        inter_request_delay_seconds=0,
    )
    list(
        ckan.search(
            subject="government_and_politics",
            formats=["csv", "xlsx"],
            organization="fin",
        )
    )
    last_request, _ = httpserver.log[-1]
    assert last_request.args["fq"] == (
        "subject:government_and_politics "
        "AND (res_format:CSV OR res_format:XLSX) "
        "AND organization:fin"
    )
