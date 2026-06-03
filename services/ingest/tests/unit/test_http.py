from __future__ import annotations

import pytest
from httpx import HTTPStatusError
from pytest_httpserver import HTTPServer

from ingest.clients.http import Downloaded, HttpClient, NotModified


@pytest.fixture
def client() -> HttpClient:
    c = HttpClient(
        user_agent="test/1.0",
        request_timeout_seconds=5.0,
        max_retries=3,
        download_delay_seconds=0,
    )
    yield c
    c.close()


# --- get_json -------------------------------------------------------------

def test_get_json_returns_parsed_payload(client: HttpClient, httpserver: HTTPServer) -> None:
    httpserver.expect_request("/x").respond_with_json(
        {"success": True, "result": [1, 2, 3]}
    )
    assert client.get_json(httpserver.url_for("/x")) == {
        "success": True,
        "result": [1, 2, 3],
    }


def test_get_json_retries_on_503_then_succeeds(
    client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_ordered_request("/x").respond_with_data("", status=503)
    httpserver.expect_ordered_request("/x").respond_with_json({"ok": True})
    assert client.get_json(httpserver.url_for("/x")) == {"ok": True}
    assert len(httpserver.log) == 2


def test_get_json_raises_on_4xx_without_retry(
    client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/x").respond_with_json(
        {"error": "not found"}, status=404
    )
    with pytest.raises(HTTPStatusError):
        client.get_json(httpserver.url_for("/x"))
    assert len(httpserver.log) == 1


def test_get_json_sends_params(client: HttpClient, httpserver: HTTPServer) -> None:
    httpserver.expect_request(
        "/x",
        query_string={"fq": "subject:x", "rows": "10"},
    ).respond_with_json({})
    client.get_json(
        httpserver.url_for("/x"),
        params={"fq": "subject:x", "rows": "10"},
    )
    assert len(httpserver.log) == 1


# --- download -------------------------------------------------------------

def test_download_returns_downloaded_on_200(
    client: HttpClient, httpserver: HTTPServer
) -> None:
    body = b"hello world"
    httpserver.expect_request("/a.csv").respond_with_data(body, status=200)
    result = client.download(httpserver.url_for("/a.csv"))
    assert isinstance(result, Downloaded)
    assert result.body == body
    assert result.status == 200


def test_download_returns_not_modified_on_304(
    client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request("/a.csv").respond_with_data("", status=304)
    result = client.download(
        httpserver.url_for("/a.csv"),
        etag='"abc123"',
    )
    assert isinstance(result, NotModified)


def test_download_sends_conditional_headers(
    client: HttpClient, httpserver: HTTPServer
) -> None:
    httpserver.expect_request(
        "/a.csv",
        headers={
            "If-None-Match": '"abc"',
            "If-Modified-Since": "Mon, 01 Jan 2026 00:00:00 GMT",
        },
    ).respond_with_data("", status=304)
    client.download(
        httpserver.url_for("/a.csv"),
        etag='"abc"',
        last_modified="Mon, 01 Jan 2026 00:00:00 GMT",
    )
    assert len(httpserver.log) == 1
