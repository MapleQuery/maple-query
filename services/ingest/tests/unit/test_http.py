from __future__ import annotations

import httpx
import pytest
import respx

from ingest.clients.http import Downloaded, HttpClient, NotModified


@pytest.fixture
def client() -> HttpClient:
    c = HttpClient(
        user_agent="test/1.0",
        request_timeout_seconds=5.0,
        max_retries=3,
    )
    yield c
    c.close()


# --- get_json -------------------------------------------------------------

@respx.mock
def test_get_json_returns_parsed_payload(client: HttpClient) -> None:
    respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(200, json={"success": True, "result": [1, 2, 3]})
    )
    assert client.get_json("https://api.example.com/x") == {
        "success": True,
        "result": [1, 2, 3],
    }


@respx.mock
def test_get_json_retries_on_503_then_succeeds(client: HttpClient) -> None:
    route = respx.get("https://api.example.com/x").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    assert client.get_json("https://api.example.com/x") == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_get_json_raises_on_4xx_without_retry(client: HttpClient) -> None:
    route = respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("https://api.example.com/x")
    assert route.call_count == 1


@respx.mock
def test_get_json_sends_params(client: HttpClient) -> None:
    route = respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(200, json={})
    )
    client.get_json("https://api.example.com/x", params={"fq": "subject:x", "rows": "10"})
    request = route.calls.last.request
    assert request.url.params["fq"] == "subject:x"
    assert request.url.params["rows"] == "10"


# --- download -------------------------------------------------------------

@respx.mock
def test_download_returns_downloaded_on_200(client: HttpClient) -> None:
    body = b"hello world"
    respx.get("https://files.example.com/a.csv").mock(
        return_value=httpx.Response(200, content=body)
    )
    result = client.download("https://files.example.com/a.csv")
    assert isinstance(result, Downloaded)
    assert result.body == body
    assert result.status == 200


@respx.mock
def test_download_returns_not_modified_on_304(client: HttpClient) -> None:
    respx.get("https://files.example.com/a.csv").mock(
        return_value=httpx.Response(304)
    )
    result = client.download(
        "https://files.example.com/a.csv",
        etag='"abc123"',
    )
    assert isinstance(result, NotModified)


@respx.mock
def test_download_sends_conditional_headers(client: HttpClient) -> None:
    route = respx.get("https://files.example.com/a.csv").mock(
        return_value=httpx.Response(304)
    )
    client.download(
        "https://files.example.com/a.csv",
        etag='"abc"',
        last_modified="Mon, 01 Jan 2026 00:00:00 GMT",
    )
    request = route.calls.last.request
    assert request.headers["if-none-match"] == '"abc"'
    assert request.headers["if-modified-since"] == "Mon, 01 Jan 2026 00:00:00 GMT"
