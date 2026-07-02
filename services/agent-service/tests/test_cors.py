"""CORS preflight covers the allowed origins and blocks the rest."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_allowed_origin_preflight(client: TestClient) -> None:
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_prod_origin_preflight(client: TestClient) -> None:
    r = client.options(
        "/chat",
        headers={
            "Origin": "https://maplequery.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert r.status_code == 200
    assert (
        r.headers["access-control-allow-origin"] == "https://maplequery.vercel.app"
    )


def test_disallowed_origin_omits_allow_header(client: TestClient) -> None:
    """Starlette's CORS middleware silently omits the allow-origin
    header when the origin isn't in the list, which the browser then
    interprets as a preflight failure. Verifying the omission is enough."""
    r = client.options(
        "/chat",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert "access-control-allow-origin" not in r.headers
