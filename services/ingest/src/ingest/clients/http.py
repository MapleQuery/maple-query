"""httpx wrapper with TLS pinning, conditional GET, rate limiting, and retries.

The retry policy itself lives in `providers/retry.py`.
"""
from __future__ import annotations

import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ingest.providers.retry import RetryableHttpError, http_retry_policy

_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


def _build_ssl_context() -> ssl.SSLContext:
    """Default-secure SSL context, forced to TLS 1.3 minimum.

    Why TLS 1.3 minimum: `open.canada.ca` sits behind an F5 BIG-IP WAF
    that silently stalls on certain TLS 1.2 Client Hello fingerprints
    produced by uv's bundled OpenSSL build. Negotiating directly at 1.3
    skips the 1.2 handshake the WAF is inspecting. Empirically verified
    2026-05-25: same request hangs at 1.2, succeeds in <1s at 1.3.

    All Government of Canada CDNs and modern open-data portals support
    TLS 1.3, so this is safe for our use case. If a future source
    requires TLS 1.2, lower the minimum or scope a separate client to
    that host.

    Last resort if a future WAF blocks Python's TLS fingerprint
    entirely (cipher list, extensions, etc.): swap this client for
    `curl_cffi.Session(impersonate="chrome")` — copies a real browser's
    full fingerprint.
    """
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    return ctx


@dataclass(frozen=True)
class NotModified:
    """Returned when the server replied 304 to a conditional GET."""


@dataclass(frozen=True)
class Downloaded:
    body: bytes
    status: int
    headers: Mapping[str, str]
    elapsed_ms: int


DownloadResult = NotModified | Downloaded


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str,
        request_timeout_seconds: float,
        max_retries: int = 3,
        download_delay_seconds: float = 0.5,
    ) -> None:
        # download_delay_seconds: trace-ca's pacing trick. open.canada.ca's
        # Akamai WAF tarpits clients that drive its rate threshold; sleeping
        # 0.5s before each download keeps us well under it. Set to 0 in
        # tests so the suite stays fast.
        self._client = httpx.Client(
            verify=_build_ssl_context(),
            timeout=httpx.Timeout(
                connect=request_timeout_seconds,
                read=request_timeout_seconds * 5,
                write=request_timeout_seconds,
                pool=request_timeout_seconds,
            ),
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, br",
            },
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._download_delay = download_delay_seconds

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> Any:
        """Lightweight JSON GET with retries. Used by the CKAN client."""
        retrier = http_retry_policy(max_attempts=self._max_retries)
        return retrier(self._do_get_json, url, dict(params or {}))

    def download(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> DownloadResult:
        """Conditional GET. Streams the body fully into memory before returning.

        Caps body at 100 MB — guarded twice: a cheap Content-Length check
        before reading, and a running counter during stream consumption
        for servers that omit the header.
        """
        if self._download_delay > 0:
            time.sleep(self._download_delay)
        retrier = http_retry_policy(max_attempts=self._max_retries)
        return retrier(self._do_download, url, etag, last_modified)

    def _do_get_json(self, url: str, params: dict[str, str]) -> Any:
        response = self._client.get(url, params=params)
        _raise_if_retryable(response)
        response.raise_for_status()
        return response.json()

    def _do_download(
        self,
        url: str,
        etag: str | None,
        last_modified: str | None,
    ) -> DownloadResult:
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        start = time.monotonic()
        with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code == 304:
                return NotModified()
            _raise_if_retryable(response)
            response.raise_for_status()

            declared = response.headers.get("Content-Length")
            if declared and int(declared) > _MAX_DOWNLOAD_BYTES:
                raise OversizedResourceError(url, int(declared))

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise OversizedResourceError(url, total)
                chunks.append(chunk)

            return Downloaded(
                body=b"".join(chunks),
                status=response.status_code,
                headers=dict(response.headers),
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )


class OversizedResourceError(Exception):
    """Raised when a download body exceeds the 100 MB cap."""

    def __init__(self, url: str, observed_bytes: int) -> None:
        super().__init__(
            f"resource exceeds {_MAX_DOWNLOAD_BYTES} byte cap "
            f"(observed={observed_bytes}): {url}"
        )
        self.url = url
        self.observed_bytes = observed_bytes


def _raise_if_retryable(response: httpx.Response) -> None:
    if response.status_code >= 500 or response.status_code == 429:
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        raise RetryableHttpError(response.status_code, retry_after_seconds=retry_after)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form is the other allowed shape; we fall back to backoff.
        return None
