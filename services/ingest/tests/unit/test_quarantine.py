from __future__ import annotations

from ingest.clients.http import Downloaded
from ingest.core.quarantine import decide


def _dl(body: bytes, *, content_length: str | None = None) -> Downloaded:
    headers: dict[str, str] = {}
    if content_length is not None:
        headers["Content-Length"] = content_length
    return Downloaded(body=body, status=200, headers=headers, elapsed_ms=10)


def test_download_error_quarantines() -> None:
    d = decide(download=None, download_error=RuntimeError("connection reset"), max_size_bytes=1024)
    assert d.quarantine is True
    assert d.reason == "download_failed"
    assert "connection reset" in d.note


def test_zero_byte_body_quarantines_as_download_failed() -> None:
    d = decide(download=_dl(b""), download_error=None, max_size_bytes=1024)
    assert d.quarantine is True
    assert d.reason == "download_failed"
    assert d.note == "zero bytes"


def test_truncated_body_quarantines() -> None:
    d = decide(download=_dl(b"hi", content_length="100"), download_error=None, max_size_bytes=1024)
    assert d.quarantine is True
    assert d.reason == "truncated_body"


def test_oversize_quarantines() -> None:
    d = decide(download=_dl(b"x" * 2048), download_error=None, max_size_bytes=1024)
    assert d.quarantine is True
    assert d.reason == "oversize"


def test_healthy_response_passes_through() -> None:
    d = decide(download=_dl(b"hello", content_length="5"), download_error=None, max_size_bytes=1024)
    assert d.quarantine is False
    assert d.reason is None


def test_content_length_garbage_is_ignored() -> None:
    # Some servers emit non-numeric Content-Length values; don't crash.
    d = decide(download=_dl(b"hello", content_length="lots"), download_error=None, max_size_bytes=1024)
    assert d.quarantine is False


def test_neither_download_nor_error_is_defensively_handled() -> None:
    d = decide(download=None, download_error=None, max_size_bytes=1024)
    assert d.quarantine is True
    assert d.reason == "download_failed"
