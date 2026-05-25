from __future__ import annotations

from ingest.core.format_sniff import CANONICAL_FMTS, sniff_format

# Minimal magic-byte signatures sufficient for puremagic to identify them.
PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 16
ZIP = b"PK\x03\x04" + b"\x00" * 20
GIF = b"GIF89a" + b"\x00" * 20


def test_pdf_via_magic_bytes() -> None:
    r = sniff_format(body=PDF, declared_format="pdf", url="https://example.gov/x.pdf")
    assert r.fmt == "pdf"
    assert r.magic_hit is True
    assert r.mismatch_with_declared is False


def test_png_via_magic_bytes() -> None:
    r = sniff_format(body=PNG, declared_format=None, url="https://example.gov/x.png")
    assert r.fmt == "png"
    assert r.magic_hit is True


def test_zip_via_magic_bytes() -> None:
    r = sniff_format(body=ZIP, declared_format=None, url="https://example.gov/archive.zip")
    assert r.fmt == "zip"
    assert r.magic_hit is True


def test_url_fallback_when_magic_misses() -> None:
    # Plain CSV-like bytes — puremagic generally won't pin this down.
    body = b"a,b,c\n1,2,3\n"
    r = sniff_format(body=body, declared_format=None, url="https://example.gov/data.csv")
    assert r.fmt == "csv"
    assert r.magic_hit is False


def test_declared_fallback_when_magic_and_url_miss() -> None:
    body = b"a,b,c\n1,2,3\n"
    r = sniff_format(body=body, declared_format="csv", url="https://example.gov/data")
    assert r.fmt == "csv"
    assert r.magic_hit is False


def test_unknown_when_nothing_matches() -> None:
    body = b"\x00\x01\x02\x03random binary"
    r = sniff_format(body=body, declared_format=None, url="https://example.gov/data")
    assert r.fmt == "unknown"
    assert r.magic_hit is False


def test_mismatch_flagged() -> None:
    # PDF body, declared as csv.
    r = sniff_format(body=PDF, declared_format="csv", url="https://example.gov/x.pdf")
    assert r.fmt == "pdf"
    assert r.mismatch_with_declared is True


def test_mismatch_not_flagged_when_fmt_is_unknown() -> None:
    body = b"\x00random"
    r = sniff_format(body=body, declared_format="csv", url="https://example.gov/data")
    # declared was csv; fmt resolves via declared fallback ⇒ csv ⇒ no mismatch.
    assert r.fmt == "csv"
    assert r.mismatch_with_declared is False


def test_declared_format_not_in_canon_list_ignored() -> None:
    body = b"\x00random"
    r = sniff_format(body=body, declared_format="msword-template", url="https://example.gov/data")
    assert r.fmt == "unknown"


def test_empty_body_yields_unknown_unless_url_or_declared_help() -> None:
    r = sniff_format(body=b"", declared_format=None, url="https://example.gov/data")
    assert r.fmt == "unknown"

    r2 = sniff_format(body=b"", declared_format="csv", url="https://example.gov/data")
    assert r2.fmt == "csv"


def test_canonical_fmts_includes_expected_entries() -> None:
    # Belt-and-braces: regression guard if anyone trims the list.
    expected = {"csv", "pdf", "xlsx", "png", "json", "zip", "unknown"} - {"unknown"}
    assert expected.issubset(CANONICAL_FMTS)
