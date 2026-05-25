from __future__ import annotations

from ingest.core.dedup import compute_checksum, compute_document_id


def test_compute_checksum_is_sha256_hex() -> None:
    c = compute_checksum(b"hello world")
    assert c == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert len(c) == 64


def test_compute_document_id_is_deterministic() -> None:
    a = compute_document_id(source_url="https://x/y", checksum="a" * 64)
    b = compute_document_id(source_url="https://x/y", checksum="a" * 64)
    assert a == b
    assert len(a) == 64


def test_compute_document_id_varies_by_url() -> None:
    a = compute_document_id(source_url="https://x/y", checksum="a" * 64)
    b = compute_document_id(source_url="https://x/z", checksum="a" * 64)
    assert a != b


def test_compute_document_id_varies_by_checksum() -> None:
    a = compute_document_id(source_url="https://x/y", checksum="a" * 64)
    b = compute_document_id(source_url="https://x/y", checksum="b" * 64)
    assert a != b
