"""Dedup helpers.

Today, dedup happens at upload time: `clients/gcs.py` does a HEAD before
each write and skips when the existing object's md5 matches the body
about to be uploaded. A re-run still pays the network cost to download,
but no duplicate writes happen.

A future iteration will add cheaper dedup layers (metadata check before
download, HTTP conditional GET with stored ETags, content-hash lookup
before upload) once BigQuery is wired in to track prior ingests.
"""
from __future__ import annotations

import hashlib


def compute_checksum(body: bytes) -> str:
    """sha256 hex of the body. 64 chars."""
    return hashlib.sha256(body).hexdigest()


def compute_document_id(*, source_url: str, checksum: str) -> str:
    """document_id = sha256(source_url || checksum). Hex, 64 chars."""
    h = hashlib.sha256()
    h.update(source_url.encode("utf-8"))
    h.update(checksum.encode("utf-8"))
    return h.hexdigest()
