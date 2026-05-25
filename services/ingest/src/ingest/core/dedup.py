"""Dedup helpers.

Phase A1 scope: we use GCS's write-time existence check (`if_generation_match=0`
with md5 comparison in `clients/gcs.py`) as the dedup mechanism — a re-run
hits the network but `IdempotentSkip` prevents duplicate writes.

Layer 1 (metadata) and Layer 2 (conditional GET) are stubs here so the
pipeline can call them; both will land in A2 once we have BQ to track
prior ingests. Layer 3 (content-hash against BQ) is also A2.
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
