"""Quarantine decision per PRD 2.2 §10.

`path_collision` and `unreadable_encoding` are part of the on-disk
schema (defined in `ingest.types.QuarantineReason`) but are NOT emitted
by this module:
- `path_collision` is emitted by the GCS writer when the write-time
  existence check fails (PRD 2.1 §8.4).
- `unreadable_encoding` is reserved for a future encoding-aware step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ingest.clients.http import Downloaded

DecideReason = Literal["download_failed", "oversize", "truncated_body"]


@dataclass(frozen=True)
class QuarantineDecision:
    quarantine: bool
    reason: DecideReason | None
    note: str = ""


def decide(
    *,
    download: Downloaded | None,
    download_error: Exception | None,
    max_size_bytes: int,
) -> QuarantineDecision:
    if download_error is not None:
        return QuarantineDecision(True, "download_failed", str(download_error))

    if download is None:
        # Defensive — the pipeline should always pass exactly one of the two.
        return QuarantineDecision(True, "download_failed", "no download and no error")

    body_len = len(download.body)
    if body_len == 0:
        return QuarantineDecision(True, "download_failed", "zero bytes")

    declared = _header(download.headers, "Content-Length")
    if declared is not None:
        try:
            declared_int = int(declared)
        except ValueError:
            declared_int = None
        if declared_int is not None and body_len < declared_int:
            return QuarantineDecision(
                True, "truncated_body",
                f"received {body_len} bytes, Content-Length declared {declared_int}",
            )

    if body_len > max_size_bytes:
        return QuarantineDecision(
            True, "oversize", f"{body_len} bytes > cap {max_size_bytes}",
        )

    return QuarantineDecision(False, None, "")


def _header(headers: object, name: str) -> str | None:
    # httpx Headers is case-insensitive; plain dicts may not be.
    try:
        return headers[name]  # type: ignore[index]
    except (KeyError, TypeError):
        pass
    if isinstance(headers, dict):
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v
    return None
