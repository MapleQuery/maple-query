"""Magic-byte format sniff with URL + declared-format fallbacks.

Algorithm:
1. Run puremagic on the first 8 KiB of body. Map MIME / extension hits
   to the canonical fmt list in `CANONICAL_FMTS`.
2. If nothing matched, fall back to the URL's file extension.
3. If still nothing, fall back to the resource's declared format.
4. If still nothing, return "unknown" — recorded for visibility, never
   silently dropped.

When puremagic returns multiple plausible matches (common for zip-based
formats: docx, xlsx, pptx, zip all share `PK\\x03\\x04`), prefer the
candidate that matches `declared_format` or the URL extension before
falling back to puremagic's top guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse

import puremagic

CANONICAL_FMTS: frozenset[str] = frozenset(
    {
        "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml",
        "xlsx", "xls", "ods", "parquet",
        "pdf", "docx", "doc", "rtf", "odt", "txt", "html", "htm",
        "zip", "tar", "gz", "7z", "rar",
        "png", "jpg", "jpeg", "gif", "svg",
    }
)

_MIME_TO_FMT: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/msword": "doc",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/zip": "zip",
    "application/x-tar": "tar",
    "application/gzip": "gz",
    "application/x-7z-compressed": "7z",
    "application/x-rar-compressed": "rar",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/html": "html",
    "text/plain": "txt",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/svg+xml": "svg",
}


@dataclass(frozen=True)
class SniffResult:
    fmt: str
    magic_hit: bool
    mismatch_with_declared: bool
    # True iff `fmt` was determined by magic bytes or URL extension.
    # False when we only had the CKAN-declared format to go on — caller
    # should treat that as low-confidence (e.g. for `-f` gating).
    verified: bool


def sniff_format(*, body: bytes, declared_format: str | None, url: str) -> SniffResult:
    """Determine canonical fmt by magic bytes, then URL suffix, then declared."""
    declared_low = declared_format.lower() if declared_format else None
    url_ext = _extension_from_url(url)

    candidates: list[str] = []
    sample = body[:8192]
    if sample:
        for match in _magic_matches(sample):
            mapped = _MIME_TO_FMT.get(match.mime_type or "")
            if mapped is None and match.extension:
                ext = match.extension.lstrip(".").lower()
                if ext in CANONICAL_FMTS:
                    mapped = ext
            if mapped and mapped not in candidates:
                candidates.append(mapped)

    fmt = "unknown"
    magic_hit = bool(candidates)
    verified = False
    if candidates:
        if declared_low and declared_low in candidates:
            fmt = declared_low
        elif url_ext in candidates:
            fmt = url_ext
        else:
            fmt = candidates[0]
        verified = True

    if fmt == "unknown" and url_ext in CANONICAL_FMTS:
        fmt = url_ext
        verified = True

    if fmt == "unknown" and declared_low and declared_low in CANONICAL_FMTS:
        fmt = declared_low
        # verified stays False — declared format alone is not byte/URL evidence.

    mismatch = (
        declared_low is not None
        and declared_low != fmt
        and fmt != "unknown"
    )
    return SniffResult(
        fmt=fmt, magic_hit=magic_hit, mismatch_with_declared=mismatch, verified=verified
    )


def _magic_matches(sample: bytes) -> list[puremagic.PureMagic]:
    try:
        return puremagic.magic_stream(BytesIO(sample))
    except puremagic.PureError:
        return []


def _extension_from_url(url: str) -> str:
    path = urlparse(url).path
    if "." in path:
        return path.rsplit(".", 1)[1].lower()
    return ""
