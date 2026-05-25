"""Slugify the last URL path segment into a safe filename.

Implements PRD docs/product-specs/milestone-1/2.1-gcs-storage-layer.md §7.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote, urlparse

_MAX_LEN = 150
_REPLACE_RE = re.compile(r"[^a-z0-9._-]")
_DASH_RUN_RE = re.compile(r"-+")
_FINAL_CHECK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def slugify(*, resource_url: str, fmt: str) -> str:
    """Return a safe filename derived from `resource_url` and the canonical `fmt`.

    `fmt` of `"unknown"` skips extension enforcement (see PRD 2.1 §6).
    """
    path = urlparse(resource_url).path
    last_segment = path.rsplit("/", 1)[-1]
    s = unquote(last_segment)

    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # Strip a matching extension up front so step 8 reapplies it canonically.
    # Without this, "$$$.csv" slugifies to "csv.csv"; PRD §7.2 expects "file.csv".
    if fmt != "unknown" and s.endswith(f".{fmt.lower()}"):
        s = s[: -(len(fmt) + 1)]

    s = _REPLACE_RE.sub("-", s)
    s = _DASH_RUN_RE.sub("-", s)
    s = s.strip("-._")

    if not s:
        s = "file"

    if fmt != "unknown":
        ext = f".{fmt}"
        if not s.endswith(ext):
            s = f"{s}{ext}"

    if len(s) > _MAX_LEN:
        if fmt != "unknown":
            ext = f".{fmt}"
            base = s[: -len(ext)]
            s = base[: _MAX_LEN - len(ext)] + ext
        else:
            s = s[:_MAX_LEN]

    if not _FINAL_CHECK_RE.match(s):
        raise ValueError(f"slugified result {s!r} failed final regex check")

    return s
