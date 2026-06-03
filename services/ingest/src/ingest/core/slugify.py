"""Slugify the last URL path segment into a safe filename.

Deterministic transform — same input always produces the same output:
1. Take the last URL path segment, percent-decoded (UTF-8).
2. Lowercase.
3. NFKD-normalize; strip combining marks (so `é` → `e`).
4. Replace any character not in `[a-z0-9._-]` with `-`.
5. Collapse runs of `-` to a single `-`.
6. Strip leading and trailing `-`, `.`, `_`.
7. If the result is empty, use the literal `file`.
8. Ensure the result ends with `.<fmt>` (no extension when fmt is
   `unknown`).
9. Truncate to 150 characters total, preserving the extension — drop
   from the basename, not the suffix.
10. Final regex check: matches `^[a-z0-9][a-z0-9._-]*$`.
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

    `fmt` of `"unknown"` skips extension enforcement — we don't know
    what to append, so we leave the basename as-is.
    """
    path = urlparse(resource_url).path
    last_segment = path.rsplit("/", 1)[-1]
    s = unquote(last_segment)

    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # Strip a matching extension up front so the step-8 re-apply produces
    # the canonical form. Without this, "$$$.csv" slugifies to "csv.csv"
    # instead of the intuitively-correct "file.csv".
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
