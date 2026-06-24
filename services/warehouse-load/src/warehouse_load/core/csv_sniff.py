"""Delimiter + encoding sniff over the first ~8 KiB of a CSV blob.

The corpus is ~97% comma, ~3% tab. We decide both per-file because
ingest's magic-byte sniff isn't trustworthy for the comma/tab axis
and `raw.documents.file_format` is 3.2-owned (we don't rewrite it on
disagreement; the rows loader logs and uses the sniffed value for
parsing — see PRD §14 decision 8).

Encoding: try UTF-8 strict → UTF-8 with BOM → `charset_normalizer`
→ latin-1 as the always-decodes fallback. Anything beyond UTF-8
bumps a counter for surveillance.
"""
from __future__ import annotations

from typing import Literal

import charset_normalizer

from warehouse_load.types import SniffResult


def sniff_csv(blob_head: bytes) -> SniffResult:
    """Sniff delimiter + encoding from a ~8 KiB head of the file.

    Returns a `SniffResult`. Does NOT decide whether the file has a
    header (that is `header_detect`), does NOT parse the full file,
    does NOT touch the catalog.

    Reads at most `len(blob_head)` bytes; the caller is responsible
    for bounding the slice (`settings.sniff_buffer_bytes`).
    """
    encoding = _detect_encoding(blob_head)
    delimiter = _detect_delimiter(blob_head, encoding=encoding)
    return SniffResult(
        delimiter=delimiter,
        encoding=encoding,
        sniff_bytes=len(blob_head),
    )


_UTF8_BOM = b"\xef\xbb\xbf"


def _detect_encoding(blob_head: bytes) -> str:
    """Pick an encoding for `blob_head`. BOM-first; then UTF-8 strict;
    then `charset_normalizer`; latin-1 always succeeds.

    The BOM check has to come before strict UTF-8 because the BOM
    *is* valid UTF-8 (decodes as U+FEFF). Without this branch order,
    a file with a BOM would report `encoding='utf-8'` and the BOM
    char would leak into the first cell at parse time.
    """
    if blob_head.startswith(_UTF8_BOM):
        return "utf-8-sig"

    try:
        blob_head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    best = charset_normalizer.from_bytes(blob_head).best()
    if best is not None and best.encoding:
        return best.encoding
    # Fall through: latin-1 decodes any byte sequence. Data may be
    # wrong but the load won't crash; the operator sees this via the
    # `csv_encoding_detected` log event the caller emits.
    return "latin-1"


def _detect_delimiter(blob_head: bytes, *, encoding: str) -> Literal[",", "\t"]:
    """Count `,` vs `\\t` in the first newline-terminated line; higher
    count wins. Tie defaults to `,` (logged as `sniff_tie` by the
    caller if it ever happens — empirically never in the corpus).
    """
    try:
        text = blob_head.decode(encoding, errors="replace")
    except LookupError:
        # Defense in depth: an unknown encoding from charset_normalizer
        # is unexpected but shouldn't crash sniffing. latin-1 always
        # works; the caller still sees `encoding=<original>` in
        # `SniffResult` and decides at parse time how to recover.
        text = blob_head.decode("latin-1", errors="replace")

    first_line, _, _ = text.partition("\n")
    comma_count = first_line.count(",")
    tab_count = first_line.count("\t")
    if tab_count > comma_count:
        return "\t"
    # Both zero → single-column file; default to comma (round-trips
    # either way). Tie + non-zero → comma (deliberate; see PRD §5.1).
    return ","
