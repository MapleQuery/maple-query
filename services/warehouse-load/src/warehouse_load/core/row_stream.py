"""Stream a CSV file → header-row pre-detection + body-row iterator.

Two entry points:

- `iter_lookahead_rows`: pulls just the first N rows for the header
  state machine. Reads the file twice (once for detection, once for
  the body stream) — cheap because the lookahead is bounded to
  `body_min_run + header_lookback` (~25 rows by default).
- `stream_body_rows`: opens polars' batched reader, skips past the
  preamble + header, and yields per-body-row dicts keyed by the
  normalised header names.

Polars only natively understands utf-8. For any other sniffed
encoding (BOM, latin-1, charset_normalizer pick) we decode the
bytes once and write a utf-8 sibling file, then point polars at
that. The 600 MB doc cap means the doubled disk footprint is
bounded; per-doc temp files are deleted after the streaming pass.

Cell rules (PRD §6.5):

- Empty string `""` → JSON `null`.
- NUL bytes (`\\x00`) are stripped (would otherwise break BQ JSON
  ingestion). One `csv_nul_stripped` event per doc on first sight.
- All other bytes preserved verbatim, including control chars and
  domain-specific sentinels like `n.a.`, `N/A`, `<MDL`.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from warehouse_load.types import HeaderResult, SniffResult


@dataclass(frozen=True)
class StagingRow:
    """One body row ready for the staging JSONL.

    `row_index` is the 0-based index *within the body* (post-preamble,
    post-header). `row` is the JSON object whose keys are the
    normalised header names.
    """

    row_index: int
    row: dict[str, str | None]
    nul_stripped: bool


def iter_lookahead_rows(
    *,
    path: Path,
    sniff: SniffResult,
    max_rows: int,
) -> Iterator[list[str | None]]:
    """Yield up to `max_rows` rows from the front of `path` for the
    header state machine.

    Uses polars with `infer_schema_length=0` so every cell is a string,
    `has_header=False` so the header isn't consumed, `n_rows=max_rows`
    to cap the read.
    """
    df = pl.read_csv(
        path,
        separator=sniff.delimiter,
        encoding=_polars_encoding(sniff.encoding),
        has_header=False,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        n_rows=max_rows,
        # `null_values=[""]` would otherwise replace empty cells with
        # None at this stage; the header state machine wants the
        # original string so its `_effective_column_count` heuristic
        # works. Body streaming applies the null rule.
    )
    for row in df.iter_rows():
        yield [_cell_to_optional_str(c) for c in row]


def stream_body_rows(
    *,
    path: Path,
    sniff: SniffResult,
    header: HeaderResult,
) -> Iterator[StagingRow]:
    """Stream the body of `path` as `StagingRow`s.

    `pl.scan_csv(...).collect_batches()` is the polars 1.x streaming
    path; it returns a generator that produces `DataFrame` chunks
    sized by polars internally (memory-bounded). The deprecated
    `read_csv_batched` had a bug where `skip_rows` landing on a
    short first row would lock the column count too low and silently
    drop trailing cells in subsequent wider rows — `scan_csv` with
    an explicit `schema` pins the column count to `len(header.keys)`,
    avoiding that.
    """
    schema = dict.fromkeys(header.keys, pl.String)
    lf = pl.scan_csv(
        path,
        separator=sniff.delimiter,
        encoding=_polars_encoding(sniff.encoding),
        has_header=False,
        skip_rows=header.body_start_index,
        schema=schema,
        truncate_ragged_lines=True,
        null_values=[""],
    )

    row_index = 0
    keys = header.keys
    for batch in lf.collect_batches():
        for raw_row in batch.iter_rows():
            cleaned, nul_seen = _row_to_dict(raw_row, keys)
            yield StagingRow(row_index=row_index, row=cleaned, nul_stripped=nul_seen)
            row_index += 1


# 1 MiB read chunks. Big enough that decode overhead is amortised,
# small enough that 4 concurrent workers stay well under the 600 MB
# per-doc cap x 4 = 2.4 GB worst case the old full-buffer path could hit.
_UTF8_CONVERT_CHUNK_BYTES = 1 * 1024 * 1024


def prepare_utf8_copy(
    *,
    source_path: Path,
    encoding: str,
    dest_path: Path,
) -> None:
    """Decode `source_path` with `encoding`, write `dest_path` as utf-8.

    Used to feed non-utf-8 files to polars, which only natively
    handles utf-8. Bounded by `max_bytes_per_doc`; the caller manages
    the lifecycle of `dest_path` (typically a `tempfile.NamedTemporaryFile`).

    Latin-1 decodes any byte sequence, so the conversion never fails
    for the §5.2.1 fallback. UTF-8-sig decode strips the BOM.

    Streams in fixed-size chunks via Python text mode so peak memory
    stays bounded per worker — a full-buffer read+decode on a 600 MB
    latin-1 file would otherwise hold ~600 MB bytes + ~1.2 GB str
    simultaneously, which scales badly under `rows_concurrency`.
    """
    with source_path.open("r", encoding=encoding, newline="") as src, \
            dest_path.open("w", encoding="utf-8", newline="") as dst:
        while True:
            chunk = src.read(_UTF8_CONVERT_CHUNK_BYTES)
            if not chunk:
                break
            dst.write(chunk)


def needs_utf8_conversion(encoding: str) -> bool:
    """polars only accepts utf-8. utf-8-sig keeps a BOM that becomes a
    leading char of cell (0,0); easier to strip it via conversion than
    to special-case the parser. Everything else (latin-1, cp1252,
    etc.) also needs conversion.
    """
    return encoding.lower() not in {"utf-8", "utf8"}


def _row_to_dict(
    raw_row: tuple[Any, ...],
    keys: tuple[str, ...],
) -> tuple[dict[str, str | None], bool]:
    """Pair `raw_row` against `keys`, normalising each cell.

    Returns the dict + a flag indicating whether any NUL byte was
    stripped from this row. The caller uses the flag to emit the
    one-per-doc `csv_nul_stripped` event.

    A row shorter than `keys` is padded with `None`; a row longer is
    truncated. Both happen on real-world ragged data.
    """
    nul_seen = False
    out: dict[str, str | None] = {}
    for i, key in enumerate(keys):
        cell = raw_row[i] if i < len(raw_row) else None
        normalised, had_nul = _normalise_cell(cell)
        if had_nul:
            nul_seen = True
        out[key] = normalised
    return out, nul_seen


def _normalise_cell(cell: Any) -> tuple[str | None, bool]:
    """Convert a polars cell into the wire shape.

    polars yields `None` for null cells (post `null_values=[""]`),
    strings for everything else under `infer_schema_length=0`. Numeric
    types shouldn't occur but we coerce defensively so an upstream
    polars surprise doesn't crash the load.

    Returns `(value, had_nul)` — `had_nul` lets the caller emit the
    `csv_nul_stripped` event exactly once per doc.
    """
    if cell is None:
        return None, False
    if not isinstance(cell, str):
        cell = str(cell)
    if "\x00" in cell:
        return cell.replace("\x00", ""), True
    return cell, False


def _cell_to_optional_str(cell: Any) -> str | None:
    """For lookahead: empty string stays empty, None stays None. The
    header state machine treats `""` and `None` equivalently when
    counting effective columns, so the distinction doesn't matter for
    detection — it's preserved only to keep the polars output shape
    intact.
    """
    if cell is None:
        return None
    if not isinstance(cell, str):
        return str(cell)
    return cell


def _polars_encoding(encoding: str) -> Literal["utf8", "utf8-lossy"]:
    """Polars accepts `"utf8"` and `"utf8-lossy"`; we always pre-convert
    non-utf8 files via `prepare_utf8_copy`, so this only ever sees
    utf-8 in normal operation. Defensive fallback to utf8-lossy keeps
    a misrouted non-utf8 read from raising.
    """
    if encoding.lower() in {"utf-8", "utf8"}:
        return "utf8"
    return "utf8-lossy"
