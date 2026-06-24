"""Header-row detection state machine.

The CSV corpus has three header shapes (master M2 §4.2 rule 3):

- Row-0 single header (97% of files).
- Preamble + row-N single header (~3%).
- Multi-row header: one or two header-shaped rows above a stable
  body, often with merged-cell forward-fill semantics.

This module reads a small lookahead buffer of rows and returns a
`HeaderResult` carrying the body start index, the source header
rows, the preamble, the confidence, and the final normalised keys.
It does NOT touch the raw bytes — it operates on already-parsed
rows from `core/row_stream`.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from warehouse_load.core.header_normalise import normalise_keys
from warehouse_load.types import HeaderResult

# Float-ish: optional sign, optional `$`, digits with optional
# thousands/decimal separators. Used as the type-mix signal in the
# body-start check. Conservative: misses a few exotic cases (e.g.
# `1,234.5` with thousands separator) but those still parse as
# digit-bearing under the int test.
_FLOAT_LIKE = re.compile(r"^\s*[-+]?\$?\d+(?:[.,]\d+)+\s*$")
_INT_LIKE = re.compile(r"^\s*[-+]?\d+\s*$")


def detect_header(
    rows: Iterable[list[str | None]],
    *,
    body_min_run: int,
    header_lookback: int,
    body_modal_match_ratio: float,
    header_max_cell_chars: int,
) -> HeaderResult:
    """Find the body start, classify the header above it.

    Reads up to `body_min_run + header_lookback` rows from the front
    of the iterator. All callers stream the rest of the file from a
    fresh polars reader skipped to `body_start_index` (we don't try
    to re-feed buffered rows back through the parser — simpler and
    matches the §7.1 design).
    """
    buffer = _take(rows, body_min_run + header_lookback)
    if not buffer:
        # Empty file: synthesise a single-key shape so downstream code
        # has something to emit. `body_start_index=0` means no rows
        # follow; `row_count=0` will land on `raw.documents`.
        return HeaderResult(
            body_start_index=0,
            header_rows=(),
            preamble_rows=(),
            confidence="low",
            keys=("__col_0",),
        )

    body_start = _find_body_start(
        buffer,
        body_min_run=body_min_run,
        body_modal_match_ratio=body_modal_match_ratio,
        header_max_cell_chars=header_max_cell_chars,
    )

    if body_start is None:
        # No stable run in the lookahead window. Best-guess: row 0 is
        # the header, body starts at row 1. The doc is still loaded;
        # `header_confidence='low'` flags it for curated review.
        header_row = buffer[0] if buffer else []
        keys = normalise_keys([_safe_str(cell) for cell in header_row])
        return HeaderResult(
            body_start_index=1,
            header_rows=(tuple(_safe_str(c) for c in header_row),),
            preamble_rows=(),
            confidence="low",
            keys=tuple(keys),
        )

    if body_start == 0:
        # Body starts at row 0 → file has no header. Synthesise keys
        # from the modal column count of the body slice.
        modal_cols = _modal_column_count(
            buffer[: min(len(buffer), body_min_run)],
        )
        keys = [f"__col_{i}" for i in range(modal_cols)]
        return HeaderResult(
            body_start_index=0,
            header_rows=(),
            preamble_rows=(),
            confidence="low",
            keys=tuple(keys),
        )

    above_index = body_start - 1
    above = buffer[above_index]
    above_above = buffer[above_index - 1] if above_index >= 1 else None

    above_is_header_shaped = _is_header_shaped(above, header_max_cell_chars)
    above_above_is_header_shaped = (
        above_above is not None
        and _is_header_shaped(above_above, header_max_cell_chars)
    )

    if above_above_is_header_shaped and above_is_header_shaped:
        # Two-row concatenated header. Forward-fill the upper row
        # across empty cells so merged-cell exports normalise cleanly.
        assert above_above is not None  # narrowed by `above_above_is_header_shaped`
        upper = _forward_fill(above_above)
        lower = above
        modal_cols = _modal_column_count(
            buffer[body_start : body_start + body_min_run],
        )
        composite = _compose_multi_row_keys(upper, lower, modal_cols)
        keys = normalise_keys(composite)
        preamble = buffer[: above_index - 1]
        return HeaderResult(
            body_start_index=body_start,
            header_rows=(_to_str_tuple(upper), _to_str_tuple(lower)),
            preamble_rows=tuple(_to_str_tuple(r) for r in preamble),
            confidence="multi_row",
            keys=tuple(keys),
        )

    if above_is_header_shaped:
        modal_cols = _modal_column_count(
            buffer[body_start : body_start + body_min_run],
        )
        single = _pad_or_trim(above, modal_cols)
        keys = normalise_keys(single)
        preamble = buffer[:above_index]
        return HeaderResult(
            body_start_index=body_start,
            header_rows=(_to_str_tuple(above),),
            preamble_rows=tuple(_to_str_tuple(r) for r in preamble),
            confidence="single",
            keys=tuple(keys),
        )

    # Body start found, but the row above isn't header-shaped (e.g.
    # the preamble itself runs all the way down to the body with no
    # header line in between). Treat as low-confidence with synthesised
    # keys; the doc is still loaded.
    modal_cols = _modal_column_count(
        buffer[body_start : body_start + body_min_run],
    )
    keys = [f"__col_{i}" for i in range(modal_cols)]
    preamble = buffer[:body_start]
    return HeaderResult(
        body_start_index=body_start,
        header_rows=(),
        preamble_rows=tuple(_to_str_tuple(r) for r in preamble),
        confidence="low",
        keys=tuple(keys),
    )


def _take(rows: Iterable[list[str | None]], n: int) -> list[list[str | None]]:
    """Pull up to `n` rows off the front of `rows`."""
    out: list[list[str | None]] = []
    iterator = iter(rows)
    for _ in range(n):
        try:
            out.append(next(iterator))
        except StopIteration:
            break
    return out


def _find_body_start(
    buffer: list[list[str | None]],
    *,
    body_min_run: int,
    body_modal_match_ratio: float,
    header_max_cell_chars: int,
) -> int | None:
    """Sliding-window scan for the first index `i` such that
    `buffer[i : i + body_min_run]` is body-shaped (stable column count
    + type-mix signal). Returns None if no `i` qualifies.

    A slice shorter than `body_min_run` still counts so long as the
    buffer doesn't have more rows to give — small files (< 25 rows)
    use whatever rows they have.
    """
    if not buffer:
        return None
    for i in range(len(buffer)):
        slice_ = buffer[i : i + body_min_run]
        if len(slice_) < min(body_min_run, len(buffer) - i):
            # Defensive: shouldn't happen given the slice semantics.
            continue
        if not slice_:
            continue
        modal_cols = _modal_column_count(slice_)
        if modal_cols == 0:
            continue
        # The first row of the slice must itself be body-shaped: an
        # all-text header row (or a single-cell preamble line) at the
        # front of an otherwise-body slice would otherwise pass the
        # slice-wide type-mix check on the strength of the body rows
        # behind it. We require the first row to have the modal
        # column count AND to NOT look like a header — that pair
        # rules out both narrow preambles ("Title: foo") and text-
        # only header rows ("year, value, name").
        if _effective_column_count(buffer[i]) != modal_cols:
            continue
        if _is_header_shaped(buffer[i], header_max_cell_chars):
            continue
        if not _passes_stability(slice_, modal_cols, body_modal_match_ratio):
            continue
        if not _has_type_mix(slice_):
            continue
        return i
    return None


def _modal_column_count(slice_: list[list[str | None]]) -> int:
    """Mode of the column counts in the slice, excluding trailing
    all-empty cells from each row. Returns 0 if the slice is empty.
    """
    if not slice_:
        return 0
    counts = [_effective_column_count(row) for row in slice_ if row is not None]
    if not counts:
        return 0
    return Counter(counts).most_common(1)[0][0]


def _effective_column_count(row: list[str | None]) -> int:
    """Width of `row` after trimming trailing all-empty cells. Ragged
    tails are common in CSV exports and would otherwise corrupt the
    modal-count signal.
    """
    last_non_empty = -1
    for i, cell in enumerate(row):
        if cell is not None and cell.strip() != "":
            last_non_empty = i
    return last_non_empty + 1


def _passes_stability(
    slice_: list[list[str | None]],
    modal_cols: int,
    ratio: float,
) -> bool:
    matches = sum(1 for row in slice_ if _effective_column_count(row) == modal_cols)
    return (matches / len(slice_)) >= ratio


def _has_type_mix(slice_: list[list[str | None]]) -> bool:
    """At least one float-like, one int-like, and one non-trivial text
    cell across the slice. Three independent signals; one missing
    fails the check.
    """
    has_float = False
    has_int = False
    has_text = False
    for row in slice_:
        for cell in row:
            if cell is None:
                continue
            value = cell.strip()
            if not value:
                continue
            if _FLOAT_LIKE.match(value):
                has_float = True
            elif _INT_LIKE.match(value):
                has_int = True
            if len(value) > 1 and not _INT_LIKE.match(value) and not _FLOAT_LIKE.match(value):
                has_text = True
            if has_float and has_int and has_text:
                return True
    return False


def _is_header_shaped(row: list[str | None], max_cell_chars: int) -> bool:
    """Every non-empty cell is text-or-empty, no cell parses as a
    number, no cell exceeds `max_cell_chars`. A row that's all-empty
    is NOT header-shaped (would synthesise __col_* below, but that
    isn't a real header signal).
    """
    saw_non_empty = False
    for cell in row:
        if cell is None:
            continue
        stripped = cell.strip()
        if not stripped:
            continue
        saw_non_empty = True
        if len(stripped) > max_cell_chars:
            return False
        if _INT_LIKE.match(stripped) or _FLOAT_LIKE.match(stripped):
            return False
    return saw_non_empty


def _forward_fill(row: list[str | None]) -> list[str | None]:
    """Carry the last non-empty cell forward across blanks. Handles
    merged-cell exports where a category label spans columns.
    """
    out: list[str | None] = []
    last: str | None = None
    for cell in row:
        if cell is None or cell.strip() == "":
            out.append(last)
        else:
            last = cell
            out.append(cell)
    return out


def _compose_multi_row_keys(
    upper: list[str | None],
    lower: list[str | None],
    modal_cols: int,
) -> list[str]:
    """Per-column composition: `upper__lower`, or whichever exists.

    Pads both rows to `modal_cols` before composing — multi-row
    headers in the wild sometimes have a short upper row.
    """
    upper_p = _pad_or_trim(upper, modal_cols)
    lower_p = _pad_or_trim(lower, modal_cols)
    keys: list[str] = []
    for index, (upper_cell, lower_cell) in enumerate(zip(upper_p, lower_p, strict=True)):
        u_clean = upper_cell.strip() if upper_cell else ""
        l_clean = lower_cell.strip() if lower_cell else ""
        if u_clean and l_clean:
            keys.append(f"{u_clean}__{l_clean}")
        elif u_clean:
            keys.append(u_clean)
        elif l_clean:
            keys.append(l_clean)
        else:
            keys.append(f"__col_{index}")
    return keys


def _pad_or_trim(row: list[str | None], width: int) -> list[str]:
    """Pad with empty strings to `width`; trim if longer. Returns
    a list of pure strings (no None) so downstream string ops are
    safe.
    """
    out = [_safe_str(cell) for cell in row[:width]]
    if len(out) < width:
        out.extend([""] * (width - len(out)))
    return out


def _safe_str(cell: str | None) -> str:
    return cell if cell is not None else ""


def _to_str_tuple(row: list[str | None] | None) -> tuple[str, ...]:
    if row is None:
        return ()
    return tuple(_safe_str(c) for c in row)
