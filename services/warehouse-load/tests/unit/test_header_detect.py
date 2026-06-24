"""Header-detection state machine.

Covers the canonical shapes called out by PRD §6.2 + §12.1.

Each test builds a small in-memory row buffer (list of `list[str | None]`)
and feeds it to `detect_header`. The body slice is sized large enough
(>= body_min_run rows) to satisfy the stability check.
"""
from __future__ import annotations

from typing import Any

from warehouse_load.core.header_detect import detect_header

# Small body_min_run for speed; the algorithm doesn't change.
# `Any`-typed values: mypy doesn't unify int+float across the kwargs
# under strict mode; this is a test harness, not a public API.
_DEFAULTS: dict[str, Any] = {
    "body_min_run": 5,
    "header_lookback": 3,
    "body_modal_match_ratio": 0.8,
    "header_max_cell_chars": 200,
}


def _body_row(i: int) -> list[str | None]:
    """Build a body row that satisfies the type-mix signal:
    one int, one float, one text."""
    return [str(i), f"{i}.5", f"label_{i}"]


def _body_block(n: int) -> list[list[str | None]]:
    return [_body_row(i) for i in range(n)]


def test_row_zero_header_no_preamble() -> None:
    rows: list[list[str | None]] = [["year", "value", "name"], *_body_block(6)]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.body_start_index == 1
    assert h.confidence == "single"
    assert h.keys == ("year", "value", "name")
    assert h.preamble_rows == ()


def test_preamble_then_single_header() -> None:
    # Second preamble row contains a bare integer cell — fails
    # `_is_header_shaped`, so the multi-row branch can't fire and the
    # algorithm settles on a single header.
    rows: list[list[str | None]] = [
        ["Title: Some Dataset", None, None],
        ["12345", None, None],
        ["year", "value", "name"],
        *_body_block(6),
    ]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.body_start_index == 3
    assert h.confidence == "single"
    assert h.keys == ("year", "value", "name")
    assert len(h.preamble_rows) == 2


def test_multi_row_header_with_forward_fill() -> None:
    # Upper row: ["Population", "", "Income", ""] (merged cell semantics)
    # Lower row: ["a", "b", "x", "y"]
    body: list[list[str | None]] = [["1", "2.5", "3", "label"]] * 6
    rows: list[list[str | None]] = [
        ["Population", "", "Income", ""],
        ["a", "b", "x", "y"],
        *body,
    ]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.confidence == "multi_row"
    assert h.body_start_index == 2
    assert h.keys == ("Population__a", "Population__b", "Income__x", "Income__y")


def test_blank_header_cell_injects_synthetic_col() -> None:
    rows: list[list[str | None]] = [["year", "", "name"], *_body_block(6)]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.confidence == "single"
    assert h.keys == ("year", "__col_1", "name")


def test_low_confidence_fallback_on_random_garbage() -> None:
    # No stable run in the lookahead; everything is short random text
    # with no type-mix signal.
    rows: list[list[str | None]] = [
        ["abc", "def"],
        ["ghi", "jkl"],
        ["mno", "pqr"],
        ["stu", "vwx"],
    ]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.confidence == "low"


def test_bilingual_header_preserves_case() -> None:
    rows: list[list[str | None]] = [["EN_YEAR", "FR_ANNEE", "value"], *_body_block(6)]
    h = detect_header(iter(rows), **_DEFAULTS)
    assert h.keys == ("EN_YEAR", "FR_ANNEE", "value")


def test_long_header_cell_disqualifies_as_header() -> None:
    """A 'header' candidate row with a >200-char cell is body-like —
    we treat it as low-confidence rather than a header."""
    long_cell = "x" * 500
    rows: list[list[str | None]] = [[long_cell, "value"], *_body_block(6)]
    h = detect_header(iter(rows), **_DEFAULTS)
    # The "header" row isn't header-shaped; the body still starts at
    # row 1 thanks to the type-mix signal in subsequent rows, but
    # confidence drops to low and keys get synthesised.
    assert h.confidence == "low"


def test_empty_input_returns_low_confidence() -> None:
    h = detect_header(iter([]), **_DEFAULTS)
    assert h.confidence == "low"
    assert h.body_start_index == 0
