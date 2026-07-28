"""Recover the real column names of a document whose header was missed.

A large share of the corpus arrives with columns called `__col_1`,
`__col_2`, ... The cause is not a broken parser in the ordinary sense:
Canadian federal statistical releases routinely ship spreadsheets with a
preamble above the header row — a title, a blank line, sometimes a
footnote marker — and the CSV reader takes the first line as the header.
That line is mostly blank, so the columns get positional names and the
*real* header survives as an ordinary data row.

The data is not broken; only the naming is. This module is the pure,
deterministic half of the fix: given the first rows of such a document,
find the row that is actually its header and read the names off it.

Nothing here does I/O, and nothing here calls a model. Column names have
to be reproducible and auditable — the same rows must always yield the
same names — and a heuristic over other people's spreadsheets needs to be
re-tunable against a fixture rather than against the warehouse.

The bias throughout is toward **declining**. A wrong name is strictly
worse than no name: with `__col_3` the model knows it does not understand
the column and says so, but with a wrong name it believes it does, writes
SQL against it, and produces a number that is wrong for a reason nobody
can see from the answer. So a candidate row must clear five independent
checks, each of which vetoes on its own, and anything unclear returns
nothing at all.

Rows are passed in as parsed row bodies in ascending `row_index` order
starting at row 0; the returned `header_row_index` indexes that sequence.
Recovery is per document — package-level and document-level column sets
diverge, so a name recovered here belongs to the document it came from.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Rows scanned for a header. The observed preambles run 1-4 deep; 8
# leaves room without turning a bounded read into a scan.
HEADER_SCAN_ROWS = 8
# Share of positions a candidate row must fill. A real header occupies
# most columns; a preamble title occupies one cell.
HEADER_MIN_DENSITY = 0.6

# `__col_<n>` — the positional name the loader synthesises when a header
# cell is empty (or when it finds no header at all). This is the one
# definition of what "unnamed" means; `generated_header_ratio` and the
# detector both key off it.
GENERATED_COL_RE = re.compile(r"__col_\d+")


def generated_header_ratio(columns: tuple[str, ...] | list[str]) -> float:
    """Share of `columns` that carry a synthesised positional name."""
    if not columns:
        return 0.0
    generated = sum(
        1 for c in columns if GENERATED_COL_RE.fullmatch(c) is not None
    )
    return generated / len(columns)


# ── cell classification ──

# Digits with optional currency, thousands separators, decimals, an
# accounting-negative paren, or a trailing percent. Deliberately broad:
# every one of these forms appears in the corpus as a *value*, and the
# point of the check is to keep value-shaped rows out of the header.
_NUMBER_RE = re.compile(
    r"""^[-+(]?\s*\$?\s*
        \d+(?:[,\u00a0\u202f ]\d{3})*
        (?:\.\d+)?
        \s*\)?\s*%?$""",
    re.VERBOSE,
)
# `2015-09-04`, `2023-06-19T18:32:00Z`, `04/16/2019`. All three appear in
# the corpus. A two-part string like `2019-2020` is deliberately NOT a
# date — fiscal-year ranges are legitimate column labels.
_DATE_RE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ].*)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$"
)
_BARE_YEAR_RE = re.compile(r"^\d{4}$")
# Years are the one numeric form that is routinely a column *name* —
# `2019 | 2020 | 2021` across a fiscal table. The range is wide enough to
# cover historical series and narrow enough that ID-like four-digit codes
# (postal sequences, health-region codes) stay numeric.
_YEAR_MIN = 1800
_YEAR_MAX = 2199

_WHITESPACE_RUN = re.compile(r"\s+")

_EMPTY = "empty"
_NUMBER = "number"
_DATE = "date"
_YEAR = "year"
_TEXT = "text"

# Kinds a header cell may hold. A year qualifies because fiscal tables
# name columns after years; everything else numeric or date-shaped is a
# value, not a name.
_HEADER_KINDS = frozenset({_TEXT, _YEAR})
# Kinds that mark a row as carrying data rather than labels.
_DATA_KINDS = frozenset({_NUMBER, _DATE})


@dataclass(frozen=True)
class HeaderRecovery:
    """The names read off a document's real header row."""

    header_row_index: int
    """Index, within the rows handed to the detector, of the real header."""

    names: Mapping[str, str]
    """`__col_3` -> `Total Amount ($000)`. Only generated columns appear,
    and only those the header row actually named; columns that already
    carry a real name are never touched."""

    preamble_rows: int
    """Rows above the header. Equal to `header_row_index`, named
    separately because that is what a caller trimming the preamble off a
    scoped read is actually asking for."""

    signals: Mapping[str, float]
    """The five gate measurements for the accepted row."""


@dataclass(frozen=True)
class HeaderReport:
    """Detection outcome plus why it came out that way.

    The measurements are kept whether or not detection succeeded, so a
    recovery-rate report can say which gate did the declining — and so a
    threshold change is reviewable against real distributions rather than
    against intuition.
    """

    recovery: HeaderRecovery | None
    reason: str
    signals: Mapping[str, float] = field(default_factory=dict)
    candidate_row_index: int | None = None


def detect_header(
    rows: Sequence[Mapping[str, object]],
    generated_columns: Sequence[str],
    *,
    scan_rows: int = HEADER_SCAN_ROWS,
    min_density: float = HEADER_MIN_DENSITY,
) -> HeaderRecovery | None:
    """Find the real header row and read the generated columns' names off
    it, or return None.

    `rows` are parsed row bodies in ascending `row_index` order beginning
    at row 0. `generated_columns` are the `__col_N` keys of that document.
    """
    return explain_header(
        rows,
        generated_columns,
        scan_rows=scan_rows,
        min_density=min_density,
    ).recovery


def explain_header(
    rows: Sequence[Mapping[str, object]],
    generated_columns: Sequence[str],
    *,
    scan_rows: int = HEADER_SCAN_ROWS,
    min_density: float = HEADER_MIN_DENSITY,
) -> HeaderReport:
    """`detect_header`, plus the measurements and the decline reason."""
    window = list(rows[:scan_rows])
    if not window:
        return HeaderReport(recovery=None, reason="no_rows")
    generated = [c for c in generated_columns if c]
    if not generated:
        return HeaderReport(recovery=None, reason="no_generated_columns")

    keys = _key_universe(window)
    if not keys:
        return HeaderReport(recovery=None, reason="no_rows")
    values = [{k: _clean(row.get(k)) for k in keys} for row in window]
    kinds = [{k: _kind(v) for k, v in row.items()} for row in values]

    body_start = _first_data_row(kinds)
    if body_start is None:
        # Every scanned row is labels or blanks. Either the preamble is
        # deeper than the window or the sheet holds no data at all; both
        # are unverifiable here, and the distinction is worth reporting
        # because only the first is fixable by widening the window.
        return HeaderReport(recovery=None, reason="no_data_rows_in_window")
    if body_start == 0:
        return HeaderReport(recovery=None, reason="data_starts_at_row_0")

    scored = [
        (index, _score(index, values, kinds, keys, body_start))
        for index in range(body_start)
    ]
    qualifying = [
        index
        for index, signals in scored
        if _passes(signals, min_density=min_density)
    ]
    if not qualifying:
        # Report the row that got furthest — the near-miss is what a
        # threshold change would move, and ties break toward the row
        # nearest the data, which is where a header sits.
        index, signals = max(
            scored,
            key=lambda pair: (_gates_passed(pair[1], min_density), pair[0]),
        )
        return HeaderReport(
            recovery=None,
            reason=_first_failing_gate(signals, min_density),
            signals=signals,
            candidate_row_index=index,
        )

    # The header sits immediately above the data, and a preamble may
    # contain a qualifying text row above it — so the last qualifier
    # wins. But two *adjacent* qualifiers are a two-tier header
    # (`2023 | 2024` over `Applicants | Amount`), where either row alone
    # is a wrong name. Decline rather than pick one.
    chosen = qualifying[-1]
    signals = dict(scored[chosen][1])
    if chosen - 1 in qualifying:
        return HeaderReport(
            recovery=None,
            reason="multi_tier",
            signals=signals,
            candidate_row_index=chosen,
        )

    # A merged cell arrives as a label followed by blanks, so a header
    # tier that spans columns leaves gaps exactly where the tier below it
    # carries the real names. If another row above the data fills a
    # column this candidate left blank, the header is split across rows
    # and neither row alone names the columns.
    #
    # Observed: a legal-aid table whose top row reads
    # `... | Criminal | (blank) | ... | Civil | (blank) | (blank)` over a
    # row reading `Adult matter | Youth matter | ... | Family matter`.
    # Taking the top tier named one column "Criminal" when it holds adult
    # matters only, and left youth unnamed — a total that looks complete
    # and is not. Density and distinctness both missed it: the candidate
    # sat exactly on the density threshold, and a merged label produces
    # blanks rather than repeats.
    header = values[chosen]
    unnamed_here = {c for c in generated if c in header and not header[c]}
    if unnamed_here:
        for index in range(body_start):
            if index == chosen:
                continue
            if any(values[index].get(c) for c in unnamed_here):
                return HeaderReport(
                    recovery=None,
                    reason="tier_split",
                    signals=signals,
                    candidate_row_index=chosen,
                )

    names = {
        column: header[column]
        for column in generated
        if column in header and _is_nameable(header[column])
    }
    if not names:
        return HeaderReport(
            recovery=None,
            reason="no_generated_names",
            signals=signals,
            candidate_row_index=chosen,
        )
    return HeaderReport(
        recovery=HeaderRecovery(
            header_row_index=chosen,
            names=names,
            preamble_rows=chosen,
            signals=signals,
        ),
        reason="accepted",
        signals=signals,
        candidate_row_index=chosen,
    )


# ── the five gates ──


def _score(
    index: int,
    values: Sequence[Mapping[str, str]],
    kinds: Sequence[Mapping[str, str]],
    keys: Sequence[str],
    body_start: int,
) -> dict[str, float]:
    """Measure a candidate row against the five signals.

    Each is a ratio rather than a verdict so the thresholds stay
    reviewable against the distribution the corpus actually produces.
    """
    row = kinds[index]
    filled = [k for k in keys if row[k] != _EMPTY]
    label_keys = [k for k in filled if row[k] in _HEADER_KINDS]
    return {
        # Above the data. A header-looking row *inside* the body is a
        # section separator or a repeated banner, not a header.
        "positional": 1.0 if index < body_start else 0.0,
        # Every filled cell is a label. One value-shaped cell means the
        # row is data that happens to be sparse.
        "all_text": (len(label_keys) / len(filled)) if filled else 0.0,
        # A header fills most positions; a preamble title fills one.
        "density": len(filled) / len(keys),
        # Headers name different things. A banner smeared across merged
        # cells repeats itself.
        "distinctness": _distinctness(values[index], filled),
        # The rows below must differ in type: values where the candidate
        # has labels. Without this, a sheet of prose looks like a header.
        "contrast": _contrast(index, kinds, label_keys),
    }


def _passes(signals: Mapping[str, float], *, min_density: float) -> bool:
    return not _first_failing_gate(signals, min_density)


def _first_failing_gate(signals: Mapping[str, float], min_density: float) -> str:
    """The first gate a candidate fails, in the order they are cheapest
    to reason about. Empty string means it cleared all five."""
    if signals.get("positional", 0.0) < 1.0:
        return "positional"
    if signals.get("density", 0.0) < min_density:
        return "density"
    if signals.get("all_text", 0.0) < 1.0:
        return "all_text"
    if signals.get("distinctness", 0.0) < 1.0:
        return "distinctness"
    if signals.get("contrast", 0.0) <= 0.0:
        return "contrast"
    return ""


def _gates_passed(signals: Mapping[str, float], min_density: float) -> int:
    return sum(
        (
            signals.get("positional", 0.0) >= 1.0,
            signals.get("density", 0.0) >= min_density,
            signals.get("all_text", 0.0) >= 1.0,
            signals.get("distinctness", 0.0) >= 1.0,
            signals.get("contrast", 0.0) > 0.0,
        )
    )


def _distinctness(row: Mapping[str, str], filled: Sequence[str]) -> float:
    """Share of the candidate's filled cells that hold a value no other
    filled cell in the row repeats."""
    if not filled:
        return 0.0
    seen = [row[key] for key in filled]
    return len(set(seen)) / len(seen)


def _contrast(
    index: int,
    kinds: Sequence[Mapping[str, str]],
    label_keys: Sequence[str],
) -> float:
    """Share of the candidate's label positions that hold a value below."""
    if not label_keys:
        return 0.0
    contrasting = sum(
        1
        for key in label_keys
        if any(row[key] in _DATA_KINDS for row in kinds[index + 1 :])
    )
    return contrasting / len(label_keys)


# ── helpers ──


def _key_universe(window: Sequence[Mapping[str, object]]) -> list[str]:
    """Every key any scanned row carries.

    Ragged CSV rows can drop trailing keys, and the density denominator
    has to be the document's full column set — a column that is empty
    everywhere is still a position the header failed to fill, and
    counting it is the conservative choice.
    """
    seen: dict[str, None] = {}
    for row in window:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


def _first_data_row(kinds: Sequence[Mapping[str, str]]) -> int | None:
    for index, row in enumerate(kinds):
        if any(kind in _DATA_KINDS for kind in row.values()):
            return index
    return None


def _clean(value: object) -> str:
    """Render a cell as the name it would be read as.

    Whitespace runs collapse to a single space and the ends are trimmed.
    That is not normalisation in the sense this module rules out — the
    text is not slugified, case is kept, and punctuation like
    `Total Amount ($000)` survives intact. It only removes the wrapping
    that a spreadsheet renders as a line break and that would otherwise
    put a raw newline inside a column name.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = value if isinstance(value, str) else str(value)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _kind(text: str) -> str:
    if not text:
        return _EMPTY
    if _BARE_YEAR_RE.fullmatch(text) and _YEAR_MIN <= int(text) <= _YEAR_MAX:
        return _YEAR
    if _NUMBER_RE.fullmatch(text):
        return _NUMBER
    if _DATE_RE.fullmatch(text):
        return _DATE
    return _TEXT


def _is_nameable(text: str) -> bool:
    """A recovered name has to be worth more than the positional key it
    replaces: non-empty, and not itself a positional key."""
    return bool(text) and GENERATED_COL_RE.fullmatch(text) is None
