"""Match a column name the model asked for against the names a document
actually has.

The model names columns before it has seen them — it writes
`required_columns=["Expenditures", "Department"]` from the words in the
question. Real headers in this corpus are longer and differently spelled:
`2023-24 Expenditures`, `Department, Agency or Crown corporation`. An
exact-equality filter therefore answers "no document has these columns"
for a package where every one of those concepts is present, and the model
reads that as *the data does not exist* and surrenders.

So matching here is deliberately loose, and everything it returns is
**advisory**. It decides only whether a document is worth showing and
what the real names are; it never rewrites SQL, and it never picks a
column on the model's behalf. The exact-name check in `run_sql`'s
doc/column pairing gate is untouched, so a loose match here still cannot
turn into a query against a column that does not exist.

Two spellings of one concept appear across documents of the same package
— `2023-24 Expenditures` with a hyphen, `2023–24 Expenditures` with an
en dash — so normalisation folds dash variants before anything else.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# Hyphen-like characters this corpus uses interchangeably inside fiscal
# years: ASCII hyphen, non-breaking hyphen, figure/en/em dash, minus.
_DASHES = "-‐‑‒–—−"
_DASH_RE = re.compile(f"[{_DASHES}]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_column(name: str) -> str:
    """Casefolded, dash-folded, punctuation-stripped form of a header."""
    folded = unicodedata.normalize("NFKD", name)
    folded = _DASH_RE.sub("-", folded)
    return _NON_ALNUM_RE.sub(" ", folded.casefold()).strip()


def _tokens(name: str) -> list[str]:
    normalized = normalize_column(name)
    return normalized.split() if normalized else []


def match_columns(requested: str, available: Sequence[str]) -> list[str]:
    """Every column in `available` that could be what `requested` means.

    Tried in order of how much is being assumed, and the first tier that
    hits wins — a document with a column named exactly what was asked for
    never also offers its looser cousins:

    1. Exact string equality.
    2. Equality after normalisation (case, dashes, punctuation).
    3. Token containment: every word of the request appears in the
       column's words. `Expenditures` reaches `2023-24 Expenditures`;
       `Department` reaches `Department, Agency or Crown corporation`.

    More than one hit at the same tier is returned in full and left
    unranked. `Expenditures` against a document holding both
    `2023-24 Expenditures` and `2025-26 Main Estimates Total` is a
    genuine ambiguity, and inventing a preference between two fiscal
    years is exactly the kind of confident guess this codebase declines
    to make elsewhere.
    """
    if not requested:
        return []

    exact = [c for c in available if c == requested]
    if exact:
        return exact

    target = normalize_column(requested)
    if not target:
        return []

    normalized = [c for c in available if normalize_column(c) == target]
    if normalized:
        return normalized

    wanted = set(_tokens(requested))
    if not wanted:
        return []
    return [c for c in available if wanted.issubset(set(_tokens(c)))]


def match_column(requested: str, available: Sequence[str]) -> str | None:
    """The single unambiguous match, or None when there is none or many."""
    hits = match_columns(requested, available)
    return hits[0] if len(hits) == 1 else None


def is_exact(requested: str, available: Sequence[str]) -> bool:
    """Whether `requested` is present under that literal name.

    The distinction that decides whether `list_documents` may narrow its
    listing: an exact hit is a fact about the document, a looser one is
    an inference about the question.
    """
    return requested in set(available)
