"""Deterministic unit-scale resolution for a captured derivation.

The semantic layer carries no structured unit/scale field — only
`column_name`, `semantic_type`, and a free-text `description`. So the
scale of a monetary column (dollars vs thousands vs millions) is
*derived* here from those existing signals, by a fixed precedence with
no model call.

The load-bearing choice: a monetary column whose scale we cannot
establish resolves to ``unknown`` — never silently to ``dollars``. A
silent dollars assumption is exactly the 1000x error class the
numeric-trust work exists to expose, so the honest fallback routes the
figure to a caveat downstream rather than vouching for it.
"""
from __future__ import annotations

import re
from typing import Literal

UnitScale = Literal[
    "dollars",
    "thousands",
    "millions",
    "billions",
    "count",
    "unknown",
    "not_monetary",
]
UnitSource = Literal[
    "column_description",
    "column_name",
    "sample_magnitude",
    "aggregation_is_count",
    "unresolved",
]

# semantic_type substrings that mark a column as monetary.
_MONETARY_TYPES = ("currency", "amount", "monetary", "money", "financial")

# A dollar/amount cue anywhere in a column name or description.
_MONEY_CUE_RE = re.compile(r"\$|\bdollar|\bamount|\bexpenditure|\bspending|\bfunding")

# Explicit scale phrases in a free-text description (highest signal).
_DESC_BILLIONS_RE = re.compile(r"billions?\s+of\s+dollars|\bin\s+billions\b|\$\s*b\b")
_DESC_MILLIONS_RE = re.compile(r"millions?\s+of\s+dollars|\bin\s+millions\b|\$\s*m\b")
_DESC_THOUSANDS_RE = re.compile(
    r"thousands?\s+of\s+dollars|\bin\s+thousands\b|\$\s*000s?\b|\$000"
)

# Scale cues embedded in a column name.
_NAME_THOUSANDS_RE = re.compile(r"_000\b|_?thousands?\b|_k\b")
_NAME_MILLIONS_RE = re.compile(r"_?millions?\b|_mm?\b")
_NAME_BILLIONS_RE = re.compile(r"_?billions?\b|_bn?\b")


def resolve_unit_scale(
    *,
    column_name: str,
    semantic_type: str | None,
    description: str | None,
    aggregation: str,
) -> tuple[UnitScale, UnitSource]:
    """Return ``(scale, source)`` by first-match precedence.

    Precedence (first match wins, fully deterministic):

    1. COUNT aggregation -> ``count``.
    2. Not monetary at all -> ``not_monetary`` (captured, not bounded).
    3. Explicit scale phrase in the description -> that scale.
    4. Scale cue in the column name -> that scale.
    5. Monetary but scale-ambiguous -> ``unknown`` (never ``dollars``).
    """
    if aggregation.upper() == "COUNT":
        return "count", "aggregation_is_count"

    name = (column_name or "").lower()
    desc = (description or "").lower()
    stype = (semantic_type or "").lower()
    # Underscores are token separators in column identifiers; a spaced
    # copy lets word-boundary money cues fire on `total_expenditure`
    # while the raw form still carries `_000`-style scale cues.
    name_spaced = name.replace("_", " ")

    is_monetary = (
        any(t in stype for t in _MONETARY_TYPES)
        or bool(_MONEY_CUE_RE.search(name_spaced))
        or bool(_MONEY_CUE_RE.search(desc))
    )
    if not is_monetary:
        return "not_monetary", "unresolved"

    desc_scale = _scale_from_description(desc)
    if desc_scale is not None:
        return desc_scale, "column_description"

    name_scale = _scale_from_name(name)
    if name_scale is not None:
        return name_scale, "column_name"

    return "unknown", "unresolved"


def _scale_from_description(desc: str) -> UnitScale | None:
    # Billions/millions checked before thousands so "$000" inside a
    # longer phrase doesn't pre-empt a clearer word.
    if _DESC_BILLIONS_RE.search(desc):
        return "billions"
    if _DESC_MILLIONS_RE.search(desc):
        return "millions"
    if _DESC_THOUSANDS_RE.search(desc):
        return "thousands"
    return None


def _scale_from_name(name: str) -> UnitScale | None:
    if _NAME_BILLIONS_RE.search(name):
        return "billions"
    if _NAME_MILLIONS_RE.search(name):
        return "millions"
    if _NAME_THOUSANDS_RE.search(name):
        return "thousands"
    return None


_SCALE_FACTOR: dict[str, float] = {
    "dollars": 1.0,
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
}


def scale_factor(unit_scale: str) -> float | None:
    """Multiplier to normalize a value to base dollars, or ``None`` when
    the scale is not a known monetary scale (``count`` / ``unknown`` /
    ``not_monetary``) and so must not be silently normalized."""
    return _SCALE_FACTOR.get(unit_scale)
