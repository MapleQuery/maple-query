"""Numeric grounding: hold the final answer accountable to what was
actually computed.

Two deterministic signals over the captured derivations, no model call:

- **headline grounding** — the largest monetary number in the answer
  must tie back to a captured scalar total (scale-aware), or the answer
  is shipping a figure it did not compute;
- **cross-source sum** — a single scalar SUM that read documents from
  two or more fiscal-year editions of the same series is the invisible
  double-count that turned ~$450B into $900.84B.

This module produces signals and the caveat *templates*; it never alters
an answer itself. The magnitude verify extension is the single
disposition authority that decides when to apply them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from semantic_enrich.core.agent.derivation_units import scale_factor

if TYPE_CHECKING:
    from semantic_enrich.core.agent.derivation import Derivation

GroundingVerdict = Literal["grounded", "ungrounded", "no_numeric_claim"]

# Relative tolerance for matching a prose number to a computed value.
# Wide enough to absorb one-significant-figure prose rounding ("$8" for
# 8.2, "$450B" for 453.2B) — the goal is to catch a *fabricated* figure,
# not to punish an answer for rounding the total it computed — yet far
# too tight to match an unrelated computed total.
_MATCH_TOLERANCE = 0.05


@dataclass(frozen=True)
class CrossSourceVerdict:
    flagged: bool
    packages: tuple[str, ...] = ()
    fiscal_years: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingReport:
    grounding: GroundingVerdict
    headline_value: float | None
    matched: bool
    cross_source_sum: CrossSourceVerdict = field(
        default_factory=lambda: CrossSourceVerdict(flagged=False)
    )


# ── number extraction ──

# $-amounts with optional magnitude word/suffix, and bare magnitude
# numbers ("1.2 billion"). Percentages tracked separately.
_MAGNITUDE = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
    "m": 1e6, "million": 1e6, "millions": 1e6, "mm": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "t": 1e12, "trillion": 1e12, "trillions": 1e12,
}
_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9.])                          # not mid-identifier/decimal
    (?P<dollar>\$\s*)?                          # optional leading $
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?        # 1,234,567(.89)
        |\d+(?:\.\d+)?)                         # or 900.84 or 8
    \s*
    (?P<suffix>%|k|thousand|thousands|m|mm|million|millions
        |b|bn|billion|billions|t|trillion|trillions)?
    (?![A-Za-z0-9])                            # suffix/number ends a token
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A standalone 4-digit year (1900-2099) is not a monetary claim.
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True)
class _Extracted:
    monetary: tuple[float, ...]
    percentages: tuple[float, ...]


def extract_numbers(text: str) -> _Extracted:
    """Conservative deterministic extraction. A miss yields fewer
    numbers, never a crash — the fail-open direction is no false
    caveat."""
    monetary: list[float] = []
    percentages: list[float] = []
    for m in _NUMBER_RE.finditer(text or ""):
        raw = m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        suffix = (m.group("suffix") or "").lower()
        if suffix == "%":
            percentages.append(value)
            continue
        if suffix in _MAGNITUDE:
            monetary.append(value * _MAGNITUDE[suffix])
            continue
        # No suffix and no $ sign: skip bare 4-digit years and other
        # non-monetary integers to avoid grounding incidental numbers.
        if m.group("dollar") is None:
            if _YEAR_RE.match(raw):
                continue
            # A comma-grouped number ("1,234,567") reads as a real
            # figure even without a $; a bare small integer usually
            # does not. Keep grouped numbers, drop bare ones.
            if "," not in m.group("num"):
                continue
        monetary.append(value)
    return _Extracted(monetary=tuple(monetary), percentages=tuple(percentages))


# ── headline grounding ──


def match_claim_to_derivation(
    answer_text: str, derivations: list[Derivation]
) -> tuple[GroundingVerdict, float | None, bool]:
    """Match the headline (largest monetary) number in the answer to a
    captured scalar total, scale-aware. Returns
    ``(verdict, headline_value, matched)``."""
    extracted = extract_numbers(answer_text)
    if not extracted.monetary:
        return "no_numeric_claim", None, False
    headline = max(extracted.monetary, key=abs)

    scalar_values = [
        v
        for v in (_normalized_value(d) for d in derivations)
        if v is not None
    ]
    if not scalar_values:
        # Numbers in prose but nothing computed to tie them to.
        return "ungrounded", headline, False

    for value in scalar_values:
        if _within_tolerance(headline, value):
            return "grounded", headline, True
    return "ungrounded", headline, False


def _normalized_value(d: Derivation) -> float | None:
    if d.result_value is None:
        return None
    factor = scale_factor(d.unit_scale)
    # Unknown/count scale: no reliable normalization, so match against
    # the raw computed value (a units mismatch is 7.3's job, not a
    # grounding miss).
    return d.result_value * factor if factor is not None else d.result_value


def _within_tolerance(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= _MATCH_TOLERANCE


# ── cross-source double-count ──

_FISCAL_YEAR_RE = re.compile(r"\b(20\d{2})[-\u2013/](\d{2}|20\d{2})\b")
_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def detect_cross_source_sum(
    derivations: list[Derivation],
) -> CrossSourceVerdict:
    """Flag a scalar SUM that added documents from two or more
    fiscal-year editions of the same series. Coarse, auditable, and a
    caveat trigger — never a reject."""
    for d in derivations:
        if d.aggregation != "SUM" or len(d.source_packages) < 2:
            continue
        years: set[str] = set()
        series_counts: dict[str, int] = {}
        for title in d.dataset_titles:
            for fy in _fiscal_years(title):
                years.add(fy)
            series = _strip_years(title)
            if series:
                series_counts[series] = series_counts.get(series, 0) + 1
        shared_series = any(c >= 2 for c in series_counts.values())
        if len(years) >= 2 or shared_series:
            return CrossSourceVerdict(
                flagged=True,
                packages=d.source_packages,
                fiscal_years=tuple(sorted(years)),
            )
    return CrossSourceVerdict(flagged=False)


def _fiscal_years(title: str) -> set[str]:
    found: set[str] = set()
    for m in _FISCAL_YEAR_RE.finditer(title):
        found.add(f"{m.group(1)}-{m.group(2)}")
    if not found:
        for m in _BARE_YEAR_RE.finditer(title):
            found.add(m.group(1))
    return found


def _strip_years(title: str) -> str:
    stripped = _FISCAL_YEAR_RE.sub(" ", title)
    stripped = _BARE_YEAR_RE.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip().lower()


# ── report + caveat templates ──


def build_grounding_report(
    answer_text: str, derivations: list[Derivation]
) -> GroundingReport:
    grounding, headline, matched = match_claim_to_derivation(
        answer_text, derivations
    )
    return GroundingReport(
        grounding=grounding,
        headline_value=headline,
        matched=matched,
        cross_source_sum=detect_cross_source_sum(derivations),
    )


def compose_cross_source_caveat(*, verdict: CrossSourceVerdict) -> str:
    years = ", ".join(verdict.fiscal_years) or "multiple years"
    n = len(verdict.packages)
    return (
        f"**Heads up:** this total sums {n} datasets spanning {years}; "
        "if those overlap, the figure may double-count. "
    )


def compose_ungrounded_caveat() -> str:
    return (
        "**Unverified figure:** I couldn't tie the number(s) in this "
        "answer to a specific computed total, so treat them with "
        "caution. "
    )
