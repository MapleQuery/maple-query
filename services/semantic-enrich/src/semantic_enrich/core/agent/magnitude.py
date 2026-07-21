"""Deterministic magnitude + units plausibility over a captured
derivation.

The fit checker judges whether an answer addresses the question; it
cannot judge whether $8 or $900.84B is a plausible federal total,
because "is 1,400 summed amount rows equal to $8" is arithmetic against
a floor, not a reasoning task — and an LLM verifier already shipped the
$8 answer once. So the bounds are *data*, evaluated here with no model
call, returning at most one finding:

- **A absurd_floor** — a monetary SUM/AVG drawn from many rows whose
  normalized total is below an absolute floor (the $8 class);
- **B over_ceiling** — a monetary total above a coarse sanity ceiling;
- **C cross_source_sum** — a scalar SUM across fiscal-year editions of
  the same series (the $900.84B class), independent of the value's size;
- **D unknown_units / ungrounded_figure** — a monetary total whose scale
  we could not establish, or a headline figure that ties to no computed
  total.

A/B/C are `hard` (the number is very likely wrong or misleading); D is
`soft` (the number may be fine but we cannot vouch for it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from semantic_enrich.core.agent.derivation_units import scale_factor
from semantic_enrich.core.agent.grounding import (
    GroundingReport,
    compose_cross_source_caveat,
    compose_ungrounded_caveat,
)

if TYPE_CHECKING:
    from semantic_enrich.config.settings import Settings
    from semantic_enrich.core.agent.derivation import Derivation

Severity = Literal["hard", "soft"]

# Scales that carry money (so the floor/ceiling apply). `unknown` is
# included on purpose: a monetary column of unresolved scale is exactly
# where a magnitude judgement is unreliable, so it is checked, not
# skipped. `count` / `not_monetary` are excluded.
_MONETARY_SCALES = frozenset({"dollars", "thousands", "millions", "billions", "unknown"})


@dataclass(frozen=True)
class MagnitudeFinding:
    tag: str
    severity: Severity
    detail: str
    caveat: str
    hint: str | None = None
    # Whether this finding may spend a research retry. True only for the
    # "very likely wrong, re-examine the column" class (absurd_floor,
    # over_ceiling): re-running research can fix those. A cross-source
    # sum is *suspicious but possibly legitimate* (a real multi-year
    # total), so it flags for a caveat that surfaces the intent — never
    # a retry that would just discard and reproduce the same sum.
    retry_eligible: bool = False


@dataclass(frozen=True)
class MagnitudeVerdict:
    finding: MagnitudeFinding | None

    @property
    def is_hard(self) -> bool:
        return self.finding is not None and self.finding.severity == "hard"

    @property
    def is_soft(self) -> bool:
        return self.finding is not None and self.finding.severity == "soft"


def evaluate_magnitude(
    derivations: list[Derivation],
    grounding: GroundingReport | None,
    settings: Settings,
) -> MagnitudeVerdict:
    """Return the single highest-severity finding (hard beats soft), or
    an empty verdict. Deterministic; identical inputs -> identical
    output."""
    findings: list[MagnitudeFinding] = []

    for d in derivations:
        if d.result_value is None:
            continue
        findings.extend(_value_checks(d, settings))

    if grounding is not None and grounding.cross_source_sum.flagged:
        findings.append(
            MagnitudeFinding(
                tag="cross_source_sum",
                severity="hard",
                detail=(
                    "summed across "
                    f"{len(grounding.cross_source_sum.packages)} datasets: "
                    f"{', '.join(grounding.cross_source_sum.fiscal_years)}"
                ),
                caveat=compose_cross_source_caveat(
                    verdict=grounding.cross_source_sum
                ),
                hint=(
                    "your total summed multiple fiscal-year datasets of the "
                    "same series; sum a single dataset, or confirm the years "
                    "do not overlap"
                ),
            )
        )
    if grounding is not None and grounding.grounding == "ungrounded":
        findings.append(
            MagnitudeFinding(
                tag="ungrounded_figure",
                severity="soft",
                detail="headline figure ties to no computed total",
                caveat=compose_ungrounded_caveat(),
            )
        )

    # Highest severity wins; hard before soft, first-found within a tier.
    hard = [f for f in findings if f.severity == "hard"]
    if hard:
        return MagnitudeVerdict(finding=hard[0])
    soft = [f for f in findings if f.severity == "soft"]
    if soft:
        return MagnitudeVerdict(finding=soft[0])
    return MagnitudeVerdict(finding=None)


def _value_checks(
    d: Derivation, settings: Settings
) -> list[MagnitudeFinding]:
    if d.unit_scale not in _MONETARY_SCALES or d.result_value is None:
        return []
    findings: list[MagnitudeFinding] = []
    factor = scale_factor(d.unit_scale)
    # Unknown scale cannot be normalized; check the raw value against the
    # floor/ceiling (an unknown-scale monetary column is also flagged
    # soft by check D, so the units caveat still ships).
    normalized = d.result_value * factor if factor is not None else d.result_value

    # A. Absurd floor — keyed on source_row_estimate (input rows), NOT
    # the scalar aggregate's row_count, which is always 1.
    if (
        d.aggregation in ("SUM", "AVG")
        and d.source_row_estimate >= settings.agent_mag_floor_min_rows
        and abs(normalized) < settings.agent_mag_absurd_floor
    ):
        findings.append(
            MagnitudeFinding(
                tag="absurd_floor",
                severity="hard",
                detail=(
                    f"total {_fmt(normalized)} over ~{d.source_row_estimate} "
                    "rows is implausibly small"
                ),
                caveat=(
                    f"**Check this figure:** this total ({_fmt(normalized)}) is "
                    f"implausibly small for a sum over ~{d.source_row_estimate} "
                    "rows — it may be summing a change/adjustment column rather "
                    "than an amount. "
                ),
                hint=(
                    f"your total of {_fmt(normalized)} drawn from "
                    f"~{d.source_row_estimate} rows is implausibly small; "
                    "re-examine whether the summed column is an amount or a "
                    "change/adjustment column"
                ),
                retry_eligible=True,
            )
        )

    # B. Coarse ceiling.
    if abs(normalized) > settings.agent_mag_ceiling:
        findings.append(
            MagnitudeFinding(
                tag="over_ceiling",
                severity="hard",
                detail=f"total {_fmt(normalized)} exceeds the sanity ceiling",
                caveat=(
                    f"**Check this figure:** this total ({_fmt(normalized)}) is "
                    "implausibly large — it may be double-counting or summing "
                    "across datasets that should not be combined. "
                ),
                hint=(
                    f"your total of {_fmt(normalized)} is implausibly large; "
                    "check for a double-count or a unit-scale error"
                ),
                retry_eligible=True,
            )
        )

    # D. Unknown units (soft) — only when not already normalized.
    if d.unit_scale == "unknown":
        findings.append(
            MagnitudeFinding(
                tag="unknown_units",
                severity="soft",
                detail="monetary column of unresolved scale",
                caveat=(
                    "**Units unverified:** I couldn't confirm whether this "
                    "column is in dollars, thousands, or millions, so the "
                    "magnitude may be off by a factor of 1,000. "
                ),
            )
        )
    return findings


def _fmt(value: float) -> str:
    """Human dollar figure for a caveat/detail line."""
    a = abs(value)
    if a >= 1e9:
        return f"${value / 1e9:.1f}B"
    if a >= 1e6:
        return f"${value / 1e6:.1f}M"
    if a >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:,.2f}"
