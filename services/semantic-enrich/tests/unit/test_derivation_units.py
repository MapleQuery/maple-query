"""The deterministic unit-scale resolver.

Load-bearing rule under test: a monetary column whose scale we cannot
establish resolves to ``unknown`` — never silently to ``dollars``.
"""
from __future__ import annotations

from semantic_enrich.core.agent.derivation_units import (
    resolve_unit_scale,
    scale_factor,
)


def _resolve(
    *,
    column_name: str = "",
    semantic_type: str | None = None,
    description: str | None = None,
    aggregation: str = "SUM",
) -> tuple[str, str]:
    return resolve_unit_scale(
        column_name=column_name,
        semantic_type=semantic_type,
        description=description,
        aggregation=aggregation,
    )


def test_count_aggregation_is_count() -> None:
    scale, source = _resolve(column_name="anything", aggregation="COUNT")
    assert scale == "count"
    assert source == "aggregation_is_count"


def test_non_monetary_column() -> None:
    scale, _source = _resolve(
        column_name="province_name",
        semantic_type="categorical",
        description="Name of the province.",
    )
    assert scale == "not_monetary"


def test_explicit_thousands_in_description_wins() -> None:
    scale, source = _resolve(
        column_name="authorities",
        semantic_type="currency",
        description="Total authorities available for use, in thousands of dollars.",
    )
    assert scale == "thousands"
    assert source == "column_description"


def test_millions_and_billions_phrases() -> None:
    assert _resolve(
        column_name="amt", description="reported in millions of dollars"
    )[0] == "millions"
    assert _resolve(
        column_name="amt", description="figures in billions of dollars"
    )[0] == "billions"


def test_scale_cue_in_column_name() -> None:
    scale, source = _resolve(
        column_name="amount_000",
        semantic_type="currency",
        description="An amount.",
    )
    assert scale == "thousands"
    assert source == "column_name"


def test_monetary_but_scale_ambiguous_is_unknown_not_dollars() -> None:
    # A clearly monetary column with no scale cue must NOT be assumed
    # dollars — that is the 1000x error class this guards.
    scale, source = _resolve(
        column_name="change_in_authorities",
        semantic_type="currency",
        description="Change in authorities from the prior period.",
    )
    assert scale == "unknown"
    assert source == "unresolved"


def test_dollar_cue_in_name_makes_it_monetary() -> None:
    # No semantic_type, but a $ / amount cue is enough to treat it as
    # money (and, absent a scale cue, unknown scale).
    scale, _ = _resolve(column_name="total_expenditure", semantic_type=None)
    assert scale == "unknown"


def test_scale_factor_only_for_known_monetary_scales() -> None:
    assert scale_factor("dollars") == 1.0
    assert scale_factor("thousands") == 1_000.0
    assert scale_factor("millions") == 1_000_000.0
    assert scale_factor("billions") == 1_000_000_000.0
    # count / unknown / not_monetary must not be silently normalized.
    assert scale_factor("unknown") is None
    assert scale_factor("count") is None
    assert scale_factor("not_monetary") is None
