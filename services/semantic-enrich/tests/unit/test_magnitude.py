"""Deterministic magnitude + units checks over a captured derivation."""
from __future__ import annotations

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.derivation import Derivation
from semantic_enrich.core.agent.grounding import (
    CrossSourceVerdict,
    GroundingReport,
)
from semantic_enrich.core.agent.magnitude import evaluate_magnitude


def _settings(**overrides: object) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,
    )


def _deriv(
    *,
    result_value: float | None,
    unit_scale: str = "dollars",
    aggregation: str = "SUM",
    source_row_estimate: int = 1000,
) -> Derivation:
    return Derivation(
        source_packages=("pkg-1",),
        source_documents=("doc-1",),
        dataset_titles=("Ledger 2020-21",),
        aggregation=aggregation,
        value_columns=("Amount",),
        group_by_columns=(),
        predicate_shape="",
        sql_shape="",
        row_count=1,
        source_row_estimate=source_row_estimate,
        result_value=result_value,
        result_label="total",
        unit_scale=unit_scale,  # type: ignore[arg-type]
        unit_source="unresolved",
        complete=True,
    )


def _grounding(
    *, flagged: bool = False, grounding: str = "grounded"
) -> GroundingReport:
    return GroundingReport(
        grounding=grounding,  # type: ignore[arg-type]
        headline_value=None,
        matched=True,
        cross_source_sum=CrossSourceVerdict(
            flagged=flagged,
            packages=("pkg-a", "pkg-b") if flagged else (),
            fiscal_years=("2024-25", "2025-26") if flagged else (),
        ),
    )


# ── A. absurd floor ──


def test_absurd_floor_fires_on_8_dollars_over_many_rows() -> None:
    d = _deriv(result_value=8.2, unit_scale="unknown", source_row_estimate=1412)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None
    assert v.finding.tag == "absurd_floor"
    assert v.finding.severity == "hard"


def test_absurd_floor_does_not_fire_below_min_rows() -> None:
    d = _deriv(result_value=8.2, unit_scale="unknown", source_row_estimate=3)
    v = evaluate_magnitude([d], None, _settings())
    # No absurd_floor; may still be a soft unknown_units finding.
    assert v.finding is None or v.finding.tag == "unknown_units"
    assert not (v.finding and v.finding.tag == "absurd_floor")


def test_scalar_row_count_of_one_does_not_defeat_the_floor() -> None:
    # row_count is 1 (scalar aggregate); the gate must key on
    # source_row_estimate, not row_count.
    d = _deriv(result_value=8.2, unit_scale="dollars", source_row_estimate=1412)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None and v.finding.tag == "absurd_floor"


def test_thousands_scale_normalized_before_floor() -> None:
    # 5 (thousands) = $5,000 normalized -> above the $1,000 floor.
    d = _deriv(result_value=5.0, unit_scale="thousands", source_row_estimate=1000)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is None


# ── B. ceiling ──


def test_over_ceiling_fires() -> None:
    d = _deriv(result_value=4e12, unit_scale="dollars")
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None and v.finding.tag == "over_ceiling"
    assert v.finding.severity == "hard"


def test_ceiling_respects_scale() -> None:
    # 4e6 millions = 4e12 dollars -> over ceiling.
    d = _deriv(result_value=4e6, unit_scale="millions")
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None and v.finding.tag == "over_ceiling"


# ── C. cross-source ──


def test_cross_source_from_grounding_is_hard() -> None:
    d = _deriv(result_value=9e11, unit_scale="dollars", source_row_estimate=100)
    v = evaluate_magnitude([d], _grounding(flagged=True), _settings())
    assert v.finding is not None and v.finding.tag == "cross_source_sum"
    assert v.finding.severity == "hard"


def test_cross_source_independent_of_value_size() -> None:
    # A double-count that lands inside floor/ceiling still flags.
    d = _deriv(result_value=5e8, unit_scale="dollars", source_row_estimate=100)
    v = evaluate_magnitude([d], _grounding(flagged=True), _settings())
    assert v.finding is not None and v.finding.tag == "cross_source_sum"


# ── D. unknown units / ungrounded ──


def test_unknown_units_is_soft() -> None:
    d = _deriv(result_value=5e8, unit_scale="unknown", source_row_estimate=10)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None and v.finding.tag == "unknown_units"
    assert v.finding.severity == "soft"


def test_ungrounded_is_soft() -> None:
    d = _deriv(result_value=5e8, unit_scale="dollars")
    v = evaluate_magnitude([d], _grounding(grounding="ungrounded"), _settings())
    assert v.finding is not None and v.finding.tag == "ungrounded_figure"
    assert v.finding.severity == "soft"


# ── severity ordering / no-op ──


def test_hard_beats_soft() -> None:
    # unknown scale + absurd floor: the hard floor finding must win.
    d = _deriv(result_value=8.2, unit_scale="unknown", source_row_estimate=1412)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is not None and v.finding.severity == "hard"


def test_clean_total_no_finding() -> None:
    d = _deriv(result_value=4.5e11, unit_scale="dollars", source_row_estimate=1000)
    v = evaluate_magnitude([d], _grounding(), _settings())
    assert v.finding is None


def test_count_aggregation_not_bounded() -> None:
    d = _deriv(result_value=8.0, unit_scale="count", aggregation="COUNT", source_row_estimate=9999)
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is None


def test_no_scalar_value_no_finding() -> None:
    d = _deriv(result_value=None, unit_scale="dollars")
    v = evaluate_magnitude([d], None, _settings())
    assert v.finding is None
