"""Deterministic numeric grounding: number extraction, scale-aware
headline matching, and cross-source double-count detection.
"""
from __future__ import annotations

from semantic_enrich.core.agent.derivation import Derivation
from semantic_enrich.core.agent.grounding import (
    build_grounding_report,
    detect_cross_source_sum,
    extract_numbers,
    match_claim_to_derivation,
)


def _deriv(
    *,
    result_value: float | None,
    unit_scale: str = "dollars",
    aggregation: str = "SUM",
    packages: tuple[str, ...] = ("pkg-1",),
    titles: tuple[str, ...] = ("Some Dataset",),
) -> Derivation:
    return Derivation(
        source_packages=packages,
        source_documents=("doc-1",),
        dataset_titles=titles,
        aggregation=aggregation,
        value_columns=("Amount",),
        group_by_columns=(),
        predicate_shape="",
        sql_shape="",
        row_count=1,
        source_row_estimate=1000,
        result_value=result_value,
        result_label="total",
        unit_scale=unit_scale,  # type: ignore[arg-type]
        unit_source="unresolved",
        complete=True,
    )


# ── extraction ──


def test_extract_dollar_amounts_and_magnitudes() -> None:
    ex = extract_numbers("It cost $1,234,567.89 or about 1.2 billion, up 45%.")
    assert 1_234_567.89 in ex.monetary
    assert 1.2e9 in ex.monetary
    assert 45.0 in ex.percentages


def test_extract_900b_suffix() -> None:
    ex = extract_numbers("The total was $900.84B for the year.")
    assert any(abs(v - 900.84e9) < 1 for v in ex.monetary)


def test_bare_year_is_not_monetary() -> None:
    ex = extract_numbers("In 2024 the program ran.")
    assert ex.monetary == ()


def test_bare_small_integer_ignored_but_grouped_kept() -> None:
    assert extract_numbers("there were 8 programs").monetary == ()
    assert 1_400_000 in extract_numbers("summed 1,400,000 rows").monetary


def test_empty_text_no_crash() -> None:
    assert extract_numbers("").monetary == ()


# ── headline matching ──


def test_grounded_scale_aware_thousands() -> None:
    # Prose $900.84B against a thousands-scale 900_840_000 -> matches.
    d = _deriv(result_value=900_840_000, unit_scale="thousands")
    verdict, headline, matched = match_claim_to_derivation(
        "The total was $900.84B.", [d]
    )
    assert verdict == "grounded"
    assert matched is True
    assert headline is not None and abs(headline - 900.84e9) < 1


def test_ungrounded_when_no_derivation_matches() -> None:
    d = _deriv(result_value=2.1e9)
    verdict, _headline, matched = match_claim_to_derivation(
        "Spending was about $8.", [d]
    )
    assert verdict == "ungrounded"
    assert matched is False


def test_no_numeric_claim() -> None:
    d = _deriv(result_value=5.0)
    verdict, headline, _matched = match_claim_to_derivation(
        "I could not find the data.", [d]
    )
    assert verdict == "no_numeric_claim"
    assert headline is None


def test_percentage_only_answer_is_no_monetary_claim() -> None:
    verdict, _h, _m = match_claim_to_derivation(
        "It rose 12% year over year.", [_deriv(result_value=5.0)]
    )
    assert verdict == "no_numeric_claim"


def test_numbers_but_no_derivation_is_ungrounded() -> None:
    verdict, _h, matched = match_claim_to_derivation("It was $1,000,000.", [])
    assert verdict == "ungrounded"
    assert matched is False


# ── cross-source ──


def test_two_fiscal_year_editions_flagged() -> None:
    d = _deriv(
        result_value=9e11,
        packages=("pkg-a", "pkg-b"),
        titles=("Main Estimates 2024-25", "Main Estimates 2025-26"),
    )
    v = detect_cross_source_sum([d])
    assert v.flagged is True
    assert set(v.fiscal_years) == {"2024-25", "2025-26"}
    assert set(v.packages) == {"pkg-a", "pkg-b"}


def test_single_package_sum_not_flagged() -> None:
    d = _deriv(result_value=5e9, packages=("pkg-a",), titles=("Main Estimates 2024-25",))
    assert detect_cross_source_sum([d]).flagged is False


def test_two_different_series_same_year_not_flagged() -> None:
    # Summing housing + fisheries for one year is a legit total.
    d = _deriv(
        result_value=5e9,
        packages=("pkg-a", "pkg-b"),
        titles=("Housing Program 2024-25", "Fisheries Program 2024-25"),
    )
    assert detect_cross_source_sum([d]).flagged is False


def test_non_sum_aggregation_not_flagged() -> None:
    d = _deriv(
        result_value=5e9,
        aggregation="AVG",
        packages=("pkg-a", "pkg-b"),
        titles=("Main Estimates 2024-25", "Main Estimates 2025-26"),
    )
    assert detect_cross_source_sum([d]).flagged is False


# ── report ──


def test_build_report_combines_both_signals() -> None:
    d = _deriv(
        result_value=900_840_000,
        unit_scale="thousands",
        packages=("pkg-a", "pkg-b"),
        titles=("Main Estimates 2024-25", "Main Estimates 2025-26"),
    )
    report = build_grounding_report("The total was $900.84B.", [d])
    assert report.grounding == "grounded"
    assert report.cross_source_sum.flagged is True
