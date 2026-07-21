"""The numeric-trust regression lock.

Deterministic tier: every fixture entry is graded through the real
magnitude + grounding code with no OpenAI cost. This test fails the
moment a change stops catching the $8 absurd-floor or $900.84B
cross-source class — that is the whole point of the fixture.
"""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.numeric_trust_eval import (
    default_fixture_path,
    grade_case,
    load_fixture,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,
    )


@pytest.fixture()
def cases() -> list:
    return load_fixture(default_fixture_path(_settings()))


def test_fixture_loads_and_has_the_headline_failures(cases: list) -> None:
    by_id = {c.id: c for c in cases}
    assert "nt01" in by_id and by_id["nt01"].kind == "absurd_floor"
    assert "nt02" in by_id and by_id["nt02"].kind == "cross_source"
    kinds = {c.kind for c in cases}
    assert {"good", "multi_year_ok", "unknown_units"} <= kinds
    assert sum(1 for c in cases if c.kind == "good") >= 3


def test_every_case_grades_as_expected(cases: list) -> None:
    settings = _settings()
    failures = []
    for case in cases:
        result = grade_case(case, settings)
        if not result.passed:
            failures.append((result.case_id, result.reasons))
    assert not failures, f"grading failures: {failures}"


def test_absurd_floor_is_caught(cases: list) -> None:
    nt01 = next(c for c in cases if c.id == "nt01")
    result = grade_case(nt01, _settings())
    assert result.actual_finding == "absurd_floor"
    assert result.actual_disposition == "caveat_or_retry"


def test_cross_source_double_count_is_caught_and_caveated(cases: list) -> None:
    nt02 = next(c for c in cases if c.id == "nt02")
    result = grade_case(nt02, _settings())
    assert result.actual_finding == "cross_source_sum"
    # Caveat, never a wasted retry on a possibly-legitimate sum.
    assert result.actual_disposition == "caveat"


def test_good_totals_raise_no_hard_finding(cases: list) -> None:
    # Precision gate: no correct total may draw a hard caveat/retry.
    settings = _settings()
    for case in (c for c in cases if c.kind == "good"):
        result = grade_case(case, settings)
        assert result.actual_finding is None, case.id
        assert result.actual_disposition == "ship", case.id


def test_multi_year_legit_sum_ships_with_caveat_not_rejected(cases: list) -> None:
    nt06 = next(c for c in cases if c.id == "nt06")
    result = grade_case(nt06, _settings())
    assert result.actual_finding == "cross_source_sum"
    assert result.actual_disposition == "caveat"


def test_unknown_units_soft_caveat(cases: list) -> None:
    nt07 = next(c for c in cases if c.id == "nt07")
    result = grade_case(nt07, _settings())
    assert result.actual_finding == "unknown_units"
    assert result.actual_disposition == "caveat"
