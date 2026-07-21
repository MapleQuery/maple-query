"""Numeric-trust regression grader.

The two real failures (the $8 absurd-floor sum and the $900.84B
cross-source double-count) plus known-good totals become a standing
regression surface. The **deterministic tier** here feeds each entry's
`recorded_derivation` straight into the real magnitude/grounding code
and asserts the finding + disposition — zero OpenAI cost, runs in CI,
and fails the moment a change stops catching either headline class.

The live tier (replaying questions through the loop against the real
warehouse to prove capture reproduces the recorded derivation) is an
opt-in operator step run at act-flip time; it is documented in the PRD
and gated on `WHENRICH_RUN_LIVE_EVALS`, not part of this module's CI
surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.derivation import Derivation
from semantic_enrich.core.agent.grounding import (
    GroundingReport,
    detect_cross_source_sum,
)
from semantic_enrich.core.agent.magnitude import (
    MagnitudeVerdict,
    evaluate_magnitude,
)

_ID_PREFIX = "nt"
_KINDS = frozenset(
    {"absurd_floor", "cross_source", "good", "multi_year_ok", "unknown_units"}
)
_DISPOSITIONS = frozenset({"ship", "caveat", "caveat_or_retry"})


class FixtureError(RuntimeError):
    """Fixture load or schema failure. Terminal for the run."""


@dataclass(frozen=True)
class NumericTrustCase:
    id: str
    question: str
    kind: str
    expect: dict[str, Any]
    recorded_derivation: dict[str, Any]
    notes: str


@dataclass(frozen=True)
class GradeResult:
    case_id: str
    passed: bool
    expected_finding: str | None
    actual_finding: str | None
    expected_disposition: str
    actual_disposition: str
    reasons: tuple[str, ...]


def load_fixture(path: Path) -> list[NumericTrustCase]:
    """Read, parse, and validate the fixture. safe_load only."""
    if not path.exists():
        raise FixtureError(f"numeric-trust fixture missing: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise FixtureError(f"fixture is not valid YAML: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise FixtureError("fixture must be a non-empty list")
    cases = [_case_from_row(i, row) for i, row in enumerate(raw)]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise FixtureError("duplicate case ids in fixture")
    return cases


def _case_from_row(index: int, row: Any) -> NumericTrustCase:
    if not isinstance(row, dict):
        raise FixtureError(f"entry {index} is not a mapping")
    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id.startswith(_ID_PREFIX):
        raise FixtureError(f"entry {index}: id must be a string like 'nt01'")
    kind = row.get("kind")
    if kind not in _KINDS:
        raise FixtureError(f"{case_id}: kind must be one of {sorted(_KINDS)}")
    expect = row.get("expect")
    if not isinstance(expect, dict):
        raise FixtureError(f"{case_id}: expect must be a mapping")
    for key in ("derivation_captured", "result_value_present"):
        if not isinstance(expect.get(key), bool):
            raise FixtureError(f"{case_id}: expect.{key} must be a bool")
    if expect.get("disposition") not in _DISPOSITIONS:
        raise FixtureError(
            f"{case_id}: expect.disposition must be one of {sorted(_DISPOSITIONS)}"
        )
    finding = expect.get("magnitude_finding")
    if finding is not None and not isinstance(finding, str):
        raise FixtureError(f"{case_id}: expect.magnitude_finding must be a string or null")
    recorded = row.get("recorded_derivation")
    if not isinstance(recorded, dict):
        raise FixtureError(f"{case_id}: recorded_derivation must be a mapping")
    if not isinstance(row.get("question"), str):
        raise FixtureError(f"{case_id}: question must be a string")
    return NumericTrustCase(
        id=case_id,
        question=row["question"],
        kind=kind,
        expect=expect,
        recorded_derivation=recorded,
        notes=str(row.get("notes", "")),
    )


def derivation_from_recorded(rec: dict[str, Any]) -> Derivation:
    """Build a Derivation from a fixture's recorded fields, defaulting
    the parts the deterministic tier does not exercise."""
    packages = tuple(str(p) for p in rec.get("source_packages", []))
    titles = tuple(str(t) for t in rec.get("dataset_titles", []))
    return Derivation(
        source_packages=packages,
        source_documents=tuple(
            str(d) for d in rec.get("source_documents", [])
        ),
        dataset_titles=titles,
        aggregation=str(rec.get("aggregation", "none")),
        value_columns=tuple(str(c) for c in rec.get("value_columns", [])),
        group_by_columns=(),
        predicate_shape="",
        sql_shape="",
        row_count=int(rec.get("row_count", 1)),
        source_row_estimate=int(rec.get("source_row_estimate", 0)),
        result_value=(
            float(rec["result_value"])
            if rec.get("result_value") is not None
            else None
        ),
        result_label=str(rec.get("result_label", "total")),
        unit_scale=str(rec.get("unit_scale", "unknown")),  # type: ignore[arg-type]
        unit_source="unresolved",
        complete=True,
    )


def _disposition(verdict: MagnitudeVerdict) -> str:
    finding = verdict.finding
    if finding is None:
        return "ship"
    if finding.severity == "hard" and finding.retry_eligible:
        return "caveat_or_retry"
    return "caveat"


def grade_case(case: NumericTrustCase, settings: Settings) -> GradeResult:
    """Feed the recorded derivation through the real magnitude +
    grounding code and compare against the case's expectation."""
    deriv = derivation_from_recorded(case.recorded_derivation)
    grounding = GroundingReport(
        grounding="grounded",
        headline_value=deriv.result_value,
        matched=True,
        cross_source_sum=detect_cross_source_sum([deriv]),
    )
    verdict = evaluate_magnitude([deriv], grounding, settings)
    actual_finding = verdict.finding.tag if verdict.finding else None
    actual_disposition = _disposition(verdict)

    reasons: list[str] = []
    if bool(deriv.complete) != bool(case.expect["derivation_captured"]):
        reasons.append("derivation_captured mismatch")
    if (deriv.result_value is not None) != bool(
        case.expect["result_value_present"]
    ):
        reasons.append("result_value_present mismatch")
    if actual_finding != case.expect.get("magnitude_finding"):
        reasons.append(
            f"finding {actual_finding!r} != {case.expect.get('magnitude_finding')!r}"
        )
    if actual_disposition != case.expect["disposition"]:
        reasons.append(
            f"disposition {actual_disposition!r} != {case.expect['disposition']!r}"
        )
    return GradeResult(
        case_id=case.id,
        passed=not reasons,
        expected_finding=case.expect.get("magnitude_finding"),
        actual_finding=actual_finding,
        expected_disposition=case.expect["disposition"],
        actual_disposition=actual_disposition,
        reasons=tuple(reasons),
    )


def default_fixture_path(settings: Settings) -> Path:
    """The committed fixture beside the other eval question sets."""
    return (
        settings.agent_verify_prompt_path.parents[3]
        / "eval"
        / "questions-numeric-trust.yaml"
    )
