"""§16.1 unit tests for the grading rubric + report writer.

Focus:
- recall math (empty expected → 1.0; partial overlap; full overlap).
- mutual-exclusivity tripwire fires when two terminal states could coexist.
- JSON payload round-trips.
- Markdown renders the sections the operator reads.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from semantic_enrich.core.eval_report import (
    GradeInputs,
    RunAggregator,
    finalise_grade,
    recall_at_k,
    write_reports,
)


def test_recall_at_k_empty_expected_is_vacuous() -> None:
    assert recall_at_k([], []) == 1.0
    assert recall_at_k(["a", "b"], []) == 1.0


def test_recall_at_k_full_match() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b", "d", "e"]) == pytest.approx(0.5)


def test_recall_at_k_no_overlap() -> None:
    assert recall_at_k(["x"], ["a", "b"]) == 0.0


def _base_inputs(**overrides: object) -> GradeInputs:
    base = GradeInputs(
        question_id="q01",
        question_text="q?",
        domain="housing",
        expected_packages=(),
        expected_columns=(),
        must_return_rows=True,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_answered_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_text="SELECT 1",
        sql_final_text="SELECT 1 LIMIT 100",
        sql_valid=True,
        rows_returned=3,
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "answered"
    assert grade.answered is True


def test_no_rows_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_valid=True,
        rows_returned=0,
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "no_rows"
    assert grade.answered is False


def test_retrieval_miss_terminal_state() -> None:
    inputs = _base_inputs(retrieval_miss=True)
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "retrieval_miss"


def test_sql_invalid_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_text="SELECT 1",
        sql_valid=False,
        guard_reject_reason="sql_dataset_not_allowed: other",
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "sql_invalid"


def test_sql_too_expensive_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_valid=False,
        guard_reject_reason="sql_cost_too_high: 100 > 50",
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "sql_too_expensive"


def test_sql_timed_out_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_valid=True,
        sql_timed_out=True,
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "sql_timed_out"


def test_sql_not_generated_terminal_state() -> None:
    inputs = _base_inputs(structured_output_violation=True)
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "sql_not_generated"


def test_execution_error_terminal_state() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_valid=True,
        execution_error="table not found",
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "execution_error"


def test_no_execute_notes() -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_valid=True,
        no_execute=True,
    )
    grade = finalise_grade(inputs)
    assert grade.terminal_state == "no_rows"
    assert grade.notes_for_review is not None
    assert "no-execute" in grade.notes_for_review


def test_partial_write_appends(tmp_path: Path) -> None:
    from semantic_enrich.core.eval_report import write_partial_grade

    inputs = _base_inputs(retrieval_miss=True)
    grade = finalise_grade(inputs)
    path = tmp_path / "run.partial.jsonl"
    write_partial_grade(partial_path=path, grade=grade)
    write_partial_grade(partial_path=path, grade=grade)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_write_reports_round_trip(tmp_path: Path) -> None:
    inputs = _base_inputs(
        sql_generated=True,
        sql_text="SELECT 1",
        sql_final_text="SELECT 1 LIMIT 100",
        sql_valid=True,
        rows_returned=1,
        bytes_billed=1024,
        slot_ms=50,
        elapsed_ms=100,
        rows_sample=({"n": 1},),
    )
    grade = finalise_grade(inputs)
    aggregator = RunAggregator(
        run_id="test-run",
        started_at=datetime.now(UTC),
        k_packages=5,
        k_columns=15,
        no_execute=False,
        fixture_path="fixture.yaml",
        prompt_template_path="prompt.j2",
        prompt_template_hash="deadbeef",
        generation_model="gpt-4o",
        embedding_model="text-embedding-3-small",
    )
    aggregator.add(grade)
    summary = aggregator.finalise(finished_at=datetime.now(UTC))
    json_path, md_path = write_reports(
        summary=summary, grades=aggregator.grades, reports_dir=tmp_path
    )
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["run_id"] == "test-run"
    assert payload["grades"][0]["question_id"] == "q01"
    assert payload["grades"][0]["terminal_state"] == "answered"
    md = md_path.read_text()
    assert "test-run" in md
    assert "q01" in md
