"""Agent-traces fixture: lint of the committed file + loader schema."""
from __future__ import annotations

from pathlib import Path

import pytest

from semantic_enrich.core.agent_eval import (
    ALLOWED_OUTCOMES,
    ALLOWED_TRIAGE,
    AgentQuestionSetError,
    load_agent_question_set,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "eval" / "questions-agent-traces.yaml"
)


def test_committed_fixture_parses_and_is_fully_labeled() -> None:
    questions = load_agent_question_set(FIXTURE)

    assert len(questions) == 14
    ids = [q.id for q in questions]
    assert len(set(ids)) == len(ids)
    for q in questions:
        assert q.question
        assert q.expected_triage in ALLOWED_TRIAGE
        assert q.expected_outcome in ALLOWED_OUTCOMES
        assert q.source == "braintrust-export-2026-07-02"
        assert q.observed_outcome  # every entry records the v1 behaviour
    # Every triage class is represented — the labels double as the
    # triage classifier's training/eval set.
    assert {q.expected_triage for q in questions} == set(ALLOWED_TRIAGE)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "fixture.yaml"
    path.write_text(content, encoding="utf-8")
    return path


_VALID_ENTRY = """
- id: t-1
  question: "a question"
  source: test
  expected:
    triage: in_scope
    outcome: answered
"""


def test_loader_accepts_minimal_entry(tmp_path: Path) -> None:
    (question,) = load_agent_question_set(_write(tmp_path, _VALID_ENTRY))
    assert question.id == "t-1"
    assert question.packages_any_of == ()
    assert question.must_caveat is False
    assert question.observed_outcome == ""


def test_loader_rejects_non_list(tmp_path: Path) -> None:
    with pytest.raises(AgentQuestionSetError, match="YAML list"):
        load_agent_question_set(_write(tmp_path, "key: value"))


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(AgentQuestionSetError, match="duplicate"):
        load_agent_question_set(
            _write(tmp_path, _VALID_ENTRY + _VALID_ENTRY)
        )


def test_loader_rejects_missing_question(tmp_path: Path) -> None:
    content = """
- id: t-1
  expected:
    triage: in_scope
    outcome: answered
"""
    with pytest.raises(AgentQuestionSetError, match="question"):
        load_agent_question_set(_write(tmp_path, content))


def test_loader_rejects_unknown_triage(tmp_path: Path) -> None:
    content = """
- id: t-1
  question: "q"
  expected:
    triage: sideways
    outcome: answered
"""
    with pytest.raises(AgentQuestionSetError, match="triage"):
        load_agent_question_set(_write(tmp_path, content))


def test_loader_rejects_unknown_outcome(tmp_path: Path) -> None:
    content = """
- id: t-1
  question: "q"
  expected:
    triage: in_scope
    outcome: shrugged
"""
    with pytest.raises(AgentQuestionSetError, match="outcome"):
        load_agent_question_set(_write(tmp_path, content))


def test_loader_rejects_non_bool_must_caveat(tmp_path: Path) -> None:
    content = """
- id: t-1
  question: "q"
  expected:
    triage: in_scope
    outcome: answered
    must_caveat: "yes"
"""
    with pytest.raises(AgentQuestionSetError, match="must_caveat"):
        load_agent_question_set(_write(tmp_path, content))


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AgentQuestionSetError, match="missing"):
        load_agent_question_set(tmp_path / "nope.yaml")
