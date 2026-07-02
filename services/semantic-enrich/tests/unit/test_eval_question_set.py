"""§16.1 unit tests for the question-set loader.

The committed fixture must parse cleanly; hand-crafted invalid fixtures
raise `QuestionSetError` with a message that names the bad field.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from semantic_enrich.core.eval_question_set import (
    QuestionSetError,
    load_question_set,
)

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _SERVICE_ROOT / "eval" / "questions.yaml"


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_real_fixture_loads_clean() -> None:
    questions = load_question_set(_FIXTURE)
    assert len(questions) == 20
    assert {q.id for q in questions} == {f"q{i:02d}" for i in range(1, 21)}
    for q in questions:
        assert q.question
        assert q.domain
        assert isinstance(q.must_return_rows, bool)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="missing"):
        load_question_set(tmp_path / "nope.yaml")


def test_non_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"not": "a list"})
    with pytest.raises(QuestionSetError, match="must be a YAML list"):
        load_question_set(path)


def test_empty_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    with pytest.raises(QuestionSetError, match="empty"):
        load_question_set(path)


def test_bad_id_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "not-a-q-id",
                "question": "q?",
                "domain": "housing",
                "expected_packages": [],
                "expected_columns": [],
                "must_return_rows": True,
            }
        ],
    )
    with pytest.raises(QuestionSetError, match=r"id must match"):
        load_question_set(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    entry = {
        "id": "q01",
        "question": "q?",
        "domain": "housing",
        "expected_packages": [],
        "expected_columns": [],
        "must_return_rows": True,
    }
    path = _write(tmp_path, [entry, dict(entry)])
    with pytest.raises(QuestionSetError, match="duplicate"):
        load_question_set(path)


def test_bad_domain_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "q01",
                "question": "q?",
                "domain": "not_a_domain",
                "expected_packages": [],
                "expected_columns": [],
                "must_return_rows": True,
            }
        ],
    )
    with pytest.raises(QuestionSetError, match="domain must be one of"):
        load_question_set(path)


def test_bad_uuid_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "q01",
                "question": "q?",
                "domain": "housing",
                "expected_packages": ["not-a-uuid"],
                "expected_columns": [],
                "must_return_rows": True,
            }
        ],
    )
    with pytest.raises(QuestionSetError, match="not a valid UUID"):
        load_question_set(path)


def test_must_return_rows_non_bool_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "q01",
                "question": "q?",
                "domain": "housing",
                "expected_packages": [],
                "expected_columns": [],
                "must_return_rows": "yes",
            }
        ],
    )
    with pytest.raises(QuestionSetError, match="must_return_rows must be a bool"):
        load_question_set(path)
