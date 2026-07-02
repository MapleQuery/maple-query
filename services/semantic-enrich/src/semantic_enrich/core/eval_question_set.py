"""Load + validate the committed retrieval-validation question fixture.

The fixture is a hand-curated YAML at
`services/semantic-enrich/eval/questions.yaml`. It is the load-bearing
regression-test surface for the semantic layer — same fixture across
runs makes reports diffable; a schema failure aborts the run before any
per-question work.

`safe_load` only. The fixture is committed and reviewed, but keeping
`safe_load` closes the door on any pyyaml-object escalation from an
adversarial edit.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

_ID_RE = re.compile(r"^q\d{2}$")
_ALLOWED_DOMAINS = frozenset({
    "housing", "taxes", "fisheries", "immigration", "census",
    "environment", "methodology", "multi_domain",
})


@dataclass(frozen=True)
class EvalQuestion:
    """One row of the fixture, validated at load time.

    `expected_packages` / `expected_columns` seed recall@k; empty lists
    contribute recall=1.0 (vacuous, documented in the report header).
    `must_return_rows` gates the `answered` terminal state.
    """

    id: str
    question: str
    domain: str
    expected_packages: tuple[str, ...]
    expected_columns: tuple[str, ...]
    must_return_rows: bool
    notes: str


class QuestionSetError(RuntimeError):
    """Fixture load or schema failure. Terminal for the run."""


def load_question_set(path: Path) -> list[EvalQuestion]:
    """Read, parse, and validate `questions.yaml`.

    Returns questions in fixture order — the runner iterates them
    sequentially so re-runs against the same warehouse produce byte-
    identical grades modulo run_id and timestamps.
    """
    if not path.exists():
        raise QuestionSetError(f"eval questions fixture missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise QuestionSetError(
            f"eval questions fixture must be a YAML list, got {type(raw).__name__}"
        )
    if not raw:
        raise QuestionSetError("eval questions fixture is empty")

    seen_ids: set[str] = set()
    questions: list[EvalQuestion] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise QuestionSetError(
                f"eval questions[{i}] must be a mapping, got {type(entry).__name__}"
            )
        question = _validate_entry(entry, index=i)
        if question.id in seen_ids:
            raise QuestionSetError(
                f"duplicate question id {question.id!r} at index {i}"
            )
        seen_ids.add(question.id)
        questions.append(question)
    return questions


def _validate_entry(entry: dict[str, object], *, index: int) -> EvalQuestion:
    prefix = f"eval questions[{index}]"

    qid = entry.get("id")
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        raise QuestionSetError(f"{prefix}.id must match /^q\\d{{2}}$/, got {qid!r}")

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        raise QuestionSetError(f"{prefix}.question must be a non-empty string")

    domain = entry.get("domain")
    if not isinstance(domain, str) or domain not in _ALLOWED_DOMAINS:
        raise QuestionSetError(
            f"{prefix}.domain must be one of {sorted(_ALLOWED_DOMAINS)}, "
            f"got {domain!r}"
        )

    expected_packages_raw = entry.get("expected_packages", [])
    if not isinstance(expected_packages_raw, list):
        raise QuestionSetError(f"{prefix}.expected_packages must be a list")
    expected_packages: list[str] = []
    for pkg in expected_packages_raw:
        if not isinstance(pkg, str):
            raise QuestionSetError(
                f"{prefix}.expected_packages entries must be strings, got {pkg!r}"
            )
        _require_uuid(pkg, context=f"{prefix}.expected_packages")
        expected_packages.append(pkg)

    expected_columns_raw = entry.get("expected_columns", [])
    if not isinstance(expected_columns_raw, list):
        raise QuestionSetError(f"{prefix}.expected_columns must be a list")
    expected_columns: list[str] = []
    for col in expected_columns_raw:
        if not isinstance(col, str) or not col.strip():
            raise QuestionSetError(
                f"{prefix}.expected_columns entries must be non-empty strings"
            )
        expected_columns.append(col)

    must_return_rows = entry.get("must_return_rows")
    if not isinstance(must_return_rows, bool):
        raise QuestionSetError(f"{prefix}.must_return_rows must be a bool")

    notes = entry.get("notes", "") or ""
    if not isinstance(notes, str):
        raise QuestionSetError(f"{prefix}.notes must be a string")

    return EvalQuestion(
        id=qid,
        question=question.strip(),
        domain=domain,
        expected_packages=tuple(expected_packages),
        expected_columns=tuple(expected_columns),
        must_return_rows=must_return_rows,
        notes=notes,
    )


def _require_uuid(value: str, *, context: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise QuestionSetError(
            f"{context} entry {value!r} is not a valid UUID"
        ) from exc
