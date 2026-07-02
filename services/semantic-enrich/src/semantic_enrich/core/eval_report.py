"""Grading rubric + JSON/Markdown report writer.

Every per-question run collapses into exactly one terminal state; the
grader asserts mutual exclusivity as a runtime tripwire, so a logic bug
that would silently ship a wrong grade blows up loudly instead.

Both report files land under `eval_reports_dir/<run_id>.{json,md}`.
`reports/` is `.gitignore`d by default; a deliberate acceptance run is
committed via `git add -f`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

TerminalState = Literal[
    "answered",
    "no_rows",
    "sql_invalid",
    "sql_too_expensive",
    "sql_timed_out",
    "retrieval_miss",
    "sql_not_generated",
    "execution_error",
]

_TERMINAL_STATES: tuple[TerminalState, ...] = (
    "answered", "no_rows", "sql_invalid", "sql_too_expensive",
    "sql_timed_out", "retrieval_miss", "sql_not_generated",
    "execution_error",
)


@dataclass(frozen=True)
class QuestionGrade:
    """Per-question grade. Serialised as-is into the JSON report."""

    question_id: str
    question_text: str
    domain: str

    top5_packages: tuple[str, ...]
    top15_columns: tuple[tuple[str, str], ...]
    retrieval_recall_packages_at_5: float
    retrieval_recall_columns_at_15: float
    retrieval_miss: bool

    sql_generated: bool
    sql_text: str | None
    sql_final_text: str | None
    rationale: str | None
    answer_summary: str | None

    sql_valid: bool
    guard_reject_reason: str | None
    dry_run_bytes: int | None

    rows_returned: int | None
    bytes_billed: int | None
    slot_ms: int | None
    elapsed_ms: int | None
    sql_timed_out: bool | None

    terminal_state: TerminalState
    answered: bool | None
    notes_for_review: str | None
    execution_error: str | None

    rows_sample: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvalRunSummary:
    """Aggregate roll-up written to `<run_id>.json`'s top-level."""

    run_id: str
    started_at: str
    finished_at: str
    questions_total: int

    recall_packages_at_5_mean: float
    recall_columns_at_15_mean: float
    retrieval_misses: int

    sql_generated_count: int
    sql_valid_count: int
    answered_count: int | None
    no_rows_count: int | None
    sql_timed_out_count: int | None

    failures_by_reason: dict[str, int]

    total_bytes_billed: int | None
    total_slot_ms: int | None
    total_openai_input_tokens: int
    total_openai_output_tokens: int

    k_packages: int
    k_columns: int
    no_execute: bool
    fixture_path: str
    prompt_template_path: str
    prompt_template_hash: str
    generation_model: str
    embedding_model: str


@dataclass
class GradeInputs:
    """Mutable per-question intermediate the runner fills as it walks
    the pipeline. Collapsed into a `QuestionGrade` by `finalise_grade`.

    Kept mutable rather than frozen because the pipeline builds it up
    stage by stage; the frozen `QuestionGrade` is the final artefact
    consumed by the report writer.
    """

    question_id: str
    question_text: str
    domain: str
    expected_packages: tuple[str, ...]
    expected_columns: tuple[str, ...]
    must_return_rows: bool

    top_packages: tuple[str, ...] = ()
    top_columns: tuple[tuple[str, str], ...] = ()
    retrieval_miss: bool = False

    sql_generated: bool = False
    sql_text: str | None = None
    rationale: str | None = None
    answer_summary: str | None = None

    sql_final_text: str | None = None
    sql_valid: bool = False
    guard_reject_reason: str | None = None
    dry_run_bytes: int | None = None

    rows_returned: int | None = None
    bytes_billed: int | None = None
    slot_ms: int | None = None
    elapsed_ms: int | None = None
    sql_timed_out: bool | None = None
    execution_error: str | None = None

    rows_sample: tuple[dict[str, Any], ...] = ()
    no_execute: bool = False
    structured_output_violation: bool = False


def recall_at_k(retrieved: Iterable[str], expected: Iterable[str]) -> float:
    """Recall — not precision — because extra top-k hits don't hurt the
    operator. Empty `expected` is vacuous → 1.0 (documented in the
    report header so nobody misreads it)."""
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    retrieved_set = set(retrieved)
    return len(retrieved_set & expected_set) / len(expected_set)


def finalise_grade(inputs: GradeInputs) -> QuestionGrade:
    """Collapse the pipeline's intermediate state into a frozen grade.

    Asserts exactly one terminal state fires; two coexisting states
    raise `RuntimeError` — logic-bug tripwire, not a data check."""
    terminal_state = _terminal_state(inputs)

    answered = terminal_state == "answered"
    notes = _notes_for_review(inputs, terminal_state)

    return QuestionGrade(
        question_id=inputs.question_id,
        question_text=inputs.question_text,
        domain=inputs.domain,
        top5_packages=inputs.top_packages,
        top15_columns=inputs.top_columns,
        retrieval_recall_packages_at_5=recall_at_k(
            inputs.top_packages, inputs.expected_packages
        ),
        retrieval_recall_columns_at_15=recall_at_k(
            (c for _, c in inputs.top_columns), inputs.expected_columns
        ),
        retrieval_miss=inputs.retrieval_miss,
        sql_generated=inputs.sql_generated,
        sql_text=inputs.sql_text,
        sql_final_text=inputs.sql_final_text,
        rationale=inputs.rationale,
        answer_summary=inputs.answer_summary,
        sql_valid=inputs.sql_valid,
        guard_reject_reason=inputs.guard_reject_reason,
        dry_run_bytes=inputs.dry_run_bytes,
        rows_returned=inputs.rows_returned,
        bytes_billed=inputs.bytes_billed,
        slot_ms=inputs.slot_ms,
        elapsed_ms=inputs.elapsed_ms,
        sql_timed_out=inputs.sql_timed_out,
        terminal_state=terminal_state,
        answered=answered if inputs.must_return_rows else None,
        notes_for_review=notes,
        execution_error=inputs.execution_error,
        rows_sample=inputs.rows_sample,
    )


def _terminal_state(inputs: GradeInputs) -> TerminalState:
    candidates: list[TerminalState] = []
    if inputs.retrieval_miss:
        candidates.append("retrieval_miss")
    elif inputs.structured_output_violation:
        candidates.append("sql_not_generated")
    elif not inputs.sql_valid:
        reason = inputs.guard_reject_reason or ""
        if reason.startswith("sql_cost_too_high"):
            candidates.append("sql_too_expensive")
        else:
            candidates.append("sql_invalid")
    elif inputs.no_execute:
        # No terminal execution state — the operator ran --no-execute
        # deliberately. Treat as `no_rows` (the least commital of the
        # execution-side states) with a note.
        candidates.append("no_rows")
    elif inputs.sql_timed_out:
        candidates.append("sql_timed_out")
    elif inputs.execution_error:
        candidates.append("execution_error")
    elif inputs.rows_returned is not None and inputs.rows_returned > 0:
        candidates.append("answered")
    else:
        candidates.append("no_rows")

    if len(candidates) != 1:
        raise RuntimeError(
            f"grader tripwire: expected one terminal state, got {candidates!r}"
        )
    terminal = candidates[0]
    if terminal not in _TERMINAL_STATES:
        raise RuntimeError(f"grader tripwire: unknown terminal state {terminal!r}")
    return terminal


def _notes_for_review(
    inputs: GradeInputs, terminal_state: TerminalState
) -> str | None:
    if terminal_state == "answered":
        if inputs.must_return_rows:
            return None
        return "returned rows but must_return_rows=false; check fixture intent"
    if terminal_state == "no_rows":
        if inputs.no_execute:
            return "no-execute mode; execution skipped"
        if inputs.must_return_rows:
            return "expected rows but query returned zero"
        return None
    if terminal_state == "retrieval_miss":
        return "zero VECTOR_SEARCH hits; verify semantic.* is populated"
    if terminal_state == "sql_not_generated":
        return "structured_output_violation despite strict mode"
    return None


# ── Aggregate + writer ──


@dataclass
class RunAggregator:
    """Streaming aggregate. Filled per question; frozen into
    `EvalRunSummary` at end of run."""

    run_id: str
    started_at: datetime
    k_packages: int
    k_columns: int
    no_execute: bool
    fixture_path: str
    prompt_template_path: str
    prompt_template_hash: str
    generation_model: str
    embedding_model: str

    grades: list[QuestionGrade] = field(default_factory=list)
    total_openai_input_tokens: int = 0
    total_openai_output_tokens: int = 0

    def add(self, grade: QuestionGrade) -> None:
        self.grades.append(grade)

    def add_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.total_openai_input_tokens += tokens_in
        self.total_openai_output_tokens += tokens_out

    def finalise(self, *, finished_at: datetime) -> EvalRunSummary:
        n = len(self.grades)
        pkg_mean = (
            sum(g.retrieval_recall_packages_at_5 for g in self.grades) / n
            if n else 0.0
        )
        col_mean = (
            sum(g.retrieval_recall_columns_at_15 for g in self.grades) / n
            if n else 0.0
        )
        retrieval_misses = sum(1 for g in self.grades if g.retrieval_miss)
        sql_generated_count = sum(1 for g in self.grades if g.sql_generated)
        sql_valid_count = sum(1 for g in self.grades if g.sql_valid)

        failures: dict[str, int] = {}
        for g in self.grades:
            if g.terminal_state not in ("answered", "no_rows"):
                failures[g.terminal_state] = failures.get(g.terminal_state, 0) + 1

        if self.no_execute:
            answered_count: int | None = None
            no_rows_count: int | None = None
            sql_timed_out_count: int | None = None
            total_bytes_billed: int | None = None
            total_slot_ms: int | None = None
        else:
            answered_count = sum(
                1 for g in self.grades if g.terminal_state == "answered"
            )
            no_rows_count = sum(
                1 for g in self.grades if g.terminal_state == "no_rows"
            )
            sql_timed_out_count = sum(
                1 for g in self.grades if g.terminal_state == "sql_timed_out"
            )
            total_bytes_billed = sum(
                g.bytes_billed or 0 for g in self.grades
            )
            total_slot_ms = sum(g.slot_ms or 0 for g in self.grades)

        return EvalRunSummary(
            run_id=self.run_id,
            started_at=self.started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            questions_total=n,
            recall_packages_at_5_mean=pkg_mean,
            recall_columns_at_15_mean=col_mean,
            retrieval_misses=retrieval_misses,
            sql_generated_count=sql_generated_count,
            sql_valid_count=sql_valid_count,
            answered_count=answered_count,
            no_rows_count=no_rows_count,
            sql_timed_out_count=sql_timed_out_count,
            failures_by_reason=failures,
            total_bytes_billed=total_bytes_billed,
            total_slot_ms=total_slot_ms,
            total_openai_input_tokens=self.total_openai_input_tokens,
            total_openai_output_tokens=self.total_openai_output_tokens,
            k_packages=self.k_packages,
            k_columns=self.k_columns,
            no_execute=self.no_execute,
            fixture_path=self.fixture_path,
            prompt_template_path=self.prompt_template_path,
            prompt_template_hash=self.prompt_template_hash,
            generation_model=self.generation_model,
            embedding_model=self.embedding_model,
        )


def write_reports(
    *,
    summary: EvalRunSummary,
    grades: list[QuestionGrade],
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write `<run_id>.json` (machine) and `<run_id>.md` (human).

    Returns the two paths so the CLI can echo them."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{summary.run_id}.json"
    md_path = reports_dir / f"{summary.run_id}.md"

    payload = {
        "summary": asdict(summary),
        "grades": [_grade_to_json(g) for g in grades],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(summary, grades), encoding="utf-8")
    return json_path, md_path


def _grade_to_json(grade: QuestionGrade) -> dict[str, Any]:
    raw = asdict(grade)
    # `top15_columns` is a tuple of 2-tuples; asdict already turns it
    # into a nested list, but JSON doesn't distinguish. Round-trip
    # cleanly for downstream diff tools.
    return raw


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _render_markdown(
    summary: EvalRunSummary, grades: list[QuestionGrade]
) -> str:
    lines: list[str] = []
    lines.append(f"# Eval run `{summary.run_id}`")
    lines.append("")
    lines.append(f"- Started: {summary.started_at}")
    lines.append(f"- Finished: {summary.finished_at}")
    lines.append(f"- Fixture: `{summary.fixture_path}`")
    lines.append(f"- Prompt: `{summary.prompt_template_path}`")
    lines.append(f"- Prompt hash: `{summary.prompt_template_hash}`")
    lines.append(
        f"- Models: generation=`{summary.generation_model}`, "
        f"embedding=`{summary.embedding_model}`"
    )
    lines.append(f"- Questions: {summary.questions_total}")
    lines.append(
        f"- Recall packages@5 mean: {summary.recall_packages_at_5_mean:.3f}"
    )
    lines.append(
        f"- Recall columns@15 mean: {summary.recall_columns_at_15_mean:.3f}"
    )
    lines.append("")
    lines.append(
        "> Empty `expected_packages` / `expected_columns` contribute 1.0 to "
        "the per-question recall (vacuous case)."
    )
    lines.append("")
    lines.append("## Questions")
    for grade in grades:
        lines.append("")
        lines.append(f"### `{grade.question_id}` — {grade.domain}")
        lines.append("")
        lines.append(f"> {grade.question_text}")
        lines.append("")
        lines.append(f"- Terminal state: `{grade.terminal_state}`")
        lines.append(
            f"- Recall packages@5: {grade.retrieval_recall_packages_at_5:.3f}"
        )
        lines.append(
            f"- Recall columns@15: {grade.retrieval_recall_columns_at_15:.3f}"
        )
        if grade.dry_run_bytes is not None:
            lines.append(f"- Dry-run bytes: {grade.dry_run_bytes:,}")
        if grade.bytes_billed is not None:
            lines.append(f"- Bytes billed: {grade.bytes_billed:,}")
        if grade.guard_reject_reason:
            lines.append(f"- Guard reject: `{grade.guard_reject_reason}`")
        if grade.execution_error:
            lines.append(f"- Execution error: `{grade.execution_error}`")
        if grade.notes_for_review:
            lines.append(f"- Notes: {grade.notes_for_review}")
        if grade.sql_text:
            lines.append("")
            lines.append("SQL (model output):")
            lines.append("")
            lines.append("```sql")
            lines.append(grade.sql_text)
            lines.append("```")
        if grade.sql_final_text and grade.sql_final_text != grade.sql_text:
            lines.append("")
            lines.append("SQL (post-guard):")
            lines.append("")
            lines.append("```sql")
            lines.append(grade.sql_final_text)
            lines.append("```")
        if grade.answer_summary:
            lines.append(f"- Answer summary: {grade.answer_summary}")
        if grade.rationale:
            lines.append(f"- Rationale: {grade.rationale}")
        if grade.rows_sample:
            lines.append("")
            lines.append(f"Rows (first {len(grade.rows_sample)}):")
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(list(grade.rows_sample), indent=2, default=_json_default)
            )
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_partial_grade(*, partial_path: Path, grade: QuestionGrade) -> None:
    """Append one grade as JSONL. Crash-recovery affordance: the runner
    writes the final report only on clean completion, so a mid-run
    crash leaves the partial file for review."""
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_grade_to_json(grade), default=_json_default))
        f.write("\n")
