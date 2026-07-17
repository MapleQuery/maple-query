"""Agent-mode eval: run a labeled question fixture through the live loop.

The fixture (`eval/questions-agent-traces.yaml`) is derived from real
agent traffic. Each entry carries the question, expected triage/outcome
labels, and what the loop actually did when the trace was captured
(`observed_v1`). The runner replays every question against the live
loop as a fresh single-turn conversation — with tracing on when
configured — and writes a JSON report that serves as the behavioural
baseline for later loop changes.

`safe_load` only, same posture as `eval_question_set`.

No automatic grading happens here: the labels are consumed by later
triage/parity work; this runner records outcomes so humans (and future
scorers) can diff them against `expected`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_dispatch import (
    LoopImpl,
    resolve_run_turn,
)
from semantic_enrich.core.agent_request import ChatRequest
from semantic_enrich.core.agent_tracing import (
    TracedDeps,
    TurnObserver,
    run_turn_traced,
    session_span_map_from_settings,
)
from semantic_enrich.providers.logging import get_logger

ALLOWED_TRIAGE = frozenset({"in_scope", "off_scope", "meta", "clarify"})
ALLOWED_OUTCOMES = frozenset({"answered", "no_data", "deflected", "clarify"})


@dataclass(frozen=True)
class AgentEvalQuestion:
    """One labeled fixture entry, validated at load time.

    `expected_triage` doubles as a training/eval label for scope
    triage; `expected_outcome` + `must_caveat` feed answer-parity
    rubrics. `observed_outcome`/`observed_note` record what the loop
    did in the source traces — free-form, since observed behaviour
    (e.g. "surrendered") is richer than the expected-outcome enum."""

    id: str
    question: str
    source: str
    expected_triage: str
    expected_outcome: str
    packages_any_of: tuple[str, ...]
    must_caveat: bool
    observed_outcome: str
    observed_note: str


class AgentQuestionSetError(RuntimeError):
    """Fixture load or schema failure. Terminal for the run."""


def load_agent_question_set(path: Path) -> list[AgentEvalQuestion]:
    """Read, parse, and validate the agent-traces fixture."""
    if not path.exists():
        raise AgentQuestionSetError(f"agent eval fixture missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise AgentQuestionSetError(
            f"agent eval fixture must be a YAML list, got {type(raw).__name__}"
        )
    if not raw:
        raise AgentQuestionSetError("agent eval fixture is empty")

    seen_ids: set[str] = set()
    questions: list[AgentEvalQuestion] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise AgentQuestionSetError(
                f"agent eval questions[{i}] must be a mapping, "
                f"got {type(entry).__name__}"
            )
        question = _validate_entry(entry, index=i)
        if question.id in seen_ids:
            raise AgentQuestionSetError(
                f"duplicate question id {question.id!r} at index {i}"
            )
        seen_ids.add(question.id)
        questions.append(question)
    return questions


def _validate_entry(
    entry: dict[str, Any], *, index: int
) -> AgentEvalQuestion:
    prefix = f"agent eval questions[{index}]"

    qid = entry.get("id")
    if not isinstance(qid, str) or not qid.strip():
        raise AgentQuestionSetError(f"{prefix}.id must be a non-empty string")

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        raise AgentQuestionSetError(
            f"{prefix}.question must be a non-empty string"
        )

    source = entry.get("source", "")
    if not isinstance(source, str):
        raise AgentQuestionSetError(f"{prefix}.source must be a string")

    expected = entry.get("expected")
    if not isinstance(expected, dict):
        raise AgentQuestionSetError(f"{prefix}.expected must be a mapping")

    triage = expected.get("triage")
    if not isinstance(triage, str) or triage not in ALLOWED_TRIAGE:
        raise AgentQuestionSetError(
            f"{prefix}.expected.triage must be one of "
            f"{sorted(ALLOWED_TRIAGE)}, got {triage!r}"
        )

    outcome = expected.get("outcome")
    if not isinstance(outcome, str) or outcome not in ALLOWED_OUTCOMES:
        raise AgentQuestionSetError(
            f"{prefix}.expected.outcome must be one of "
            f"{sorted(ALLOWED_OUTCOMES)}, got {outcome!r}"
        )

    packages_raw = expected.get("packages_any_of", [])
    if not isinstance(packages_raw, list) or not all(
        isinstance(p, str) and p.strip() for p in packages_raw
    ):
        raise AgentQuestionSetError(
            f"{prefix}.expected.packages_any_of must be a list of "
            "non-empty strings"
        )

    must_caveat = expected.get("must_caveat", False)
    if not isinstance(must_caveat, bool):
        raise AgentQuestionSetError(
            f"{prefix}.expected.must_caveat must be a bool"
        )

    observed = entry.get("observed_v1", {}) or {}
    if not isinstance(observed, dict):
        raise AgentQuestionSetError(f"{prefix}.observed_v1 must be a mapping")
    observed_outcome = observed.get("outcome", "")
    observed_note = observed.get("note", "")
    if not isinstance(observed_outcome, str) or not isinstance(
        observed_note, str
    ):
        raise AgentQuestionSetError(
            f"{prefix}.observed_v1.outcome/.note must be strings"
        )

    return AgentEvalQuestion(
        id=qid.strip(),
        question=question.strip(),
        source=source,
        expected_triage=triage,
        expected_outcome=outcome,
        packages_any_of=tuple(packages_raw),
        must_caveat=must_caveat,
        observed_outcome=observed_outcome,
        observed_note=observed_note,
    )


# ── Baseline runner ──


@dataclass(frozen=True)
class AgentEvalRequest:
    """CLI intent for one agent-mode eval run."""

    run_id: str
    limit: int | None
    output_override: Path | None
    question_ids: tuple[str, ...] | None = None


def run_agent_eval(
    *,
    request: AgentEvalRequest,
    settings: Settings,
    deps: TracedDeps,
    loop_impl: LoopImpl = "v1",
    logger: structlog.BoundLogger | None = None,
) -> dict[str, Any]:
    """Run every fixture question through the loop, one fresh
    single-turn conversation each, and write the JSON baseline report.

    Returns the report dict (also written to disk)."""
    log = logger or get_logger("semantic_enrich.agent_eval")
    started = datetime.now(UTC)

    questions = load_agent_question_set(settings.eval_questions_path)
    if request.question_ids:
        wanted = set(request.question_ids)
        questions = [q for q in questions if q.id in wanted]
        missing = wanted - {q.id for q in questions}
        if missing:
            raise RuntimeError(
                f"agent eval question id(s) not in fixture: {sorted(missing)}"
            )
    if request.limit is not None:
        questions = questions[: request.limit]

    session_map = session_span_map_from_settings(settings)

    log.info(
        "agent_eval_start",
        run_id=request.run_id,
        questions_count=len(questions),
        fixture_path=str(settings.eval_questions_path),
        generation_model=settings.openai_generation_model,
    )

    results: list[dict[str, Any]] = []
    for q in questions:
        conversation_id = f"agent-eval-{request.run_id[:8]}-{q.id}"
        chat_request = ChatRequest(
            conversation_id=conversation_id,
            history=[],
            question=q.question,
        )
        observer = TurnObserver()
        turn_started = time.monotonic()
        for event in run_turn_traced(
            request=chat_request,
            deps=deps,
            session_parent=session_map.get_or_create(conversation_id),
            run_turn_fn=resolve_run_turn(loop_impl),
            loop_impl=loop_impl,
        ):
            observer.observe(event)
        elapsed_ms = int((time.monotonic() - turn_started) * 1000)
        results.append(
            {
                "id": q.id,
                "question": q.question,
                "expected": {
                    "triage": q.expected_triage,
                    "outcome": q.expected_outcome,
                    "packages_any_of": list(q.packages_any_of),
                    "must_caveat": q.must_caveat,
                },
                "observed_v1_trace": {
                    "outcome": q.observed_outcome,
                    "note": q.observed_note,
                },
                "run": {
                    "terminal": observer.terminal,
                    "final_message": observer.final_message,
                    "tool_call_count": observer.tool_call_count,
                    "dollars_spent": observer.dollars_spent,
                    "tokens_in_per_call": observer.tokens_in_per_call,
                    "tokens_out_per_call": observer.tokens_out_per_call,
                    "elapsed_ms": elapsed_ms,
                    "cached": observer.cached,
                    "top_similarities": observer.top_similarities,
                    "reformulations": observer.reformulations,
                },
            }
        )
        log.info(
            "agent_eval_question_finish",
            question_id=q.id,
            terminal=observer.terminal,
            tool_calls=observer.tool_call_count,
            dollars=round(observer.dollars_spent, 6),
            elapsed_ms=elapsed_ms,
        )

    finished = datetime.now(UTC)
    report: dict[str, Any] = {
        "run_id": request.run_id,
        "mode": "agent",
        "loop_impl": loop_impl,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "fixture_path": str(settings.eval_questions_path),
        "prompt_hash": deps.prompt_hash,
        "system_prompt_tokens": deps.system_prompt_tokens,
        "generation_model": settings.openai_generation_model,
        "questions_count": len(results),
        "totals": {
            "dollars_spent": round(
                sum(r["run"]["dollars_spent"] for r in results), 6
            ),
            "tool_calls": sum(r["run"]["tool_call_count"] for r in results),
            "terminal_counts": _terminal_counts(results),
        },
        "similarity_calibration": _similarity_calibration(
            results, floor=settings.agent_search_similarity_floor
        ),
        "questions": results,
    }

    output_path = request.output_override or (
        settings.eval_reports_dir / f"agent-traces-{request.run_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    log.info(
        "agent_eval_finish",
        run_id=request.run_id,
        report_path=str(output_path),
        totals=report["totals"],
    )
    return report


def _similarity_calibration(
    results: list[dict[str, Any]], *, floor: float
) -> dict[str, Any]:
    """Distribution of first-search top_similarity, grouped by the
    fixture's expected outcome. This is the evidence for choosing the
    weak-retrieval floor: pick the value that separates questions the
    corpus can answer from questions it genuinely can't."""
    by_outcome: dict[str, list[float]] = {}
    for r in results:
        sims = [
            s
            for s in r["run"].get("top_similarities", [])
            if isinstance(s, int | float)
        ]
        if not sims:
            continue
        outcome = str(r["expected"]["outcome"])
        by_outcome.setdefault(outcome, []).append(float(sims[0]))
    summary: dict[str, Any] = {"floor": floor, "by_expected_outcome": {}}
    for outcome, sims in sorted(by_outcome.items()):
        ordered = sorted(sims)
        summary["by_expected_outcome"][outcome] = {
            "count": len(ordered),
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "below_floor": sum(1 for s in ordered if s < floor),
            "values": ordered,
        }
    return summary


def _terminal_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        terminal = str(r["run"]["terminal"])
        counts[terminal] = counts.get(terminal, 0) + 1
    return counts


# Re-exported for tests that want to build fixture rows programmatically.
__all__ = [
    "ALLOWED_OUTCOMES",
    "ALLOWED_TRIAGE",
    "AgentEvalQuestion",
    "AgentEvalRequest",
    "AgentQuestionSetError",
    "load_agent_question_set",
    "run_agent_eval",
]
