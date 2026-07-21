"""The v2 turn orchestrator.

One generator drives five explicit phases — triage, memory, research,
verify, answer — through typed interfaces. Phases return their events;
this module is the only place that yields (research streams through a
recording shim), which keeps event ordering auditable in one spot.

External contract is v1's: SSE event stream, exactly one terminal
`done`/`error` per turn, all v1 event payloads unchanged. Additive
`phase_start`/`turn_record` events are ignored by v1 consumers.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any

from semantic_enrich.core import agent_events, agent_history
from semantic_enrich.core.agent import grounding, records, research
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
    Verdict,
)
from semantic_enrich.core.agent_request import ChatRequest
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent.pipeline")


@dataclass
class PipelineOutcome:
    """Collected form of a turn, mirroring the v1 outcome type so the
    harness can grade either loop without re-parsing the event log."""

    turn_id: str
    final_message: str
    tool_call_count: int
    dollars_spent: float
    events: list[agent_events.AgentEvent] = field(default_factory=list)
    cache_hit: bool = False


def run_turn(
    *,
    request: ChatRequest,
    deps: PipelineDeps,
) -> Iterator[agent_events.AgentEvent]:
    """Run one turn, yielding events lazily. Same calling convention
    as the v1 loop so the tracing wrapper and callers swap freely."""
    # History validation precedes everything, including the snapshot
    # lookup in `TurnContext.begin` — an invalid history must not cost
    # a warehouse query.
    try:
        agent_history.validate(request.history, settings=deps.settings)
    except agent_history.InvalidHistoryError as exc:
        yield agent_events.TurnStart(
            conversation_id=request.conversation_id,
            turn_id=uuid.uuid4().hex,
            cached=False,
        )
        yield agent_events.ErrorEvent(
            message=str(exc),
            retryable=False,
            reason="invalid_history",
        )
        return

    ctx = TurnContext.begin(request=request, deps=deps)

    def record(
        event: agent_events.AgentEvent,
    ) -> agent_events.AgentEvent:
        ctx.events.append(event)
        return event

    # ── triage ──
    yield record(agent_events.PhaseStart(phase="triage"))
    triage = deps.triage.classify(ctx)
    ctx.triage_category = triage.category
    for event in triage.events:
        yield record(event)
    if triage.short_circuit is not None:
        yield from _finish(
            ctx, message=triage.short_circuit, result=None, record=record
        )
        return

    # ── memory ──
    yield record(agent_events.PhaseStart(phase="memory"))
    recall = deps.memory.recall(ctx)
    for event in recall.events:
        yield record(event)
    if recall.replay is not None:
        # Cache replay: pass through unrecorded — replayed turns are
        # never re-cached.
        yield from recall.replay
        return
    ctx.history_messages = recall.history_messages

    yield record(
        agent_events.TurnStart(
            conversation_id=request.conversation_id,
            turn_id=ctx.turn_id,
            cached=False,
        )
    )
    ctx.turn_start_emitted = True

    # ── research ──
    yield record(agent_events.PhaseStart(phase="research"))
    result = yield from _record_stream(
        ctx, research.run(ctx, hints=recall.hints)
    )
    if _is_terminal_error(result):
        yield record(_terminal_error_event(result))
        return

    # ── grounding (deterministic; feeds verify) ──
    _attach_grounding(ctx, result)

    # ── verify ──
    yield record(agent_events.PhaseStart(phase="verify"))
    if _skip_verify(ctx, result):
        verdict = Verdict(action="accept")
    else:
        verdict = deps.verifier.check(ctx, result)
        for event in verdict.events:
            yield record(event)
        if verdict.action == "retry" and ctx.retries_remaining():
            ctx.verify_retries_used += 1
            yield record(agent_events.PhaseStart(phase="research"))
            result = yield from _record_stream(
                ctx, research.run(ctx, hints=verdict.hints)
            )
            if _is_terminal_error(result):
                yield record(_terminal_error_event(result))
                return
            _attach_grounding(ctx, result)
            if _skip_verify(ctx, result):
                verdict = Verdict(action="accept")
            else:
                verdict = deps.verifier.check(ctx, result, final=True)
                for event in verdict.events:
                    yield record(event)

    yield from _finish(
        ctx,
        message=_compose(result, verdict),
        result=result,
        record=record,
        outcome_override=verdict.outcome_override,
    )


def run_turn_collected(
    *, request: ChatRequest, deps: PipelineDeps
) -> PipelineOutcome:
    """Drain `run_turn` into a `PipelineOutcome` for the harness and
    tests; streaming callers consume the iterator directly."""
    events: list[agent_events.AgentEvent] = []
    for event in run_turn(request=request, deps=deps):
        events.append(event)
    turn_id = ""
    final = ""
    tool_calls = 0
    dollars = 0.0
    cache_hit = False
    for event in events:
        if isinstance(event, agent_events.TurnStart):
            turn_id = event.turn_id
            cache_hit = event.cached
        elif isinstance(event, agent_events.MessageDelta):
            final += event.delta
        elif isinstance(event, agent_events.Done):
            tool_calls = event.total_tool_calls
            dollars = event.total_dollars
    return PipelineOutcome(
        turn_id=turn_id,
        final_message=final,
        tool_call_count=tool_calls,
        dollars_spent=dollars,
        events=events,
        cache_hit=cache_hit,
    )


def _record_stream(
    ctx: TurnContext,
    inner: Generator[agent_events.AgentEvent, None, ResearchResult],
) -> Generator[agent_events.AgentEvent, None, ResearchResult]:
    """Yield everything from `inner`, recording each event on the
    context for the commit-time cache write."""
    while True:
        try:
            event = next(inner)
        except StopIteration as stop:
            result: ResearchResult = stop.value
            return result
        ctx.events.append(event)
        yield event


def _derivation_events(
    result: ResearchResult | None,
) -> list[agent_events.DerivationEvent]:
    """One curated 'how I got this number' event per value-bearing
    derivation. Additive and emit-only; shape-only derivations (no
    scalar value) are not surfaced as a panel in M6."""
    if result is None:
        return []
    cross = (
        result.grounding is not None
        and result.grounding.cross_source_sum.flagged
    )
    ungrounded = (
        result.grounding is not None
        and result.grounding.grounding == "ungrounded"
    )
    events: list[agent_events.DerivationEvent] = []
    for d in result.derivations:
        if d.result_value is None:
            continue
        flags: list[str] = []
        if cross and d.aggregation == "SUM" and len(d.source_packages) >= 2:
            flags.append("cross_source_sum")
        if d.unit_scale == "unknown":
            flags.append("unknown_units")
        if ungrounded:
            flags.append("ungrounded")
        events.append(
            agent_events.DerivationEvent(
                dataset_titles=list(d.dataset_titles),
                source_packages=list(d.source_packages),
                aggregation=d.aggregation,
                value_columns=list(d.value_columns),
                scope=d.predicate_shape,
                row_count=d.row_count,
                source_row_estimate=d.source_row_estimate,
                result_value=d.result_value,
                result_label=d.result_label,
                unit_scale=d.unit_scale,
                unit_source=d.unit_source,
                flags=flags,
            )
        )

    # The invariant: a numeric answer never ships with no trace. When
    # the answer states a figure that ties to no computed total (read
    # from a cell, or fabricated), emit an explicit "unverified" trace
    # so the panel still appears — saying plainly that there is no
    # computation behind the number, rather than showing nothing.
    if not events and ungrounded and result.grounding is not None:
        headline = result.grounding.headline_value
        titles = _packages_and_titles(result)
        events.append(
            agent_events.DerivationEvent(
                dataset_titles=titles,
                source_packages=list(result.packages_cited),
                aggregation="none",
                value_columns=[],
                scope="",
                row_count=0,
                source_row_estimate=0,
                result_value=headline,
                result_label=None,
                unit_scale="unknown",
                unit_source="unresolved",
                flags=["unverified"],
            )
        )
    return events


def _packages_and_titles(result: ResearchResult) -> list[str]:
    seen: list[str] = []
    for d in result.derivations:
        for title in d.dataset_titles:
            if title and title not in seen:
                seen.append(title)
    return seen


def _attach_grounding(ctx: TurnContext, result: ResearchResult) -> None:
    """Compute the deterministic grounding report over the candidate
    answer and stash it on the result for the verify phase. Runs on the
    same set of turns verify does (research-produced, non-clarify
    answers); always logged, never alters the answer here."""
    if _skip_verify(ctx, result):
        return
    report = grounding.build_grounding_report(
        result.candidate_answer, result.derivations
    )
    result.grounding = report
    _LOG.info(
        "grounding",
        grounding=report.grounding,
        matched=report.matched,
        headline_value=report.headline_value,
        cross_source_flagged=report.cross_source_sum.flagged,
        fiscal_years=list(report.cross_source_sum.fiscal_years),
    )


def _is_terminal_error(result: ResearchResult) -> bool:
    return result.terminal_reason in ("error", "timeout")


def _terminal_error_event(result: ResearchResult) -> agent_events.ErrorEvent:
    if result.terminal_reason == "timeout":
        return agent_events.ErrorEvent(
            message="turn wall-clock exceeded",
            retryable=False,
            reason="turn_timeout",
        )
    return agent_events.ErrorEvent(
        message=result.error_message,
        retryable=result.error_retryable,
        reason=result.error_reason,
    )


def _skip_verify(ctx: TurnContext, result: ResearchResult) -> bool:
    """Verification runs only on research-produced candidate answers:
    budget/timeout-forced answers already carry their best-effort
    framing (delaying them further is worse), and a clarifying
    question is not a claim to check."""
    if result.terminal_reason != "final_answer":
        return True
    return (
        _outcome(ctx, message=result.candidate_answer, result=result)
        == "clarified"
    )


def _compose(result: ResearchResult, verdict: Verdict) -> str:
    # The verify phase never rewrites the model's text: it either
    # ships the candidate unchanged, wraps it under a template caveat,
    # or replaces a surrender with a clarifying question.
    return verdict.composed_message or result.candidate_answer


def _finish(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
    record: Callable[[agent_events.AgentEvent], agent_events.AgentEvent],
    outcome_override: str | None = None,
) -> Iterator[agent_events.AgentEvent]:
    if not ctx.turn_start_emitted:
        # Short-circuit paths (triage) still owe the FE a turn_start.
        yield record(
            agent_events.TurnStart(
                conversation_id=ctx.request.conversation_id,
                turn_id=ctx.turn_id,
                cached=False,
            )
        )
        ctx.turn_start_emitted = True
    yield record(agent_events.PhaseStart(phase="answer"))
    if message:
        yield record(agent_events.MessageDelta(delta=message))
    for event in _derivation_events(result):
        yield record(event)
    yield record(
        agent_events.TurnRecordEvent(
            record=_turn_record(
                ctx,
                message=message,
                result=result,
                outcome_override=outcome_override,
            )
        )
    )
    done = agent_events.Done(
        turn_id=ctx.turn_id,
        total_tool_calls=ctx.tool_call_count,
        total_dollars=round(ctx.dollars_spent, 6),
        elapsed_ms=ctx.elapsed_ms(),
    )
    # Record before commit so the cache write sees the complete,
    # `done`-terminated stream, then yield.
    ctx.events.append(done)
    ctx.deps.memory.commit(ctx)
    yield done


def _outcome(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
) -> str:
    """Coarse outcome tag for the turn record. A clarify is detected
    deterministically rather than by parsing intent: the loop served
    the cap-reached steer, no SQL succeeded, and the final message
    asks a question. An answer with no successful SQL behind it is a
    no-data claim, not an answer — the distinction gates the replay
    cache and plan-hint selection."""
    if result is None:
        return (
            "clarified"
            if ctx.triage_category == "clarify"
            else "deflected"
        )
    if result.terminal_reason in ("error", "timeout"):
        return "error"
    sql_succeeded = any(
        run.get("status") == "ok" for run in result.sql_runs
    )
    if (
        ctx.state.clarify_steer_issued
        and not sql_succeeded
        and "?" in message
    ):
        return "clarified"
    return "answered" if sql_succeeded else "no_data"


def _turn_record(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
    outcome_override: str | None = None,
) -> dict[str, Any]:
    outcome = outcome_override or _outcome(
        ctx, message=message, result=result
    )
    return records.build(
        ctx, message=message, result=result, outcome=outcome
    )
