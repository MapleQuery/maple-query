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
from semantic_enrich.core.agent import research
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
    Verdict,
)
from semantic_enrich.core.agent_request import ChatRequest


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

    # ── verify ──
    yield record(agent_events.PhaseStart(phase="verify"))
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
        verdict = deps.verifier.check(ctx, result, final=True)
        for event in verdict.events:
            yield record(event)

    yield from _finish(
        ctx,
        message=_compose(result, verdict),
        result=result,
        record=record,
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


def _compose(result: ResearchResult, verdict: Verdict) -> str:
    # The stub verifier never rewrites; the verify phase later shapes
    # caveats/retries through its hints instead of editing the text.
    del verdict
    return result.candidate_answer


def _finish(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
    record: Callable[[agent_events.AgentEvent], agent_events.AgentEvent],
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
    yield record(
        agent_events.TurnRecordEvent(
            record=_turn_record(ctx, message=message, result=result)
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


def _turn_record(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
) -> dict[str, Any]:
    """Minimal turn-record skeleton. The memory phase owns the real
    schema; until it lands this carries what the context knows."""
    return {
        "turn_id": ctx.turn_id,
        "conversation_id": ctx.request.conversation_id,
        "question": ctx.request.question,
        "answer": message,
        "loop_impl": "v2",
        "terminal_reason": (
            result.terminal_reason if result else "triage_short_circuit"
        ),
        "packages": list(result.packages_cited) if result else [],
        "columns_referenced": (
            list(result.columns_referenced) if result else []
        ),
        "sql_run_count": len(result.sql_runs) if result else 0,
        "tool_call_count": ctx.tool_call_count,
        "tokens_in_total": ctx.tokens_in_total,
        "tokens_out_total": ctx.tokens_out_total,
        "dollars_spent": round(ctx.dollars_spent, 6),
        "verify_retries_used": ctx.verify_retries_used,
        "reformulations_used": ctx.reformulations_used,
        "snapshot_hash": ctx.snapshot_hash,
    }
