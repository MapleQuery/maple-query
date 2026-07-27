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

from semantic_enrich.core import agent_events, agent_history
from semantic_enrich.core.agent import (
    evidence,
    grounding,
    records,
    research,
    scope,
)
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    SystemHint,
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
        ctx, research.run(ctx, hints=_hints(ctx, recall.hints))
    )
    if _is_terminal_error(result):
        yield record(_terminal_error_event(result))
        return

    # ── grounding (deterministic; feeds verify + the trace) ──
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
        message=_verdict_message(result, verdict),
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


def _hints(
    ctx: TurnContext, recalled: list[SystemHint]
) -> list[SystemHint]:
    """The recalled hints plus the package scope, if this turn carries
    one — and the whitelist admission the scope hint needs to mean
    anything.

    `LoopState.known_package_ids` is empty at the start of every turn
    and populated only by `search_datasets`, while `list_documents` and
    `search_columns` reject ids that are not in it. So a model that
    *obeys* the scope hint, reaching for `list_documents` as its first
    move, would take a tool error and quietly fall back to a broad
    search: the turn completes, ships a plausible answer, and the scope
    does nothing, invisibly. `memory.recall` already admits plan-hint
    packages for exactly this reason; this is the same step for the same
    cause. Admission widens nothing — the scope reaches only packages
    the user could already retrieve by searching, and `list_documents`
    still resolves every id against the warehouse.

    Lives here rather than in the memory phase so a scoped turn behaves
    the same under `SessionMemory` and `NoopMemory`.
    """
    if not ctx.scope_package_ids:
        return recalled
    ctx.state.known_package_ids.update(ctx.scope_package_ids)
    text = scope.render_hint(
        ctx.scope_package_ids,
        titles=scope.titles_from_records(ctx.request.turn_records),
    )
    return [*recalled, SystemHint(text=text)] if text else recalled


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
    answer and stash it on the result. Grounding is pure and free (no
    model call), so it runs on every real candidate answer — including
    budget-forced ones. Those carry the least-trusted numbers (the loop
    ran out of room before it could tie the figure to a computed total),
    so they are exactly the answers that most need the unverified trace;
    gating grounding on `_skip_verify` used to drop it for them. Verify
    itself still skips forced answers — this only feeds the trace."""
    if _skip_grounding(ctx, result):
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
    if _is_descriptive_scope(ctx, result):
        return True
    return _candidate_is_clarify(ctx, result)


def _skip_grounding(ctx: TurnContext, result: ResearchResult) -> bool:
    """Grounding feeds the trace, not just verify, so it runs on a wider
    set of turns than `_skip_verify`: every answer that makes a claim,
    forced or not. It skips only non-answers (error/timeout ship an
    error event, no claim), clarifying questions (not a claim to
    ground), and descriptive scoped turns.

    That last one is not symmetry for its own sake. `_derivation_events`
    emits an explicit "unverified" trace whenever grounding comes back
    `ungrounded`, and a dataset summary quoting a monetary range — "the
    amounts run from $1,000 to $2.5M" — yields monetary numbers with no
    derivation behind them. It would ground as `ungrounded` and paint a
    "no computation behind this number" panel on a turn that never
    claimed a total. (A summary quoting only row and column counts is
    already safe: `extract_numbers` keys on monetary values, so it lands
    on `no_numeric_claim`. The false panel needs a dollar sign, which
    dataset summaries routinely have.)"""
    if result.terminal_reason not in ("final_answer", "budget_forced"):
        return True
    if _is_descriptive_scope(ctx, result):
        return True
    return _candidate_is_clarify(ctx, result)


def _sql_succeeded(result: ResearchResult | None) -> bool:
    if result is None:
        return False
    return any(run.get("status") == "ok" for run in result.sql_runs)


def _is_descriptive_scope(
    ctx: TurnContext, result: ResearchResult | None
) -> bool:
    """A scoped turn that ran no successful SQL is an exploration, and
    exploration is judged on description rather than on answer fit.

    The `sql_succeeded` half is the important half and is not
    negotiable. The obvious rule — scoped turn, skip verify — would be a
    regression: a numeric follow-up ("now sum these columns") is *also*
    a scoped turn, so keying the bypass on scope alone would strip the
    derivation, grounding, and magnitude protection from exactly the
    answers guided recovery exists to produce. A numeric follow-up runs
    SQL, so it takes the full path unchanged.

    This also closes the clarify-replacement failure at its root rather
    than by adding a demotion: a descriptive turn never reaches verify,
    so `clarify` cannot fire on it and `compose_clarify` cannot replace
    a dataset summary with a question back to the user.
    """
    return bool(ctx.scope_package_ids) and not _sql_succeeded(result)


def _candidate_is_clarify(ctx: TurnContext, result: ResearchResult) -> bool:
    """Both skip gates ask the same question of the raw candidate — is
    the model's own text a clarifying question rather than a claim."""
    return (
        _outcome(ctx, message=result.candidate_answer, result=result)
        == "clarified"
    )


def _verdict_message(result: ResearchResult, verdict: Verdict) -> str:
    # The verify phase never rewrites the model's text: it either
    # ships the candidate unchanged, wraps it under a template caveat,
    # or replaces a surrender with a clarifying question.
    return verdict.composed_message or result.candidate_answer


def _compose(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
    outcome: str,
) -> str:
    """The single funnel every shipping message passes through, and the
    only place the evidence footer is appended.

    The obvious hook for a footer is `verify.compose_caveat`, since the
    surrender the milestone was written against carries a caveat
    header. That would be wrong: the caveat composer covers one of the
    five paths that ship a surrender. A verify `fits=True` on a no-data
    answer, verify in `off`/`log` mode, and a budget- or timeout-forced
    answer all ship the model's raw text — and those are precisely the
    paths carrying the "check with the relevant departments" ending
    this footer exists to follow. Only the clarify path is excluded,
    and it is excluded by the outcome gate below rather than by where
    the hook sits.

    `outcome` is computed by the caller on the *pre-footer* message.
    That ordering matters: `_outcome` detects a clarify partly by
    testing `"?" in message`, so gating the footer on an outcome
    derived from the footered message would be circular.
    """
    if outcome != "no_data":
        # `answered` / `answered_with_caveat` already give the user
        # something to act on. `clarified` only fires when retrieval was
        # weak, so a dataset list under the question would invite the
        # user to pick something the loop already scored as irrelevant.
        return message
    if not ctx.deps.settings.agent_evidence_footer:
        return message
    return message + evidence.compose_footer(
        evidence.collect_evidence(ctx, result)
    )


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
    # Computed once, on the pre-footer message, and shared between the
    # footer decision and the turn record. See `_compose`.
    outcome = outcome_override or _outcome(
        ctx, message=message, result=result
    )
    message = _compose(ctx, message=message, result=result, outcome=outcome)
    yield record(agent_events.PhaseStart(phase="answer"))
    if message:
        yield record(agent_events.MessageDelta(delta=message))
    for event in _derivation_events(result):
        yield record(event)
    yield record(
        agent_events.TurnRecordEvent(
            record=records.build(
                ctx, message=message, result=result, outcome=outcome
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
    sql_succeeded = _sql_succeeded(result)
    if (
        ctx.state.clarify_steer_issued
        and not sql_succeeded
        and "?" in message
    ):
        return "clarified"
    if _is_descriptive_scope(ctx, result):
        # A first-class tag, not reused `no_data`: it is what lets the
        # fixture assert "explorations are never recorded as
        # surrenders", and what keeps the surrender-rate metric honest
        # once exploration is common. Registered in `records.OUTCOMES`
        # in the same commit — `records.build` coerces unknown tags to
        # "error" silently, so splitting the two edits would record
        # every exploration as a failure with nothing raised.
        return "explored"
    return "answered" if sql_succeeded else "no_data"
