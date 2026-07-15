"""Phase interfaces, the turn context, and the default (identity)
phase implementations for the v2 pipeline.

Design rule: phases return their events (`events: list[AgentEvent]`)
instead of yielding — the orchestrator in `pipeline.py` is the only
generator, so event ordering is auditable in one place and phases stay
unit-testable as plain functions. The research phase is the one
exception (it streams tool events as they happen); see `research.py`.

The default implementations make v2-with-stubs behave exactly like the
v1 loop: `PassthroughTriage` always classifies in-scope, `NoopMemory`
delegates to the existing history compaction and response cache
(including the cache's known replay quirks — kept bug-for-bug until
the real memory phase replaces it), and `AlwaysFitsVerifier` accepts
every answer.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_history, agent_tools
from semantic_enrich.core.agent_cache import (
    CacheEntry,
    ResponseCache,
    cache_key,
)
from semantic_enrich.core.agent_request import ChatRequest


@dataclass(frozen=True)
class SystemHint:
    """One hint line rendered into a system-role message ahead of the
    research phase. Empty in the stubbed pipeline; the reformulation
    and memory phases populate it."""

    text: str


@dataclass
class TurnTrace:
    """Cross-phase observations accumulated during research.

    Consumed by the verify/memory phases and folded into the
    `turn_record` event. Loosely typed on purpose — the record schema
    is owned by the memory phase."""

    top_similarities: list[float] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sql_runs: list[dict[str, Any]] = field(default_factory=list)
    packages_researched: list[str] = field(default_factory=list)
    columns_referenced: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TriageOutcome:
    category: str
    events: list[agent_events.AgentEvent] = field(default_factory=list)
    # Final answer text when triage terminates the turn without
    # research (out-of-scope, meta questions). None = proceed.
    short_circuit: str | None = None


@dataclass
class RecallOutcome:
    events: list[agent_events.AgentEvent] = field(default_factory=list)
    hints: list[SystemHint] = field(default_factory=list)
    # Compacted, OpenAI-ready history for the research phase. Empty on
    # the replay path (research never runs).
    history_messages: list[dict[str, Any]] = field(default_factory=list)
    # Cache-replay event stream; non-None short-circuits the turn.
    replay: Iterator[agent_events.AgentEvent] | None = None


@dataclass(frozen=True)
class Verdict:
    action: Literal["accept", "retry"]
    events: list[agent_events.AgentEvent] = field(default_factory=list)
    hints: list[SystemHint] = field(default_factory=list)


@dataclass
class ResearchResult:
    """What the research phase learned; the orchestrator owns
    termination, so this never carries terminal events itself."""

    candidate_answer: str
    terminal_reason: Literal[
        "final_answer", "budget_forced", "error", "timeout"
    ]
    sql_runs: list[dict[str, Any]] = field(default_factory=list)
    packages_cited: list[str] = field(default_factory=list)
    columns_referenced: list[str] = field(default_factory=list)
    # Set when terminal_reason == "error"; the orchestrator turns these
    # into the terminal ErrorEvent.
    error_message: str = ""
    error_retryable: bool = False
    error_reason: str | None = None


class TriagePhase(Protocol):
    def classify(self, ctx: TurnContext) -> TriageOutcome: ...


class MemoryPhase(Protocol):
    def recall(self, ctx: TurnContext) -> RecallOutcome: ...

    def commit(self, ctx: TurnContext) -> None: ...


class VerifyPhase(Protocol):
    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict: ...


@dataclass
class PipelineDeps:
    """Everything the v2 pipeline needs. Field names shared with the
    v1 deps type on purpose: the tracing wrapper reads
    settings/prompt_hash/snapshot_hash_provider/system_prompt_tokens
    and `dataclasses.replace`s trace_parent on either flavour."""

    bq: BqClient
    openai_client: OpenAIClient
    settings: Settings
    system_prompt: str
    prompt_hash: str
    cache: ResponseCache
    snapshot_hash_provider: Callable[[], str]
    system_prompt_tokens: int = 0
    trace_parent: str | None = None
    triage: TriagePhase = field(default_factory=lambda: PassthroughTriage())
    memory: MemoryPhase = field(default_factory=lambda: NoopMemory())
    verifier: VerifyPhase = field(
        default_factory=lambda: AlwaysFitsVerifier()
    )


@dataclass
class TurnContext:
    """The one mutable object owning everything cross-phase.

    Budgets and the wall-clock timeout are enforced here so every
    phase pays into the same meters — a verify-triggered research
    retry cannot exceed the turn's global budget."""

    request: ChatRequest
    deps: PipelineDeps
    turn_id: str
    started_monotonic: float
    snapshot_hash: str
    state: agent_tools.LoopState
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    tool_call_count: int = 0
    verify_retries_used: int = 0
    reformulations_used: int = 0
    turn_start_emitted: bool = False
    history_messages: list[dict[str, Any]] = field(default_factory=list)
    trace: TurnTrace = field(default_factory=TurnTrace)
    # Every event yielded by the orchestrator, recorded for the cache
    # write at commit time (the v1 loop's `events` list, centralized).
    events: list[agent_events.AgentEvent] = field(default_factory=list)

    @classmethod
    def begin(cls, *, request: ChatRequest, deps: PipelineDeps) -> TurnContext:
        turn_id = uuid.uuid4().hex
        return cls(
            request=request,
            deps=deps,
            turn_id=turn_id,
            started_monotonic=time.monotonic(),
            snapshot_hash=deps.snapshot_hash_provider(),
            state=agent_tools.LoopState(
                conversation_id=request.conversation_id,
                turn_id=turn_id,
                question=request.question,
            ),
        )

    # ── shared meters ──

    def charge_model_call(
        self, *, tokens_in: int, tokens_out: int
    ) -> agent_events.CostUpdate:
        self.tokens_in_total += tokens_in
        self.tokens_out_total += tokens_out
        return agent_events.CostUpdate(
            tokens_in_total=self.tokens_in_total,
            tokens_out_total=self.tokens_out_total,
            dollars_spent=round(self.dollars_spent, 6),
        )

    def charge_tool_calls(self, n: int) -> None:
        self.tool_call_count += n
        # Mirror into the tool-side state so tools observing the count
        # see the same meter.
        self.state.tool_call_count = self.tool_call_count

    def remaining_tool_calls(self) -> int:
        return self.deps.settings.agent_max_tool_calls - self.tool_call_count

    def check_wallclock(self) -> tuple[int, int] | None:
        """Return `(elapsed_ms, cap_ms)` when the turn has blown its
        wall-clock budget, None otherwise."""
        elapsed_ms = int((time.monotonic() - self.started_monotonic) * 1000)
        cap_ms = self.deps.settings.agent_turn_timeout_seconds * 1000
        if elapsed_ms > cap_ms:
            return elapsed_ms, cap_ms
        return None

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_monotonic) * 1000)

    def retries_remaining(self) -> bool:
        return (
            self.verify_retries_used
            < self.deps.settings.agent_verify_max_retries
        )

    @property
    def dollars_spent(self) -> float:
        settings = self.deps.settings
        return (
            self.tokens_in_total * settings.agent_model_input_rate / 1000.0
            + self.tokens_out_total
            * settings.agent_model_output_rate
            / 1000.0
        )


# ── default phase implementations ──


class PassthroughTriage:
    """Every question is in scope; no events, never short-circuits."""

    def classify(self, ctx: TurnContext) -> TriageOutcome:
        return TriageOutcome(category="in_scope")


class AlwaysFitsVerifier:
    """Every answer fits; never requests a retry."""

    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict:
        return Verdict(action="accept")


class NoopMemory:
    """v1-compatible memory: recall = response-cache lookup + history
    compaction, commit = the v1 cache write. No plan hints.

    Delegates to the existing `agent_cache` / `agent_history` modules
    so v2-with-stubs matches the v1 loop exactly — including the
    cache's replay quirks (the recorded turn_start replays alongside
    the fresh one) — until the real memory phase replaces it. The
    commit-side cache write lives here rather than being a true no-op
    because dropping it would silently diverge from v1 across turns.
    """

    def recall(self, ctx: TurnContext) -> RecallOutcome:
        deps = ctx.deps
        key = cache_key(
            question=ctx.request.question,
            prompt_hash=deps.prompt_hash,
            snapshot_hash=ctx.snapshot_hash,
        )
        cached = deps.cache.get(key)
        if cached is not None:
            return RecallOutcome(
                replay=_replay_events(
                    turn_id=ctx.turn_id,
                    conversation_id=ctx.request.conversation_id,
                    cached=cached,
                    key=key,
                    settings=deps.settings,
                )
            )
        compacted = agent_history.compact(
            history=ctx.request.history,
            settings=deps.settings,
            openai_client=deps.openai_client,
        )
        return RecallOutcome(history_messages=compacted.messages)

    def commit(self, ctx: TurnContext) -> None:
        # Only successful, budget-clean turns get cached; a turn that
        # errored out would just fail again on replay.
        if not ctx.events:
            return
        if not isinstance(ctx.events[-1], agent_events.Done):
            return
        ctx.deps.cache.put(
            cache_key(
                question=ctx.request.question,
                prompt_hash=ctx.deps.prompt_hash,
                snapshot_hash=ctx.snapshot_hash,
            ),
            events=ctx.events,
            prompt_hash=ctx.deps.prompt_hash,
            snapshot_hash=ctx.snapshot_hash,
        )


def _replay_events(
    *,
    turn_id: str,
    conversation_id: str,
    cached: CacheEntry,
    key: str,
    settings: Settings,
) -> Iterator[agent_events.AgentEvent]:
    yield agent_events.TurnStart(
        conversation_id=conversation_id,
        turn_id=turn_id,
        cached=True,
    )
    yield agent_events.CacheHit(cache_key_prefix=key[:12])
    delay = max(0, settings.agent_cache_replay_delay_ms) / 1000.0
    for payload in cached.events:
        event_type = payload.get("type")
        if event_type in ("turn_start", "done"):
            # Rewrite the turn_id so the replay matches this turn's id
            # rather than the recording's.
            payload = {**payload, "turn_id": turn_id}
            if event_type == "turn_start":
                payload["cached"] = True
                payload["conversation_id"] = conversation_id
        frame = _dict_to_frame(payload)
        if frame is None:
            continue
        if delay:
            time.sleep(delay)
        yield frame


def _dict_to_frame(
    payload: dict[str, Any],
) -> agent_events.AgentEvent | None:
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return None
    body = {k: v for k, v in payload.items() if k != "type"}
    cls = agent_events._EVENT_CLASSES.get(event_type)
    if cls is None:
        return None
    try:
        return cls(**body)  # type: ignore[return-value]
    except TypeError:
        return None
