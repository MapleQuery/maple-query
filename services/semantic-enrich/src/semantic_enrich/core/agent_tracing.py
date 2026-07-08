"""Turn/session span plumbing + prompt-size gauge for the agent loop.

Three pieces:

- `run_turn_traced` — generator driver that owns the `agent.run_turn`
  Braintrust span around `run_turn`. Loop-agnostic: it observes the
  event stream, never the loop internals, so a future pipeline that
  emits the same events gets the same tracing for free.
- `SessionSpanMap` — conversation_id → exported session-span parent
  string. A conversation spans multiple HTTP requests, so the session
  root can't be one long-lived open span (instances recycle); instead
  the root is opened, exported, and closed on first sight of a
  conversation_id, and every turn attaches to the export string.
- `count_prompt_tokens` — tiktoken gauge for the rendered system
  prompt, measured once at startup and stamped on every turn span.

All of it is a no-op passthrough when tracing is unconfigured
(`agent_trace_sessions=False` or no Braintrust key): `run_turn_traced`
delegates straight to `run_turn` and the event stream is byte-identical
to the untraced loop.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_loop import ChatRequest, LoopDeps, run_turn
from semantic_enrich.providers import braintrust_tracing
from semantic_enrich.providers.logging import get_logger


def tracing_active(settings: Settings) -> bool:
    """True when the turn/session/tool span wrappers should run."""
    return settings.agent_trace_sessions and braintrust_tracing.is_enabled()


def count_prompt_tokens(text: str, *, model: str) -> int | None:
    """Token count of `text` under `model`'s tokenizer, or None when the
    tokenizer is unavailable (unknown model falls back to o200k_base;
    None only happens when tiktoken can't load an encoding at all, e.g.
    first run on an offline machine with a cold cache)."""
    import tiktoken

    try:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return None


class TurnObserver:
    """Accumulate turn-level output/metadata from the event stream.

    `tokens_in_per_call` / `tokens_out_per_call` are per-model-call
    deltas recovered from the cumulative `CostUpdate` totals — the
    series that shows prompt growth across loop iterations."""

    def __init__(self) -> None:
        self.turn_id = ""
        self.cached = False
        self.final_message = ""
        self.tool_call_count = 0
        self.dollars_spent = 0.0
        # "abandoned" survives only when the caller drops the iterator
        # before a terminal event arrives (client disconnect).
        self.terminal = "abandoned"
        self.tokens_in_per_call: list[int] = []
        self.tokens_out_per_call: list[int] = []
        self._prev_tokens_in = 0
        self._prev_tokens_out = 0

    def observe(self, event: agent_events.AgentEvent) -> None:
        if isinstance(event, agent_events.TurnStart):
            self.turn_id = event.turn_id
            self.cached = event.cached
        elif isinstance(event, agent_events.MessageDelta):
            self.final_message += event.delta
        elif isinstance(event, agent_events.CostUpdate):
            self.tokens_in_per_call.append(
                event.tokens_in_total - self._prev_tokens_in
            )
            self.tokens_out_per_call.append(
                event.tokens_out_total - self._prev_tokens_out
            )
            self._prev_tokens_in = event.tokens_in_total
            self._prev_tokens_out = event.tokens_out_total
        elif isinstance(event, agent_events.Done):
            self.tool_call_count = event.total_tool_calls
            self.dollars_spent = event.total_dollars
            self.terminal = "done"
        elif isinstance(event, agent_events.ErrorEvent):
            self.terminal = (
                "timeout" if event.reason == "turn_timeout" else "error"
            )

    def output(self) -> dict[str, object]:
        return {
            "final_message": self.final_message,
            "tool_call_count": self.tool_call_count,
            "dollars_spent": self.dollars_spent,
            "terminal": self.terminal,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "cached": self.cached,
            "tokens_in_per_call": self.tokens_in_per_call,
            "tokens_out_per_call": self.tokens_out_per_call,
        }


def run_turn_traced(
    *,
    request: ChatRequest,
    deps: LoopDeps,
    session_parent: str | None = None,
) -> Iterator[agent_events.AgentEvent]:
    """Drive one turn with an `agent.run_turn` span around it.

    The `with` block lives inside this generator, so the span closes on
    every exit path: normal exhaustion, an exception, or the caller
    abandoning the iterator (GeneratorExit / GC both unwind through the
    context manager). Each resumption re-enters the current-span scope
    so `wrap_openai` auto-spans attach even when the caller iterates
    from a threadpool.

    With tracing off this is a plain passthrough to `run_turn`."""
    if not tracing_active(deps.settings):
        yield from run_turn(request=request, deps=deps)
        return

    # One snapshot-hash lookup serves both the span metadata and the
    # loop's own cache-key computation.
    snapshot_provider = _memoized(deps.snapshot_hash_provider)
    snapshot_hash = snapshot_provider()

    with braintrust_tracing.trace_span(
        name="agent.run_turn",
        parent=session_parent,
        input_={
            "question": request.question,
            "conversation_id": request.conversation_id,
        },
        metadata={
            "prompt_hash": deps.prompt_hash,
            "snapshot_hash": snapshot_hash,
            "loop_impl": "v1",
            "system_prompt_tokens": deps.system_prompt_tokens,
        },
    ) as span:
        turn_deps = replace(
            deps,
            snapshot_hash_provider=snapshot_provider,
            trace_parent=span.export() or None,
        )
        observer = TurnObserver()
        inner = run_turn(request=request, deps=turn_deps)
        try:
            while True:
                with braintrust_tracing.span_current_scope(span):
                    try:
                        event = next(inner)
                    except StopIteration:
                        break
                observer.observe(event)
                yield event
        finally:
            span.log(output=observer.output(), metadata=observer.metadata())


@dataclass
class SessionSpanMap:
    """Thread-safe LRU+TTL map of conversation_id → session-span export.

    In-process and best-effort: an instance restart mid-conversation
    yields a second session root for the same conversation_id, which is
    accepted at current scale. The lock covers span creation too, so
    concurrent first requests for one conversation create exactly one
    session span."""

    max_entries: int
    ttl_seconds: int
    enabled: bool = True
    _entries: OrderedDict[str, tuple[str, float]] = field(
        default_factory=OrderedDict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(self, conversation_id: str) -> str | None:
        """Return the exported session parent for `conversation_id`,
        creating (and immediately closing) the session root on first
        sight. None when tracing is disabled."""
        if not self.enabled or not braintrust_tracing.is_enabled():
            return None
        now = time.monotonic()
        with self._lock:
            hit = self._entries.get(conversation_id)
            if hit is not None:
                export, created_at = hit
                if now - created_at <= self.ttl_seconds:
                    self._entries.move_to_end(conversation_id)
                    return export
                self._entries.pop(conversation_id, None)
            export = braintrust_tracing.export_new_span(
                name="session",
                input_={"conversation_id": conversation_id},
                metadata={"conversation_id": conversation_id},
            )
            if not export:
                return None
            self._entries[conversation_id] = (export, now)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return export


def session_span_map_from_settings(settings: Settings) -> SessionSpanMap:
    return SessionSpanMap(
        max_entries=settings.agent_trace_session_map_entries,
        ttl_seconds=settings.agent_trace_session_ttl_seconds,
        enabled=settings.agent_trace_sessions,
    )


def log_prompt_gauge(
    *, prompt: str, prompt_hash: str, settings: Settings
) -> int:
    """Measure the rendered system prompt and log the startup gauge.

    Returns the token count (0 when the tokenizer is unavailable) so
    callers can stash it on `LoopDeps.system_prompt_tokens`."""
    log = get_logger("semantic_enrich.agent_tracing")
    tokens = count_prompt_tokens(
        prompt, model=settings.openai_generation_model
    )
    if tokens is None:
        log.warning(
            "system_prompt_gauge_unavailable", prompt_hash=prompt_hash
        )
        return 0
    log.info(
        "system_prompt_gauge",
        system_prompt_tokens=tokens,
        prompt_hash=prompt_hash,
    )
    return tokens


def _memoized(fn: Callable[[], str]) -> Callable[[], str]:
    value: list[str] = []

    def wrapper() -> str:
        if not value:
            value.append(fn())
        return value[0]

    return wrapper
