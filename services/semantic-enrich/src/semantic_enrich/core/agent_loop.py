"""The 5.1 agent loop.

Thin driver around OpenAI tool-calling. Owns turn budgets, cost
tracking, cache reads/writes, and the event stream. Delegates actual
work to `agent_tools`, `agent_history`, `agent_cache`, and the 4.6
retrieval / guard / executor modules.

Sync by design: the existing stack (BQ client, OpenAI client, retry
policy) is all sync. Parallel tool calls fan out via a bounded
`ThreadPoolExecutor` rather than `asyncio.gather` so we don't split
the codebase's execution model.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import ChatToolCall, OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_history, agent_tools
from semantic_enrich.core.agent_cache import (
    CacheEntry,
    ResponseCache,
    cache_key,
)
from semantic_enrich.providers.logging import get_logger


@dataclass(frozen=True)
class ChatRequest:
    """One `POST /chat`-shaped input.

    `history` follows OpenAI's chat message schema: role, content,
    tool_calls, tool_call_id. `question` is appended by the loop as
    the current turn's user message — the client sends it separately
    so the loop can key the cache off just the current question.
    """

    conversation_id: str
    history: list[dict[str, Any]]
    question: str


@dataclass
class LoopDeps:
    """Everything the loop needs. Constructed once per process (except
    the cache, which is a shared in-memory LRU) and passed in from the
    CLI or HTTP surface."""

    bq: BqClient
    openai_client: OpenAIClient
    settings: Settings
    system_prompt: str
    prompt_hash: str
    cache: ResponseCache
    snapshot_hash_provider: Callable[[], str]


@dataclass
class TurnOutcome:
    """The concrete result of a turn, in addition to the events.

    Callers (CLI, HTTP) usually only care about the streamed events;
    this is what the harness (5.4) uses to grade a turn without
    re-parsing the event log."""

    turn_id: str
    final_message: str
    tool_call_count: int
    dollars_spent: float
    events: list[agent_events.AgentEvent] = field(default_factory=list)
    cache_hit: bool = False


def load_system_prompt(path: Path, settings: Settings) -> tuple[str, str]:
    """Render the system prompt template once. Returns `(text, sha256)`.

    Hash feeds the cache key. Prompt edits invalidate on redeploy."""
    if not path.exists():
        raise RuntimeError(f"agent system prompt missing: {path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(path.name)
    rendered = template.render(
        allowed_datasets=list(settings.eval_allowed_datasets),
        row_limit=settings.eval_row_limit,
    )
    tool_defs = json.dumps(agent_tools.tool_schemas(), sort_keys=True)
    digest = hashlib.sha256(
        (rendered + "\n---\n" + tool_defs).encode("utf-8")
    ).hexdigest()
    return rendered, digest


def make_snapshot_hash_provider(
    bq: BqClient, settings: Settings
) -> Callable[[], str]:
    """Return a callable that emits the current warehouse snapshot hash.

    Refreshed by the caller on `agent_snapshot_refresh_seconds`; the
    provider itself just runs the two MAX(loaded_at) queries."""
    project_id = settings.gcp_project_id
    if not project_id:
        # No project → no meaningful snapshot; the cache key falls back
        # to a stable literal, effectively disabling snapshot
        # invalidation. Only expected under `--dry-run`.
        return lambda: "no-snapshot"
    sql_ds = (
        f"SELECT CAST(MAX(loaded_at) AS STRING) AS max_ts "
        f"FROM `{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_datasets_table}`"
    )
    sql_cols = (
        f"SELECT CAST(MAX(loaded_at) AS STRING) AS max_ts "
        f"FROM `{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_columns_table}`"
    )

    def _provider() -> str:
        try:
            ds_rows = list(bq.query_rows(sql_ds))
            col_rows = list(bq.query_rows(sql_cols))
        except Exception:  # pragma: no cover - defensive
            return "unknown-snapshot"
        ds_ts = ds_rows[0].get("max_ts") if ds_rows else None
        col_ts = col_rows[0].get("max_ts") if col_rows else None
        raw = f"{ds_ts}||{col_ts}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return _provider


def run_turn(
    *,
    request: ChatRequest,
    deps: LoopDeps,
) -> Iterator[agent_events.AgentEvent]:
    """Run one turn, yielding events lazily.

    Callers (CLI / HTTP) consume the iterator directly and either
    print / SSE-frame each event. The turn's post-hoc `TurnOutcome`
    is discoverable through the terminal event (`done` or `error`);
    a caller that needs the un-serialised outcome can use
    `run_turn_collected`."""
    log = get_logger("semantic_enrich.agent_loop")
    turn_id = uuid.uuid4().hex
    turn_started = time.monotonic()
    events: list[agent_events.AgentEvent] = []

    def emit(event: agent_events.AgentEvent) -> None:
        events.append(event)

    # 1. Validate + compact history.
    try:
        agent_history.validate(request.history, settings=deps.settings)
    except agent_history.InvalidHistoryError as exc:
        yield agent_events.TurnStart(
            conversation_id=request.conversation_id,
            turn_id=turn_id,
            cached=False,
        )
        yield agent_events.ErrorEvent(
            message=str(exc),
            retryable=False,
            reason="invalid_history",
        )
        return

    # 2. Cache lookup.
    snapshot_hash = deps.snapshot_hash_provider()
    key = cache_key(
        question=request.question,
        prompt_hash=deps.prompt_hash,
        snapshot_hash=snapshot_hash,
    )
    cached = deps.cache.get(key)
    if cached is not None:
        yield from _replay_cache(
            turn_id=turn_id,
            conversation_id=request.conversation_id,
            cached=cached,
            key=key,
            settings=deps.settings,
        )
        return

    yield agent_events.TurnStart(
        conversation_id=request.conversation_id,
        turn_id=turn_id,
        cached=False,
    )
    events.append(
        agent_events.TurnStart(
            conversation_id=request.conversation_id,
            turn_id=turn_id,
            cached=False,
        )
    )

    compacted = agent_history.compact(
        history=request.history,
        settings=deps.settings,
        openai_client=deps.openai_client,
    )

    state = agent_tools.LoopState(
        conversation_id=request.conversation_id,
        turn_id=turn_id,
        question=request.question,
    )
    ctx = agent_tools.ToolContext(
        bq=deps.bq,
        openai_client=deps.openai_client,
        settings=deps.settings,
        state=state,
        emit=emit,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": deps.system_prompt},
        *compacted.messages,
        {"role": "user", "content": request.question},
    ]

    tools = agent_tools.tool_schemas()
    budget_forced = False

    while True:
        # 3a. Wall-clock timeout.
        elapsed_ms = int((time.monotonic() - turn_started) * 1000)
        cap_ms = deps.settings.agent_turn_timeout_seconds * 1000
        if elapsed_ms > cap_ms:
            timeout_event = agent_events.TurnTimeout(
                elapsed_ms=elapsed_ms, cap_ms=cap_ms
            )
            events.append(timeout_event)
            yield timeout_event
            error_event = agent_events.ErrorEvent(
                message="turn wall-clock exceeded",
                retryable=False,
                reason="turn_timeout",
            )
            events.append(error_event)
            yield error_event
            return

        # 3b. Model call.
        try:
            completion = deps.openai_client.chat_with_tools(
                messages=messages,
                tools=tools,
                model=deps.settings.openai_generation_model,
                temperature=deps.settings.openai_generation_temperature,
                max_tokens=deps.settings.openai_generation_max_tokens,
                parallel_tool_calls=(
                    deps.settings.agent_parallel_tool_calls > 1
                ),
            )
        except Exception as exc:
            log.error("chat_with_tools_failed", error=str(exc))
            retryable = _is_openai_retryable(exc)
            error_event = agent_events.ErrorEvent(
                message=str(exc),
                retryable=retryable,
                reason="openai_error",
            )
            events.append(error_event)
            yield error_event
            return

        state.tokens_in_total += completion.tokens_in
        state.tokens_out_total += completion.tokens_out
        state.dollars_spent = _dollars(state, deps.settings)
        cost_event = agent_events.CostUpdate(
            tokens_in_total=state.tokens_in_total,
            tokens_out_total=state.tokens_out_total,
            dollars_spent=round(state.dollars_spent, 6),
        )
        events.append(cost_event)
        yield cost_event

        # 3c. Terminal — model returned final text.
        if not completion.tool_calls:
            content = completion.content or ""
            if content:
                delta_event = agent_events.MessageDelta(delta=content)
                events.append(delta_event)
                yield delta_event
            done_event = agent_events.Done(
                turn_id=turn_id,
                total_tool_calls=state.tool_call_count,
                total_dollars=round(state.dollars_spent, 6),
                elapsed_ms=int((time.monotonic() - turn_started) * 1000),
            )
            events.append(done_event)
            yield done_event
            _maybe_cache(
                deps=deps,
                key=key,
                events=events,
                snapshot_hash=snapshot_hash,
            )
            return

        # 3d. Tool calls. Append the assistant message verbatim so the
        # next iteration sees its own tool_calls.
        messages.append(
            {
                "role": "assistant",
                "content": completion.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in completion.tool_calls
                ],
            }
        )

        # 3e. Budget check BEFORE executing the batch. Each call in the
        # batch consumes one slot; refuse the whole batch when it would
        # push us past the cap.
        remaining_calls = (
            deps.settings.agent_max_tool_calls - state.tool_call_count
        )
        if remaining_calls <= 0 and not budget_forced:
            over_event = agent_events.BudgetExceeded(
                which="tool_calls",
                value=state.tool_call_count,
                cap=deps.settings.agent_max_tool_calls,
            )
            events.append(over_event)
            yield over_event
            # Force one final assistant turn with a synthetic user
            # message. The system prompt covers the "stop and answer"
            # path; here we make the situation explicit.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You've reached the tool-call budget. Stop "
                        "calling tools and produce the best answer "
                        "you can from what you've already learned. "
                        "If you don't have enough evidence, say so."
                    ),
                }
            )
            budget_forced = True
            continue

        # 3f. Execute the batch (capped concurrency).
        batch = completion.tool_calls[:remaining_calls]
        if len(batch) < len(completion.tool_calls):
            over_event = agent_events.BudgetExceeded(
                which="tool_calls",
                value=state.tool_call_count + len(completion.tool_calls),
                cap=deps.settings.agent_max_tool_calls,
            )
            events.append(over_event)
            yield over_event
        for tool_msg, tool_events in _execute_batch(
            batch=batch, ctx=ctx
        ):
            for te in tool_events:
                events.append(te)
                yield te
            messages.append(tool_msg)
        state.tool_call_count += len(batch)

        # 3g. If we forced budget above, this iteration was the final
        # answer path — loop back to consume it.
        if budget_forced:
            continue


def run_turn_collected(
    *, request: ChatRequest, deps: LoopDeps
) -> TurnOutcome:
    """Convenience helper — drain `run_turn` into a `TurnOutcome`.

    Not what the CLI uses (it streams events), but the harness (5.4)
    and unit tests find the collected form easier."""
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
    return TurnOutcome(
        turn_id=turn_id,
        final_message=final,
        tool_call_count=tool_calls,
        dollars_spent=dollars,
        events=events,
        cache_hit=cache_hit,
    )


def _execute_batch(
    *,
    batch: list[ChatToolCall],
    ctx: agent_tools.ToolContext,
) -> Iterable[
    tuple[dict[str, Any], list[agent_events.AgentEvent]]
]:
    """Run a batch of tool calls concurrently. Returns each
    `(tool-result-message, events-captured-during-that-call)` pair in
    the same order the model requested them.

    Concurrency is capped at settings.agent_parallel_tool_calls but
    also constrained by the safety rule that two `run_sql` calls in
    parallel is disallowed (both count against the same budget and
    could race)."""
    if not batch:
        return []
    # Two run_sql calls in the same batch → serialize the whole batch.
    # Anything else runs concurrently up to the cap.
    run_sql_count = sum(1 for tc in batch if tc.name == "run_sql")
    max_workers = 1 if run_sql_count > 1 else max(
        1, ctx.settings.agent_parallel_tool_calls
    )
    results: list[
        tuple[dict[str, Any], list[agent_events.AgentEvent]]
    ] = []
    if max_workers == 1 or len(batch) == 1:
        for tc in batch:
            results.append(_run_one(tc=tc, ctx=ctx))
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Each call gets its own event buffer so events stay grouped
        # by tool call in the output stream.
        futures = [pool.submit(_run_one, tc=tc, ctx=ctx) for tc in batch]
        for fut in futures:
            results.append(fut.result())
    return results


def _run_one(
    *,
    tc: ChatToolCall,
    ctx: agent_tools.ToolContext,
) -> tuple[dict[str, Any], list[agent_events.AgentEvent]]:
    """Execute one tool call. Returns
    `(assistant-facing-tool-message, events-emitted)`.

    The events buffer is per-call so the caller can preserve order in
    the outer event stream."""
    captured: list[agent_events.AgentEvent] = []

    def local_emit(event: agent_events.AgentEvent) -> None:
        captured.append(event)

    scoped_ctx = agent_tools.ToolContext(
        bq=ctx.bq,
        openai_client=ctx.openai_client,
        settings=ctx.settings,
        state=ctx.state,
        emit=local_emit,
    )
    try:
        result = agent_tools.dispatch(
            ctx=scoped_ctx, tool_name=tc.name, args=tc.arguments
        )
        tool_msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, default=str),
        }
        return tool_msg, captured
    except agent_tools.InvalidToolArgsError as exc:
        err_event = agent_events.ToolError(tool=tc.name, message=str(exc))
        captured.append(err_event)
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(
                {"status": "tool_error", "message": str(exc)}
            ),
        }
        return tool_msg, captured
    except Exception as exc:  # pragma: no cover - defensive
        err_event = agent_events.ToolError(
            tool=tc.name, message=f"internal_error: {exc}"
        )
        captured.append(err_event)
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(
                {"status": "tool_error", "message": str(exc)}
            ),
        }
        return tool_msg, captured


def _replay_cache(
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
        # Reconstruct typed event from the recorded dict via the sse
        # round-trip helper — keeps event schema evolution isolated.
        event_type = payload.get("type")
        if event_type in ("turn_start", "done"):
            # Rewrite the turn_id so the replay matches this turn's id
            # rather than the recording's. Everything else is
            # semantically identical.
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


def _dict_to_frame(payload: dict[str, Any]) -> agent_events.AgentEvent | None:
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return None
    body = {k: v for k, v in payload.items() if k != "type"}
    cls_map = agent_events._EVENT_CLASSES
    cls = cls_map.get(event_type)
    if cls is None:
        return None
    try:
        return cls(**body)  # type: ignore[return-value]
    except TypeError:
        return None


def _maybe_cache(
    *,
    deps: LoopDeps,
    key: str,
    events: list[agent_events.AgentEvent],
    snapshot_hash: str,
) -> None:
    # Only successful, budget-clean turns end up cached. A turn that
    # blew the wall-clock or errored out is not worth memoising —
    # replaying it would just fail again.
    if not events:
        return
    terminal = events[-1]
    if not isinstance(terminal, agent_events.Done):
        return
    deps.cache.put(
        key,
        events=events,
        prompt_hash=deps.prompt_hash,
        snapshot_hash=snapshot_hash,
    )


def _dollars(state: agent_tools.LoopState, settings: Settings) -> float:
    return (
        state.tokens_in_total * settings.agent_model_input_rate / 1000.0
        + state.tokens_out_total * settings.agent_model_output_rate / 1000.0
    )


def _is_openai_retryable(exc: BaseException) -> bool:
    # Match the shape from `providers/openai_retry.py` without
    # re-importing the concrete classes; the loop treats every OpenAI
    # failure the retry policy already exhausted as non-retryable, and
    # any other transport failure as retryable so the client can retry
    # the turn.
    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return True
    return "timeout" in msg or "temporarily" in msg
