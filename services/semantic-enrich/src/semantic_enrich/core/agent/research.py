"""The research phase: the model/tool loop extracted from the v1
driver, unchanged in behaviour except that

- budgets and the wall-clock are read/charged through `TurnContext`,
  so a verify-triggered re-entry pays into the same meters;
- `hints` render as one system-role message after the main system
  prompt (empty in the stubbed pipeline);
- retrieval confidence and per-call outcomes land in `ctx.trace`;
- it returns a `ResearchResult` instead of terminating the turn — the
  orchestrator owns the terminal `done`/`error`.

The batch-execution concurrency machinery (serialize when a batch has
more than one `run_sql`, cap at `agent_parallel_tool_calls`, per-call
event buffers, contextvars copy for tracing) carries over verbatim.
"""
from __future__ import annotations

import contextvars
import json
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Literal

from semantic_enrich.clients.openai import ChatToolCall
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent.phases import (
    ResearchResult,
    SystemHint,
    TurnContext,
)
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent.research")


def run(
    ctx: TurnContext, *, hints: list[SystemHint]
) -> Generator[agent_events.AgentEvent, None, ResearchResult]:
    """Drive model calls and tool batches until the model produces a
    final message, a budget forces one, or the turn errors/times out.

    Yields the same event stream the v1 loop produced for this segment
    (cost updates, tool events, budget/timeout markers). The final
    `message_delta`/`done` are NOT emitted here."""
    deps = ctx.deps
    tool_ctx = agent_tools.ToolContext(
        bq=deps.bq,
        openai_client=deps.openai_client,
        settings=deps.settings,
        state=ctx.state,
        emit=lambda _event: None,
        trace_parent=deps.trace_parent,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": deps.system_prompt},
    ]
    if hints:
        messages.append(
            {
                "role": "system",
                "content": "\n".join(h.text for h in hints),
            }
        )
    messages.extend(ctx.history_messages)
    messages.append({"role": "user", "content": ctx.request.question})

    tools = agent_tools.tool_schemas()
    budget_forced = False

    while True:
        over = ctx.check_wallclock()
        if over is not None:
            elapsed_ms, cap_ms = over
            yield agent_events.TurnTimeout(
                elapsed_ms=elapsed_ms, cap_ms=cap_ms
            )
            return _result(ctx, answer="", reason="timeout")

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
            _LOG.error("chat_with_tools_failed", error=str(exc))
            result = _result(ctx, answer="", reason="error")
            result.error_message = str(exc)
            result.error_retryable = _is_openai_retryable(exc)
            result.error_reason = "openai_error"
            return result

        yield ctx.charge_model_call(
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
        )

        # Terminal — model returned final text.
        if not completion.tool_calls:
            return _result(
                ctx,
                answer=completion.content or "",
                reason="budget_forced" if budget_forced else "final_answer",
            )

        # Append the assistant message verbatim so the next iteration
        # sees its own tool_calls.
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

        # Budget check BEFORE executing the batch. Refused calls still
        # receive a tool message — the OpenAI API rejects the whole
        # conversation unless every tool_call_id gets a response.
        remaining_calls = ctx.remaining_tool_calls()
        if remaining_calls <= 0 and not budget_forced:
            yield agent_events.BudgetExceeded(
                which="tool_calls",
                value=ctx.tool_call_count,
                cap=deps.settings.agent_max_tool_calls,
            )
            for tc in completion.tool_calls:
                messages.append(_budget_refusal_msg(tc))
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

        # Execute the batch (capped concurrency). Calls past the
        # remaining budget are refused with a tool message, not dropped.
        batch = completion.tool_calls[:remaining_calls]
        refused = completion.tool_calls[remaining_calls:]
        if refused:
            yield agent_events.BudgetExceeded(
                which="tool_calls",
                value=ctx.tool_call_count + len(completion.tool_calls),
                cap=deps.settings.agent_max_tool_calls,
            )
        for tc, tool_msg, tool_events, result_payload in _execute_batch(
            batch=batch, ctx=tool_ctx
        ):
            yield from tool_events
            messages.append(tool_msg)
            _record_trace(ctx, tc=tc, result=result_payload)
        for tc in refused:
            messages.append(_budget_refusal_msg(tc))
        ctx.charge_tool_calls(len(batch))

        # If budget was forced above, the next iteration consumes the
        # forced final answer.
        if budget_forced:
            continue


def _result(
    ctx: TurnContext,
    *,
    answer: str,
    reason: Literal["final_answer", "budget_forced", "error", "timeout"],
) -> ResearchResult:
    return ResearchResult(
        candidate_answer=answer,
        terminal_reason=reason,
        sql_runs=list(ctx.trace.sql_runs),
        packages_cited=list(ctx.trace.packages_researched),
        columns_referenced=list(ctx.trace.columns_referenced),
    )


def _record_trace(
    ctx: TurnContext,
    *,
    tc: ChatToolCall,
    result: dict[str, Any] | None,
) -> None:
    """Fold one tool call's outcome into the turn trace. Best-effort:
    a tool error leaves `result` as None and only the call is noted."""
    status = str(result.get("status", "ok")) if result else "tool_error"
    ctx.trace.tool_calls.append({"tool": tc.name, "status": status})
    if result is None:
        return
    if tc.name == "search_datasets":
        top = result.get("top_similarity")
        if isinstance(top, int | float):
            ctx.trace.top_similarities.append(float(top))
    elif tc.name == "list_documents":
        for pid in tc.arguments.get("package_ids") or []:
            if (
                isinstance(pid, str)
                and pid not in ctx.trace.packages_researched
            ):
                ctx.trace.packages_researched.append(pid)
    elif tc.name == "run_sql":
        sql = str(tc.arguments.get("sql", ""))
        ctx.trace.sql_runs.append(
            {
                "sql": sql,
                "status": status,
                "row_count": result.get("row_count"),
                "null_ratio_warning": result.get("null_ratio_warning"),
            }
        )
        for col in sorted(agent_tools._extract_json_path_columns(sql)):
            if col not in ctx.trace.columns_referenced:
                ctx.trace.columns_referenced.append(col)


def _budget_refusal_msg(tc: ChatToolCall) -> dict[str, Any]:
    """Tool response for a call refused by the tool-call budget."""
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(
            {
                "status": "budget_exceeded",
                "message": (
                    "Tool-call budget exhausted; this call was not "
                    "executed. Answer from what you already have."
                ),
            }
        ),
    }


def _execute_batch(
    *,
    batch: list[ChatToolCall],
    ctx: agent_tools.ToolContext,
) -> Iterable[
    tuple[
        ChatToolCall,
        dict[str, Any],
        list[agent_events.AgentEvent],
        dict[str, Any] | None,
    ]
]:
    """Run a batch of tool calls concurrently; results come back in
    request order. Two `run_sql` calls in one batch serialize the whole
    batch (both charge the same budget and could race)."""
    if not batch:
        return []
    run_sql_count = sum(1 for tc in batch if tc.name == "run_sql")
    max_workers = 1 if run_sql_count > 1 else max(
        1, ctx.settings.agent_parallel_tool_calls
    )
    results: list[
        tuple[
            ChatToolCall,
            dict[str, Any],
            list[agent_events.AgentEvent],
            dict[str, Any] | None,
        ]
    ] = []
    if max_workers == 1 or len(batch) == 1:
        for tc in batch:
            results.append(_run_one(tc=tc, ctx=ctx))
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Per-call event buffers keep events grouped by call in the
        # output stream; the contextvars copy makes the tracing
        # current-span visible on pool threads.
        futures = [
            pool.submit(
                contextvars.copy_context().run,
                partial(_run_one, tc=tc, ctx=ctx),
            )
            for tc in batch
        ]
        for fut in futures:
            results.append(fut.result())
    return results


def _run_one(
    *,
    tc: ChatToolCall,
    ctx: agent_tools.ToolContext,
) -> tuple[
    ChatToolCall,
    dict[str, Any],
    list[agent_events.AgentEvent],
    dict[str, Any] | None,
]:
    """Execute one tool call. Returns the call, the assistant-facing
    tool message, the events it emitted, and the raw result payload
    (None when the call errored) for trace capture."""
    captured: list[agent_events.AgentEvent] = []

    def local_emit(event: agent_events.AgentEvent) -> None:
        captured.append(event)

    scoped_ctx = agent_tools.ToolContext(
        bq=ctx.bq,
        openai_client=ctx.openai_client,
        settings=ctx.settings,
        state=ctx.state,
        emit=local_emit,
        trace_parent=ctx.trace_parent,
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
        return tc, tool_msg, captured, result
    except agent_tools.InvalidToolArgsError as exc:
        captured.append(
            agent_events.ToolError(tool=tc.name, message=str(exc))
        )
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(
                {"status": "tool_error", "message": str(exc)}
            ),
        }
        return tc, tool_msg, captured, None
    except Exception as exc:  # pragma: no cover - defensive
        captured.append(
            agent_events.ToolError(
                tool=tc.name, message=f"internal_error: {exc}"
            )
        )
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(
                {"status": "tool_error", "message": str(exc)}
            ),
        }
        return tc, tool_msg, captured, None


def _is_openai_retryable(exc: BaseException) -> bool:
    # Same shape-matching as the v1 loop: rate limits and transport
    # blips are retryable, everything the retry policy already
    # exhausted is not.
    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return True
    return "timeout" in msg or "temporarily" in msg
