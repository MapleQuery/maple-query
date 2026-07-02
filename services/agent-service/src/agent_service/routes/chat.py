"""`POST /chat` — SSE-framed agent turn.

Delegates the whole turn to the 5.1 loop; the route only marshals the
request body, wraps the sync event iterator into an async byte stream,
and slaps the SSE-required response headers on the way out.
"""
from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from semantic_enrich.core.agent_events import Done, ErrorEvent
from semantic_enrich.core.agent_loop import ChatRequest, run_turn

from agent_service.auth import BearerAuth
from agent_service.deps import AppState, get_app_state
from agent_service.sse import sse_stream
from agent_service.telemetry import capture

router = APIRouter()
_log = structlog.get_logger("agent_service.routes.chat")


class ChatBody(BaseModel):
    """`POST /chat` request body. Fields match 5.1 §6 verbatim.

    `history` is treated as opaque to Pydantic — the loop's
    `agent_history.validate` runs the structural checks before the model
    ever sees it, and rejects malformed shapes as an `error` SSE event.
    """

    conversation_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/chat", dependencies=[BearerAuth])
async def chat(
    body: ChatBody,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    chat_request = ChatRequest(
        conversation_id=body.conversation_id,
        history=body.history,
        question=body.question,
    )

    events = run_turn(request=chat_request, deps=state.loop_deps)

    started = time.monotonic()
    tool_calls_total = 0
    dollars_total = 0.0
    terminal_state = "unknown"

    def _observe(chunk_iter: Any) -> Any:
        """Wrap the loop iterator so the route can log a single
        `turn_finished` event after the SSE stream drains — regardless of
        whether the client stayed connected."""
        nonlocal tool_calls_total, dollars_total, terminal_state
        for event in chunk_iter:
            if isinstance(event, Done):
                tool_calls_total = event.total_tool_calls
                dollars_total = event.total_dollars
                terminal_state = "done"
            elif isinstance(event, ErrorEvent):
                terminal_state = "error"
            yield event

    async def stream() -> Any:
        try:
            async for chunk in sse_stream(_observe(events), request=request):
                yield chunk
        finally:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            _log.info(
                "turn_finished",
                conversation_id=body.conversation_id,
                tool_calls=tool_calls_total,
                dollars=round(dollars_total, 6),
                elapsed_ms=elapsed_ms,
                terminal_state=terminal_state,
            )
            capture(
                state.posthog,
                distinct_id=body.conversation_id,
                event="chat_turn_finished",
                properties={
                    "conversation_id": body.conversation_id,
                    "tool_calls": tool_calls_total,
                    "dollars": round(dollars_total, 6),
                    "elapsed_ms": elapsed_ms,
                    "terminal_state": terminal_state,
                    "question_length": len(body.question),
                    "history_length": len(body.history),
                },
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (Cloud Run's edge, nginx-style
            # intermediaries) so SSE frames flush to the client
            # immediately rather than pooling until the first N bytes.
            "X-Accel-Buffering": "no",
        },
    )
