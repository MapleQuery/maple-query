"""SSE encoding helpers.

The 5.1 loop yields typed events; each has its own `to_sse_frame()`.
The service adds a keep-alive comment every so often so intermediaries
(Cloud Run's edge, browser fetch buffers) don't drop the connection on
a quiet turn, and translates client aborts into loop cancellation.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from semantic_enrich.core.agent_events import AgentEvent
from starlette.requests import Request

# One `:` comment every 15s. SSE spec ignores comment lines but the
# transport keeps the connection alive.
_KEEPALIVE_INTERVAL_S = 15.0


async def sse_stream(
    events: Iterator[AgentEvent],
    *,
    request: Request | None = None,
) -> AsyncIterator[bytes]:
    """Adapt the sync event iterator into an async byte stream.

    Runs the sync loop on a worker thread so the ASGI event loop stays
    unblocked. Consumes one event per await, allowing the client's
    disconnect to short-circuit the stream between events.
    """
    loop = asyncio.get_event_loop()
    iterator: Iterator[AgentEvent] = events
    sentinel: Any = object()
    last_yield = loop.time()
    while True:
        # Best-effort client-disconnect check between events. Cloud Run
        # signals a closed socket via the ASGI receive channel; Starlette
        # exposes it through `Request.is_disconnected()`.
        if request is not None and await request.is_disconnected():
            return

        # Pull the next event on a worker thread so we don't block the
        # ASGI loop while the sync loop calls into OpenAI / BQ.
        try:
            event = await loop.run_in_executor(
                None, lambda: next(iterator, sentinel)
            )
        except Exception as exc:  # pragma: no cover - defensive
            # A raw exception escaping the loop iterator is exceptional;
            # surface it to the client as a synthetic error frame rather
            # than blowing up the ASGI middleware chain.
            yield _error_frame(str(exc))
            return

        if event is sentinel:
            return

        yield event.to_sse_frame().encode("utf-8")
        now = loop.time()
        if now - last_yield > _KEEPALIVE_INTERVAL_S:
            yield b": keepalive\n\n"
            last_yield = now
        else:
            last_yield = now


def _error_frame(message: str) -> bytes:
    """Fallback SSE frame for the "raw exception escaped the loop" path.

    Mirrors `ErrorEvent` shape so a client parser can consume it the
    same way as a loop-emitted terminal error.
    """
    import json

    payload = {
        "type": "error",
        "message": message,
        "retryable": True,
        "reason": "stream_internal_error",
    }
    return (
        "event: error\n"
        f"data: {json.dumps(payload)}\n\n"
    ).encode()
