"""PostHog server-side capture helpers.

The service fires two product-analytics events:

- `chat_turn_finished` — one per drained SSE stream, with tool-call
  count, dollars, elapsed ms, and the terminal state.
- `sql_run_finished` — one per `POST /sql/run`, with the guard status
  and bytes billed.

A missing key silences everything: `build_posthog_client` returns
`None` and `capture` is a no-op. That keeps unit tests and unconfigured
local runs quiet without conditional call sites everywhere.
"""
from __future__ import annotations

from typing import Any

import structlog

from agent_service.config import AgentServiceSettings

_log = structlog.get_logger("agent_service.telemetry")


def build_posthog_client(settings: AgentServiceSettings) -> Any:
    """Return a configured `posthog.Posthog` client, or `None` when no
    API key is set.

    Called once at startup by `build_app_state` and stored on
    `AppState.posthog`. Route handlers pass it to `capture(...)`.
    """
    if settings.posthog_api_key is None:
        return None
    try:
        from posthog import Posthog
    except ImportError:
        _log.warning("posthog_sdk_missing")
        return None
    return Posthog(  # type: ignore[no-untyped-call]
        project_api_key=settings.posthog_api_key.get_secret_value(),
        host=settings.posthog_host,
    )


def capture(
    client: Any,
    *,
    distinct_id: str,
    event: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Fire an event when a client is configured.

    PostHog's SDK swallows transport errors internally, but we still
    guard the call so an SDK regression can't take a request down.
    """
    if client is None:
        return
    try:
        client.capture(
            distinct_id=distinct_id,
            event=event,
            properties=properties or {},
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("posthog_capture_failed", event=event, error=str(exc))
