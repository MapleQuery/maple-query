"""Braintrust tracing: init + OpenAI client wrapper.

Braintrust ships an OpenAI SDK wrapper that logs every chat/embedding
call as a span. We call `init_logger` once at process start when a key
is configured, then wrap the raw `OpenAI` client through
`wrap_openai_client` before it's handed to `RealOpenAIClient`. If no
key is set the wrapper is a no-op — the client and every call site
behave exactly as before, so unconfigured environments (unit tests,
laptop dev without a Braintrust account) don't pay any cost.
"""
from __future__ import annotations

from typing import Any, TypeVar

_configured = False
_enabled = False

_ClientT = TypeVar("_ClientT")


def configure_braintrust(
    *,
    api_key: str | None,
    project: str,
    enabled: bool = True,
) -> bool:
    """Initialize the Braintrust logger. Idempotent.

    Returns True when tracing is active after the call, False when it's
    disabled (missing key, `enabled=False`, or SDK import failure). The
    return value is what the caller logs so operators can tell from
    startup output whether traces are flowing.
    """
    global _configured, _enabled
    if _configured:
        return _enabled
    _configured = True
    if not enabled or not api_key:
        _enabled = False
        return False
    try:
        from braintrust import init_logger
    except ImportError:
        _enabled = False
        return False
    init_logger(project=project, api_key=api_key)
    _enabled = True
    return True


def wrap_openai_client(client: _ClientT) -> _ClientT:
    """Return the OpenAI client wrapped for Braintrust tracing, or
    the client unchanged if tracing is disabled.

    Called by `RealOpenAIClient.__init__` right after it builds the raw
    `openai.OpenAI` instance. Safe to call before `configure_braintrust`
    — an unconfigured or disabled tracer returns the client as-is. The
    generic return type preserves the input's type so callers keep the
    same static surface whether tracing is on or off.
    """
    if not _enabled:
        return client
    try:
        from braintrust import wrap_openai
    except ImportError:
        return client
    wrapped: Any = wrap_openai(client)
    return wrapped  # type: ignore[no-any-return]


def is_enabled() -> bool:
    return _enabled
