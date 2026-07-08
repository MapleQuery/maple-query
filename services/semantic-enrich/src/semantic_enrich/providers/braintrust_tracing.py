"""Braintrust tracing: init, OpenAI client wrapper, and span helpers.

Braintrust ships an OpenAI SDK wrapper that logs every chat/embedding
call as a span. We call `init_logger` once at process start when a key
is configured, then wrap the raw `OpenAI` client through
`wrap_openai_client` before it's handed to `RealOpenAIClient`. If no
key is set the wrapper is a no-op — the client and every call site
behave exactly as before, so unconfigured environments (unit tests,
laptop dev without a Braintrust account) don't pay any cost.

The span helpers (`trace_span`, `export_new_span`, `span_current_scope`)
are the single gate the agent loop's turn/session/tool spans go through.
They all collapse to no-ops when tracing is off, so callers never need
their own `is_enabled()` checks around individual span operations.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

_configured = False
_enabled = False


class SpanLike(Protocol):
    """The slice of Braintrust's `Span` surface the agent loop uses."""

    def log(self, **event: Any) -> None: ...

    def export(self) -> str: ...

    def set_current(self) -> None: ...

    def unset_current(self) -> None: ...


class NoopSpan:
    """Inert stand-in yielded when tracing is disabled."""

    def log(self, **event: Any) -> None:
        return None

    def export(self) -> str:
        return ""

    def set_current(self) -> None:
        return None

    def unset_current(self) -> None:
        return None


NOOP_SPAN = NoopSpan()


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


def wrap_openai_client[ClientT](client: ClientT) -> ClientT:
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


@contextmanager
def trace_span(
    *,
    name: str,
    parent: str | None = None,
    input_: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[SpanLike]:
    """Open a Braintrust span as a context manager.

    `parent` is an exported span string (distributed-tracing form) —
    passing one attaches the new span under it even if the parent was
    created in another request or process and has already closed.
    Yields `NOOP_SPAN` when tracing is disabled so call sites don't
    branch."""
    if not _enabled:
        yield NOOP_SPAN
        return
    try:
        from braintrust import start_span
    except ImportError:
        yield NOOP_SPAN
        return
    kwargs: dict[str, Any] = {}
    if input_ is not None:
        kwargs["input"] = input_
    if metadata is not None:
        kwargs["metadata"] = metadata
    with start_span(name=name, parent=parent, **kwargs) as span:
        yield span


def export_new_span(
    *,
    name: str,
    input_: Any = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a span, export it, and close it. Returns the export string
    ("" when tracing is disabled).

    This is the session-root pattern: the span itself lives only for
    this call, but Braintrust allows attaching children to an exported
    span after the parent closed, so the export string works as a
    durable parent handle across requests."""
    if not _enabled:
        return ""
    try:
        from braintrust import start_span
    except ImportError:
        return ""
    kwargs: dict[str, Any] = {}
    if input_ is not None:
        kwargs["input"] = input_
    if metadata is not None:
        kwargs["metadata"] = metadata
    with start_span(name=name, **kwargs) as span:
        return str(span.export())


@contextmanager
def span_current_scope(span: SpanLike) -> Iterator[None]:
    """Mark `span` as the current span for the duration of the block.

    The turn driver is a generator: each resumption may run on a
    different thread / contextvars copy (Starlette iterates sync
    generators through a threadpool), so the current-span state set
    when the span opened does not survive across yields. Re-entering
    this scope per resumption keeps `wrap_openai`'s auto-instrumented
    chat/embedding spans attached to the turn."""
    if not _enabled or isinstance(span, NoopSpan):
        yield
        return
    span.set_current()
    try:
        yield
    finally:
        span.unset_current()
