"""Typed SSE event payloads emitted by the agent loop.

The loop emits events via a callback; 5.2 wraps them in `event:
<type>\\ndata: <json>\\n\\n` SSE frames. Keeping the events as frozen
dataclasses in one place lets the CLI, the future HTTP surface, and
the harness (5.4) all consume the same shape without duplicating a
Union type.

`AgentEvent.to_sse_frame` renders the ` event: … / data: … ` form
directly so 5.2 doesn't need its own encoder. `from_sse_frame` on the
Union round-trips them for tests.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventType = Literal[
    "turn_start",
    "cache_hit",
    "retrieval_started",
    "datasets_ranked",
    "columns_ranked",
    "documents_listed",
    "sample_rows",
    "sql_generated",
    "sql_guarded",
    "sql_executed",
    "rows",
    "message_delta",
    "cost_update",
    "budget_exceeded",
    "turn_timeout",
    "tool_error",
    "done",
    "error",
]


@dataclass(frozen=True)
class _EventBase:
    """Base for typed events. Subclasses set `type` as a class var so
    the value lives in the schema rather than requiring every event
    constructor to repeat it."""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.event_type
        return payload

    def to_sse_frame(self) -> str:
        return (
            f"event: {self.event_type}\n"
            f"data: {json.dumps(self.to_dict(), default=str)}\n\n"
        )

    @property
    def event_type(self) -> EventType:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class TurnStart(_EventBase):
    conversation_id: str
    turn_id: str
    cached: bool

    @property
    def event_type(self) -> EventType:
        return "turn_start"


@dataclass(frozen=True)
class CacheHit(_EventBase):
    cache_key_prefix: str

    @property
    def event_type(self) -> EventType:
        return "cache_hit"


@dataclass(frozen=True)
class RetrievalStarted(_EventBase):
    query: str
    k: int

    @property
    def event_type(self) -> EventType:
        return "retrieval_started"


@dataclass(frozen=True)
class DatasetsRanked(_EventBase):
    candidates: list[dict[str, Any]]

    @property
    def event_type(self) -> EventType:
        return "datasets_ranked"


@dataclass(frozen=True)
class ColumnsRanked(_EventBase):
    package_ids: list[str]
    candidates: list[dict[str, Any]]

    @property
    def event_type(self) -> EventType:
        return "columns_ranked"


@dataclass(frozen=True)
class DocumentsListed(_EventBase):
    """Emitted after `list_documents` resolves candidate docs.

    Each `documents[i]["columns"]` is a `dict[str, list[str]]` mapping
    column name to a small sample of values from the doc's leading
    rows. SSE consumers can render the map directly; the keys are the
    canonical column set for that doc.
    """

    package_ids: list[str]
    documents: list[dict[str, Any]]

    @property
    def event_type(self) -> EventType:
        return "documents_listed"


@dataclass(frozen=True)
class SampleRows(_EventBase):
    package_id: str
    rows: list[dict[str, Any]]

    @property
    def event_type(self) -> EventType:
        return "sample_rows"


@dataclass(frozen=True)
class SqlGenerated(_EventBase):
    sql: str
    rationale: str

    @property
    def event_type(self) -> EventType:
        return "sql_generated"


@dataclass(frozen=True)
class SqlGuarded(_EventBase):
    accepted: bool
    reason: str | None
    sql_final: str
    dry_run_bytes: int | None

    @property
    def event_type(self) -> EventType:
        return "sql_guarded"


@dataclass(frozen=True)
class SqlExecuted(_EventBase):
    row_count: int
    bytes_billed: int
    elapsed_ms: int
    sample_rows: list[dict[str, Any]]

    @property
    def event_type(self) -> EventType:
        return "sql_executed"


@dataclass(frozen=True)
class Rows(_EventBase):
    sql_call_id: str
    rows: list[dict[str, Any]]
    is_last: bool

    @property
    def event_type(self) -> EventType:
        return "rows"


@dataclass(frozen=True)
class MessageDelta(_EventBase):
    delta: str

    @property
    def event_type(self) -> EventType:
        return "message_delta"


@dataclass(frozen=True)
class CostUpdate(_EventBase):
    tokens_in_total: int
    tokens_out_total: int
    dollars_spent: float

    @property
    def event_type(self) -> EventType:
        return "cost_update"


@dataclass(frozen=True)
class BudgetExceeded(_EventBase):
    which: Literal["tool_calls", "sql_executions"]
    value: int
    cap: int

    @property
    def event_type(self) -> EventType:
        return "budget_exceeded"


@dataclass(frozen=True)
class TurnTimeout(_EventBase):
    elapsed_ms: int
    cap_ms: int

    @property
    def event_type(self) -> EventType:
        return "turn_timeout"


@dataclass(frozen=True)
class ToolError(_EventBase):
    tool: str
    message: str

    @property
    def event_type(self) -> EventType:
        return "tool_error"


@dataclass(frozen=True)
class Done(_EventBase):
    turn_id: str
    total_tool_calls: int
    total_dollars: float
    elapsed_ms: int

    @property
    def event_type(self) -> EventType:
        return "done"


@dataclass(frozen=True)
class ErrorEvent(_EventBase):
    message: str
    retryable: bool
    reason: str | None = field(default=None)

    @property
    def event_type(self) -> EventType:
        return "error"


AgentEvent = (
    TurnStart
    | CacheHit
    | RetrievalStarted
    | DatasetsRanked
    | ColumnsRanked
    | DocumentsListed
    | SampleRows
    | SqlGenerated
    | SqlGuarded
    | SqlExecuted
    | Rows
    | MessageDelta
    | CostUpdate
    | BudgetExceeded
    | TurnTimeout
    | ToolError
    | Done
    | ErrorEvent
)


_EVENT_CLASSES: dict[str, type[_EventBase]] = {
    "turn_start": TurnStart,
    "cache_hit": CacheHit,
    "retrieval_started": RetrievalStarted,
    "datasets_ranked": DatasetsRanked,
    "columns_ranked": ColumnsRanked,
    "documents_listed": DocumentsListed,
    "sample_rows": SampleRows,
    "sql_generated": SqlGenerated,
    "sql_guarded": SqlGuarded,
    "sql_executed": SqlExecuted,
    "rows": Rows,
    "message_delta": MessageDelta,
    "cost_update": CostUpdate,
    "budget_exceeded": BudgetExceeded,
    "turn_timeout": TurnTimeout,
    "tool_error": ToolError,
    "done": Done,
    "error": ErrorEvent,
}


def from_sse_frame(frame: str) -> AgentEvent:
    """Parse a single SSE frame back into an event. Round-trips
    `to_sse_frame`. Used by tests + the cache replay path so recorded
    events stay first-class values rather than opaque strings."""
    lines = [line for line in frame.strip().splitlines() if line.strip()]
    event_line = next(
        (line for line in lines if line.startswith("event:")), None
    )
    data_line = next(
        (line for line in lines if line.startswith("data:")), None
    )
    if event_line is None or data_line is None:
        raise ValueError(f"malformed sse frame: {frame!r}")
    event_type = event_line.removeprefix("event:").strip()
    payload = json.loads(data_line.removeprefix("data:").strip())
    payload.pop("type", None)
    cls = _EVENT_CLASSES.get(event_type)
    if cls is None:
        raise ValueError(f"unknown event type: {event_type!r}")
    return cls(**payload)  # type: ignore[return-value]
