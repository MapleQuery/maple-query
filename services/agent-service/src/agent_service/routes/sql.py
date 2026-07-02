"""`POST /sql/run` — public wrap around the loop's `run_sql` tool.

Same guardrails as the agent's own `run_sql`. No bypass. Powers the
"edit this step" affordance in the explorer surface.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from semantic_enrich.core import agent_tools
from semantic_enrich.core.agent_events import AgentEvent

from agent_service.auth import BearerAuth
from agent_service.deps import AppState, get_app_state
from agent_service.telemetry import capture

router = APIRouter()


class SqlBody(BaseModel):
    sql: str = Field(..., min_length=1)
    rationale: str | None = Field(default=None)


class SqlResponse(BaseModel):
    """Stable response shape.

    `status` widens PRD 5.2 §2.2 to match the 5.1 loop's actual return
    surface: `ok | guard_rejected | column_not_in_doc | execution_error`.
    The extra statuses are cheap for a caller to ignore but preserving
    them keeps the guardrails identical between the model-invoked path
    and this public endpoint.
    """

    status: str
    reason: str | None = None
    sql_final: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    bytes_billed: int = 0
    elapsed_ms: int = 0
    truncated: bool = False


@router.post("/sql/run", dependencies=[BearerAuth], response_model=SqlResponse)
def run_sql(
    body: SqlBody,
    state: AppState = Depends(get_app_state),
) -> SqlResponse:
    started = time.monotonic()
    call_id = uuid.uuid4().hex

    # Wire a scratch LoopState so the `run_sql` tool implementation has
    # a state object to write against. Nothing else on the endpoint
    # touches state (no other tools ran this "turn"), so an empty
    # `known_package_ids` and no `doc_columns` is correct — the guard is
    # what enforces safety, not the doc pairing check.
    tool_state = agent_tools.LoopState(
        conversation_id="sql-run",
        turn_id=call_id,
        question=body.rationale or "",
    )
    events: list[AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=state.bq,
        openai_client=state.openai_client,
        settings=state.loop_settings,
        state=tool_state,
        emit=events.append,
    )

    result = agent_tools.run_run_sql(
        ctx=ctx,
        args={"sql": body.sql, "rationale": body.rationale or "adhoc run"},
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    status = str(result.get("status", "unknown"))
    reason = result.get("reason")
    sql_final = _resolve_sql_final(events, body.sql, result)
    rows = list(result.get("rows") or [])
    row_count = int(result.get("row_count") or 0)
    bytes_billed = int(result.get("bytes_billed") or 0)
    truncated = bool(result.get("truncated") or False)
    result_elapsed = int(result.get("elapsed_ms") or elapsed_ms)
    capture(
        state.posthog,
        distinct_id=call_id,
        event="sql_run_finished",
        properties={
            "status": status,
            "reason": reason,
            "row_count": row_count,
            "bytes_billed": bytes_billed,
            "elapsed_ms": result_elapsed,
            "truncated": truncated,
        },
    )
    return SqlResponse(
        status=status,
        reason=str(reason) if reason else None,
        sql_final=sql_final,
        rows=rows,
        row_count=row_count,
        bytes_billed=bytes_billed,
        elapsed_ms=result_elapsed,
        truncated=truncated,
    )


def _resolve_sql_final(
    events: list[AgentEvent], original_sql: str, result: dict[str, Any]
) -> str:
    """Pick the guard-normalised SQL when available, else fall back to
    the input.

    The `sql_guarded` event carries `sql_final` after any LIMIT wrap.
    When the guard is skipped (budget_exceeded / column_not_in_doc), we
    fall back to whatever the tool result echoes back, and lastly to the
    caller's original SQL."""
    from semantic_enrich.core.agent_events import SqlGuarded

    for event in reversed(events):
        if isinstance(event, SqlGuarded):
            return str(event.sql_final)
    tool_final = result.get("sql_final")
    if isinstance(tool_final, str) and tool_final:
        return tool_final
    return original_sql
