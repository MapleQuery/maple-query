"""Client-supplied conversation history: validation + rolling summary.

Every `/chat` turn carries the full transcript in the request. The
loop:

1. Validates the shape (roles, tool_call_id references, message cap).
2. Compacts anything older than the last N turns into a rolling
   `system`-role summary message. The summary is emitted by a cheap
   `gpt-4o` call on first overflow and re-used on subsequent turns.

The compacted history is what the loop sends to OpenAI. The client
persists the un-compacted history in localStorage; the summary
message is tagged with a `mq_summary: true` metadata field so the
client can hoist it across turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
_SUMMARY_MARKER = "mq_summary"


class InvalidHistoryError(ValueError):
    """Raised for structurally-broken histories.

    HTTP layer surfaces this as `400 {"error": "invalid_history", ...}`.
    """


@dataclass(frozen=True)
class CompactionResult:
    """Return value of `compact`.

    `messages` is the OpenAI-ready list. `summary_message` (if any) is
    the freshly-emitted summary — the loop returns it to the FE so it
    can persist it and skip re-emitting next turn.
    """

    messages: list[dict[str, Any]]
    summary_message: dict[str, Any] | None


def validate(history: list[dict[str, Any]], *, settings: Settings) -> None:
    """Validate the client-supplied history.

    Rejects unknown roles, orphan tool messages, and histories longer
    than the sanity cap. The compaction step handles length in
    practice — this cap catches pathological clients or state leaks
    before they hit the summariser.
    """
    if not isinstance(history, list):
        raise InvalidHistoryError("history must be a list")
    if len(history) > settings.agent_history_max_messages:
        raise InvalidHistoryError(
            f"history_too_long: {len(history)} > "
            f"{settings.agent_history_max_messages}"
        )
    seen_tool_call_ids: set[str] = set()
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            raise InvalidHistoryError(f"message[{i}] is not an object")
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise InvalidHistoryError(f"message[{i}] has unknown role: {role!r}")
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id") if isinstance(call, dict) else None
                if isinstance(cid, str):
                    seen_tool_call_ids.add(cid)
        if role == "tool":
            tcid = msg.get("tool_call_id")
            if not isinstance(tcid, str):
                raise InvalidHistoryError(
                    f"message[{i}] tool role missing tool_call_id"
                )
            if tcid not in seen_tool_call_ids:
                raise InvalidHistoryError(
                    f"message[{i}] tool_call_id {tcid!r} has no "
                    "matching assistant tool_call earlier in the history"
                )


def compact(
    *,
    history: list[dict[str, Any]],
    settings: Settings,
    openai_client: OpenAIClient,
) -> CompactionResult:
    """Compact `history` for the next model call.

    Keeps the last `agent_history_keep_turns` user/assistant/tool
    groups verbatim; summarises everything older into one
    `system`-role message tagged with `mq_summary: true`. If the
    history already contains a summary marker, that summary is
    preserved and extended lazily (i.e. re-emitted verbatim; upstream
    turns rewrite it when the verbatim window shifts again)."""
    if not history:
        return CompactionResult(messages=[], summary_message=None)

    existing_summary = _existing_summary(history)
    non_summary_history = [
        m for m in history if not _is_summary(m)
    ]
    turn_starts = _turn_start_indices(non_summary_history)
    keep = settings.agent_history_keep_turns

    if len(turn_starts) <= keep:
        # Everything already fits under the verbatim window; if a
        # summary was previously emitted, keep it at the front.
        prefix = [existing_summary] if existing_summary else []
        return CompactionResult(
            messages=[*prefix, *non_summary_history],
            summary_message=None,
        )

    cutoff_idx = turn_starts[-keep]
    older = non_summary_history[:cutoff_idx]
    recent = non_summary_history[cutoff_idx:]

    summary_text = _summarise(
        older=older,
        prior_summary=existing_summary,
        openai_client=openai_client,
        settings=settings,
    )
    summary_message: dict[str, Any] = {
        "role": "system",
        "content": summary_text,
        _SUMMARY_MARKER: True,
    }
    return CompactionResult(
        messages=[summary_message, *recent],
        summary_message=summary_message,
    )


def _existing_summary(
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for msg in history:
        if _is_summary(msg):
            return msg
    return None


def _is_summary(msg: dict[str, Any]) -> bool:
    return bool(msg.get(_SUMMARY_MARKER)) and msg.get("role") == "system"


def _turn_start_indices(history: list[dict[str, Any]]) -> list[int]:
    """Return the index of each user-role message (a turn start).

    Assistant + tool messages between two user messages belong to the
    same turn. A history that starts with an assistant / system
    message (unusual) still has its first user message counted as the
    first turn boundary."""
    return [i for i, m in enumerate(history) if m.get("role") == "user"]


def _summarise(
    *,
    older: list[dict[str, Any]],
    prior_summary: dict[str, Any] | None,
    openai_client: OpenAIClient,
    settings: Settings,
) -> str:
    """Emit a rolling summary via a cheap gpt-4o call.

    Structured Outputs keeps the response shape flat and cheap to
    parse. The summariser sees the prior summary (if any) so it can
    fold it in rather than re-summarising from raw."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    prior = ""
    if prior_summary is not None:
        prior = f"Prior summary:\n{prior_summary.get('content', '')}\n\n"
    lines: list[str] = []
    for m in older:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        # Trim each message so the summariser call itself stays cheap.
        if len(content) > 800:
            content = content[:800] + "..."
        lines.append(f"[{role}] {content}")
    body = "\n".join(lines) if lines else "(no earlier messages)"
    prompt = (
        "You are compacting an earlier segment of a research chat "
        "about Canadian government data so the ongoing agent can "
        "remain coherent without carrying the raw turns.\n\n"
        f"{prior}"
        "Earlier messages (oldest first):\n"
        f"{body}\n\n"
        "Emit a 3-6 bullet summary. Cover: what the user asked, which "
        "packages/columns were surfaced, what SQL shapes were run, "
        "and any conclusions or open threads."
    )
    result = openai_client.generate_structured(
        prompt=prompt,
        schema=schema,
        schema_name="mq_history_summary",
        model=settings.openai_generation_model,
        temperature=settings.openai_generation_temperature,
        max_tokens=settings.openai_generation_max_tokens,
    )
    summary_body = result.parsed.get("summary", "")
    if not isinstance(summary_body, str):
        summary_body = str(summary_body)
    return f"Prior conversation summary:\n{summary_body.strip()}"
