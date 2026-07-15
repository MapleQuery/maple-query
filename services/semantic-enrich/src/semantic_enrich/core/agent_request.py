"""The chat-turn request shape shared by both loop implementations.

Lives in its own module (rather than `agent_loop`) so the v2 pipeline
can import it without importing the v1 loop — the import-linter
independence contract between the two orchestrators depends on the
wire types being neutral ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatRequest:
    """One `POST /chat`-shaped input.

    `history` follows OpenAI's chat message schema: role, content,
    tool_calls, tool_call_id. `question` is appended by the loop as
    the current turn's user message — the client sends it separately
    so the loop can key the cache off just the current question.

    `turn_records` settles the wire shape for client-persisted turn
    memory ahead of the phase that consumes it: accepted by both loop
    implementations, currently ignored by both.
    """

    conversation_id: str
    history: list[dict[str, Any]]
    question: str
    turn_records: list[dict[str, Any]] = field(default_factory=list)
