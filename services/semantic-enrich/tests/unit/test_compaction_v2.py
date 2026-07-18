"""Compaction v2: keep-window verbatim, everything older dropped, no
model call ever."""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent.memory import compact_v2


def _turn(i: int) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]


def test_keeps_last_n_turns_verbatim() -> None:
    history: list[dict[str, Any]] = []
    for i in range(5):
        history.extend(_turn(i))
    out = compact_v2(history, keep_turns=2)
    assert out == _turn(3) + _turn(4)


def test_short_history_kept_whole() -> None:
    history = _turn(0)
    assert compact_v2(history, keep_turns=2) == history


def test_tool_messages_and_nonstring_content_excluded() -> None:
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "{}"},
        {"role": "assistant", "content": "a"},
    ]
    assert compact_v2(history, keep_turns=2) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_zero_keep_turns_or_empty_history() -> None:
    assert compact_v2([], keep_turns=2) == []
    assert compact_v2(_turn(0), keep_turns=0) == []


def test_partial_last_turn_counts_from_user_message() -> None:
    history = _turn(0) + _turn(1) + [{"role": "user", "content": "q2"}]
    out = compact_v2(history, keep_turns=2)
    assert out == [*_turn(1), {"role": "user", "content": "q2"}]
