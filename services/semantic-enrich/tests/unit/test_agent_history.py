"""History validation + rolling-summary compaction."""
from __future__ import annotations

import pytest

from semantic_enrich.clients.openai import StructuredGenerationResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_history


class _FakeOpenAI:
    """Records summariser calls, returns a canned summary."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def embed(self, texts):  # pragma: no cover - unused by the summariser
        return [[0.0]] * len(texts)

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: dict,
        schema_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> StructuredGenerationResult:
        self.calls.append(
            {"prompt": prompt, "schema_name": schema_name, "model": model}
        )
        return StructuredGenerationResult(
            parsed={"summary": "- user asked X\n- ran SQL Y"},
            tokens_in=10,
            tokens_out=5,
        )

    def chat_with_tools(self, **kwargs):  # pragma: no cover - unused here
        raise RuntimeError("not used in history tests")


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def test_validate_rejects_unknown_role() -> None:
    with pytest.raises(agent_history.InvalidHistoryError):
        agent_history.validate(
            [{"role": "wizard", "content": "hi"}], settings=_settings()
        )


def test_validate_rejects_orphan_tool_message() -> None:
    with pytest.raises(agent_history.InvalidHistoryError):
        agent_history.validate(
            [{"role": "tool", "tool_call_id": "abc", "content": "{}"}],
            settings=_settings(),
        )


def test_validate_accepts_matched_tool_message() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc",
                    "type": "function",
                    "function": {"name": "search_datasets", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "abc", "content": "{}"},
    ]
    agent_history.validate(history, settings=_settings())


def test_validate_rejects_over_max() -> None:
    settings = _settings().model_copy(update={"agent_history_max_messages": 3})
    with pytest.raises(agent_history.InvalidHistoryError):
        agent_history.validate(
            [{"role": "user", "content": "x"}] * 4, settings=settings
        )


def test_compact_short_history_returns_verbatim() -> None:
    settings = _settings()
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
    ]
    result = agent_history.compact(
        history=history, settings=settings, openai_client=_FakeOpenAI()
    )
    assert result.summary_message is None
    assert result.messages == history


def test_compact_emits_summary_on_overflow() -> None:
    settings = _settings().model_copy(update={"agent_history_keep_turns": 2})
    openai = _FakeOpenAI()
    history: list[dict] = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    result = agent_history.compact(
        history=history, settings=settings, openai_client=openai
    )
    assert result.summary_message is not None
    assert result.summary_message.get("mq_summary") is True
    # Summariser was invoked exactly once.
    assert len(openai.calls) == 1
    # Verbatim window kept: last 2 user + assistant pairs = 4 messages.
    assert result.messages[0] == result.summary_message
    assert len(result.messages) == 1 + 4


def test_compact_reuses_existing_summary_when_no_new_overflow() -> None:
    settings = _settings().model_copy(update={"agent_history_keep_turns": 2})
    openai = _FakeOpenAI()
    summary = {"role": "system", "content": "prior summary", "mq_summary": True}
    history = [
        summary,
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    result = agent_history.compact(
        history=history, settings=settings, openai_client=openai
    )
    # No re-summarisation because verbatim window covers everything.
    assert openai.calls == []
    assert result.summary_message is None
    assert result.messages[0] == summary
