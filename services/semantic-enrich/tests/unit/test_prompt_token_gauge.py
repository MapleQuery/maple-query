"""System-prompt token gauge + the prompt-size regression bound.

The bound (< 3.0K tokens, ~30% headroom over the current rendered
prompt) turns silent prompt growth into a CI failure instead of a
silently rising per-call bill."""
from __future__ import annotations

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_loop import load_system_prompt
from semantic_enrich.core.agent_tracing import (
    count_prompt_tokens,
    log_prompt_gauge,
)

PROMPT_TOKEN_BUDGET = 3_000


def _tokens_or_skip(text: str, model: str) -> int:
    tokens = count_prompt_tokens(text, model=model)
    if tokens is None:
        pytest.skip("tiktoken encoding unavailable (offline, cold cache)")
    return tokens


def test_count_prompt_tokens_returns_plausible_count() -> None:
    tokens = _tokens_or_skip(
        "hello world, this is a short token gauge probe.", "gpt-4o"
    )
    assert 5 <= tokens <= 20


def test_unknown_model_falls_back_to_default_encoding() -> None:
    tokens = _tokens_or_skip("hello world", "definitely-not-a-model")
    assert tokens > 0


def test_rendered_v1_prompt_stays_under_budget() -> None:
    settings = Settings()
    prompt, prompt_hash = load_system_prompt(
        settings.agent_system_prompt_path, settings
    )
    tokens = _tokens_or_skip(prompt, settings.openai_generation_model)
    assert prompt_hash
    # Lower bound is a sanity check that we measured the real prompt,
    # not an empty render.
    assert 500 < tokens < PROMPT_TOKEN_BUDGET


def test_log_prompt_gauge_returns_count() -> None:
    tokens = log_prompt_gauge(
        prompt="hello world", prompt_hash="h-1", settings=Settings()
    )
    assert tokens >= 0
