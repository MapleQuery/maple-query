"""The v2 system prompt: size bound + content contract.

The rewrite is only safe because rule enforcement moved into the
tools; these tests pin that the prompt stays lean (the moved rule
prose must not creep back) and that the contracts the tools cannot
own — identity, citation links, tool inventory — are present.
"""
from __future__ import annotations

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_loop import load_system_prompt
from semantic_enrich.core.agent_tools import TOOL_NAMES
from semantic_enrich.core.agent_tracing import count_prompt_tokens

PROMPT_V2_TOKEN_BUDGET = 1_300


def _rendered() -> str:
    settings = Settings()
    prompt, prompt_hash = load_system_prompt(
        settings.agent_prompt_v2_path, settings
    )
    assert prompt_hash
    return prompt


def test_renders_under_budget() -> None:
    settings = Settings()
    prompt = _rendered()
    tokens = count_prompt_tokens(
        prompt, model=settings.openai_generation_model
    )
    if tokens is None:
        pytest.skip("tiktoken encoding unavailable (offline, cold cache)")
    # Lower bound sanity-checks that we measured a real render.
    assert 300 < tokens <= PROMPT_V2_TOKEN_BUDGET


def test_contains_identity_line() -> None:
    prompt = _rendered()
    assert "You are MapleQuery" in prompt
    assert "not an OpenAI or GPT product" in prompt


def test_names_all_six_tools() -> None:
    prompt = _rendered()
    for tool in TOOL_NAMES:
        assert tool in prompt, f"tool {tool} missing from prompt v2"


def test_contains_citation_contract() -> None:
    prompt = _rendered()
    assert "[<title>](/datasets/<package_id>)" in prompt
    assert "verbatim" in prompt


def test_contains_retrieval_and_quality_hooks() -> None:
    prompt = _rendered()
    assert "reformulation_threshold" in prompt
    assert "null_ratio_warning" in prompt


def test_moved_rule_prose_is_gone() -> None:
    """The tool contract owns these now; their prose must not return."""
    prompt = _rendered()
    for banned in ("SET INTERSECTION", "PARSE_JSON", "JSONPath"):
        assert banned not in prompt, f"moved rule prose leaked: {banned}"


def test_row_limit_is_rendered() -> None:
    settings = Settings()
    prompt = _rendered()
    assert f"LIMIT {settings.eval_row_limit}" in prompt
