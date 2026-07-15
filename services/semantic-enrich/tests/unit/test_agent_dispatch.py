"""Flag-gated loop construction: the dispatch layer picks the deps
flavour, the prompt template, and the untraced driver per impl."""
from __future__ import annotations

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_loop
from semantic_enrich.core.agent import phases, pipeline
from semantic_enrich.core.agent_dispatch import (
    build_loop_handle,
    resolve_run_turn,
)
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: object) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def test_resolve_run_turn_selects_the_orchestrator() -> None:
    assert resolve_run_turn("v1") is agent_loop.run_turn
    assert resolve_run_turn("v2") is pipeline.run_turn


def test_default_flag_builds_v1_deps_with_v1_prompt() -> None:
    settings = _settings()
    handle = build_loop_handle(
        settings=settings,
        bq=object(),  # type: ignore[arg-type]
        openai_client=FakeOpenAIClient(),
        snapshot_hash_provider=lambda: "snap",
    )
    assert handle.loop_impl == "v1"
    assert isinstance(handle.deps, agent_loop.LoopDeps)
    _, v1_hash = agent_loop.load_system_prompt(
        settings.agent_system_prompt_path, settings
    )
    assert handle.prompt_hash == v1_hash


def test_v2_flag_builds_pipeline_deps_with_v2_prompt() -> None:
    settings = _settings(agent_loop_impl="v2")
    handle = build_loop_handle(
        settings=settings,
        bq=object(),  # type: ignore[arg-type]
        openai_client=FakeOpenAIClient(),
        snapshot_hash_provider=lambda: "snap",
    )
    assert handle.loop_impl == "v2"
    assert isinstance(handle.deps, phases.PipelineDeps)
    _, v2_hash = agent_loop.load_system_prompt(
        settings.agent_prompt_v2_path, settings
    )
    assert handle.prompt_hash == v2_hash
    # Stub phases installed by default.
    assert isinstance(handle.deps.triage, phases.PassthroughTriage)
    assert isinstance(handle.deps.memory, phases.NoopMemory)
    assert isinstance(handle.deps.verifier, phases.AlwaysFitsVerifier)


def test_explicit_override_beats_the_settings_flag() -> None:
    settings = _settings(agent_loop_impl="v1")
    handle = build_loop_handle(
        settings=settings,
        bq=object(),  # type: ignore[arg-type]
        openai_client=FakeOpenAIClient(),
        snapshot_hash_provider=lambda: "snap",
        loop_impl="v2",
    )
    assert handle.loop_impl == "v2"
    assert isinstance(handle.deps, phases.PipelineDeps)


def test_handle_run_turn_serves_a_turn_end_to_end() -> None:
    settings = _settings(agent_loop_impl="v2")
    handle = build_loop_handle(
        settings=settings,
        bq=object(),  # type: ignore[arg-type]
        openai_client=FakeOpenAIClient(
            chat_script=[{"content": "served by v2."}]
        ),
        snapshot_hash_provider=lambda: "snap",
    )
    from semantic_enrich.core.agent_request import ChatRequest

    events = list(
        handle.run_turn(
            ChatRequest(conversation_id="c1", history=[], question="q?")
        )
    )
    types = [e.event_type for e in events]
    assert "phase_start" in types
    assert types[-1] == "done"
