"""Flag-gated loop construction shared by the CLI and HTTP surfaces.

`agent_loop_impl` selects which orchestrator serves turns. This module
is the only place that knows both — the two loop modules themselves
never import each other (import-linter enforces it). Callers get a
`LoopHandle` whose `run_turn` hides the choice behind one traced
callable.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_loop
from semantic_enrich.core.agent import memory, phases, pipeline, triage, verify
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from semantic_enrich.core.agent_tracing import (
    RunTurnFn,
    TracedDeps,
    log_prompt_gauge,
    run_turn_traced,
)

LoopImpl = Literal["v1", "v2"]


def resolve_run_turn(loop_impl: LoopImpl) -> RunTurnFn:
    """The untraced turn driver for `loop_impl`."""
    if loop_impl == "v2":
        return pipeline.run_turn
    return agent_loop.run_turn


@dataclass
class LoopHandle:
    """One built loop: deps plus the traced entrypoint bound to the
    selected implementation."""

    loop_impl: LoopImpl
    deps: TracedDeps
    prompt_hash: str
    system_prompt_tokens: int

    def run_turn(
        self,
        request: ChatRequest,
        *,
        session_parent: str | None = None,
    ) -> Iterator[agent_events.AgentEvent]:
        return run_turn_traced(
            request=request,
            deps=self.deps,
            session_parent=session_parent,
            run_turn_fn=resolve_run_turn(self.loop_impl),
            loop_impl=self.loop_impl,
        )


def build_loop_handle(
    *,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    snapshot_hash_provider: Callable[[], str] | None = None,
    loop_impl: LoopImpl | None = None,
) -> LoopHandle:
    """Load the implementation's prompt, build the shared cache and
    snapshot provider, and assemble the right deps flavour.

    `loop_impl` overrides `settings.agent_loop_impl` (CLI flag);
    `snapshot_hash_provider` overrides the warehouse lookup (dry-run).
    Raises RuntimeError when the prompt template is missing, matching
    the v1 startup behaviour."""
    impl: LoopImpl = loop_impl or settings.agent_loop_impl
    prompt_path = (
        settings.agent_prompt_v2_path
        if impl == "v2"
        else settings.agent_system_prompt_path
    )
    prompt, prompt_hash = agent_loop.load_system_prompt(
        prompt_path, settings
    )
    cache = ResponseCache(
        max_entries=settings.agent_cache_max_entries,
        max_value_bytes=settings.agent_cache_max_value_bytes,
        ttl_seconds=settings.agent_cache_ttl_seconds,
    )
    provider = (
        snapshot_hash_provider
        if snapshot_hash_provider is not None
        else agent_loop.make_snapshot_hash_provider(bq, settings)
    )
    tokens = log_prompt_gauge(
        prompt=prompt, prompt_hash=prompt_hash, settings=settings
    )
    deps: TracedDeps
    if impl == "v2":
        # The triage kill switch ("off") falls back to the passthrough
        # stub; "log" and "act" both classify, so both need the prompt
        # template present at startup.
        triage_phase: phases.TriagePhase = (
            phases.PassthroughTriage()
            if settings.agent_triage_mode == "off"
            else triage.QueryTriage.from_settings(settings)
        )
        # Same kill-switch shape for the verify phase: "log" and "act"
        # both check (and need the template at startup), "off" wires
        # the always-fits stub.
        verify_phase: phases.VerifyPhase = (
            phases.AlwaysFitsVerifier()
            if settings.agent_verify_mode == "off"
            else verify.AnswerFitVerifier.from_settings(settings)
        )
        # v2 memory: digest replay cache + cached snapshot hash. The
        # snapshot queries run at most once per refresh window, and a
        # changed hash purges stale replay entries at that boundary.
        replay_cache = memory.ReplayCacheV2(
            max_entries=settings.agent_cache_max_entries,
            ttl_seconds=settings.agent_cache_ttl_seconds,
        )
        raw_provider = (
            snapshot_hash_provider
            if snapshot_hash_provider is not None
            else memory.make_snapshot_hash_provider_v2(bq, settings)
        )
        cached_provider = memory.CachedSnapshotHash(
            provider=raw_provider,
            refresh_seconds=settings.agent_snapshot_refresh_seconds,
            on_change=replay_cache.invalidate_on_snapshot,
        )
        deps = phases.PipelineDeps(
            bq=bq,
            openai_client=openai_client,
            settings=settings,
            system_prompt=prompt,
            prompt_hash=prompt_hash,
            cache=cache,
            snapshot_hash_provider=cached_provider,
            system_prompt_tokens=tokens,
            triage=triage_phase,
            memory=memory.SessionMemory(cache=replay_cache),
            verifier=verify_phase,
        )
    else:
        deps = agent_loop.LoopDeps(
            bq=bq,
            openai_client=openai_client,
            settings=settings,
            system_prompt=prompt,
            prompt_hash=prompt_hash,
            cache=cache,
            snapshot_hash_provider=provider,
            system_prompt_tokens=tokens,
        )
    return LoopHandle(
        loop_impl=impl,
        deps=deps,
        prompt_hash=prompt_hash,
        system_prompt_tokens=tokens,
    )
