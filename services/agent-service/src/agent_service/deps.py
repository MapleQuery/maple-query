"""Dependency wiring for the FastAPI app.

Builds the shared, process-lifetime dependencies (BQ client, OpenAI
client, response cache, loop deps) once at startup and stores them on
`app.state`. Route handlers pull them from there via typed accessors so
tests can swap them wholesale by mutating `app.state` before the client
runs a request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import structlog
from fastapi import Request
from semantic_enrich.clients.bq import BqClient, RealBqClient
from semantic_enrich.clients.openai import OpenAIClient, RealOpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import (
    LoopDeps,
    load_system_prompt,
    make_snapshot_hash_provider,
)
from semantic_enrich.core.agent_tracing import (
    SessionSpanMap,
    log_prompt_gauge,
    session_span_map_from_settings,
)
from semantic_enrich.providers.braintrust_tracing import configure_braintrust

from agent_service.config import AgentServiceSettings
from agent_service.telemetry import build_posthog_client

_log = structlog.get_logger("agent_service.deps")


class ProbeFn(Protocol):
    def __call__(self) -> None: ...


@dataclass
class AppState:
    """Everything the routes need. Attached to `app.state.deps`.

    The FastAPI test suite constructs one of these directly with fake
    clients; production uses `build_app_state` from the service settings.
    """

    service_settings: AgentServiceSettings
    loop_settings: Settings
    loop_deps: LoopDeps
    bq: BqClient
    openai_client: OpenAIClient
    # Optional PostHog client. `None` when no key is configured — every
    # capture call site null-checks before firing.
    posthog: object | None = None
    # conversation_id → exported Braintrust session-span parent. Inert
    # (get_or_create returns None) when tracing is unconfigured, so
    # tests constructing AppState directly can rely on the default.
    session_spans: SessionSpanMap = field(
        default_factory=lambda: SessionSpanMap(
            max_entries=1000, ttl_seconds=86_400
        )
    )


def build_loop_settings(service_settings: AgentServiceSettings) -> Settings:
    """Construct the semantic-enrich Settings, letting service-layer
    overrides win when both are present.

    The Settings object reads its own env vars (WHENRICH_*) too, so the
    forward from service settings only fills in fields the operator set
    at the service level."""
    updates: dict[str, object] = {}
    if service_settings.openai_api_key is not None:
        updates["openai_api_key"] = service_settings.openai_api_key
    if service_settings.gcp_project_id is not None:
        updates["gcp_project_id"] = service_settings.gcp_project_id
    settings = Settings()
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def build_app_state(service_settings: AgentServiceSettings) -> AppState:
    """Instantiate real BQ + OpenAI clients, load the system prompt,
    build the shared cache, and assemble `LoopDeps`.

    Called once at process start via the FastAPI lifespan. Missing
    required config raises RuntimeError so Cloud Run rolls the revision
    back rather than serving a broken instance."""
    loop_settings = build_loop_settings(service_settings)
    if not loop_settings.gcp_project_id:
        raise RuntimeError(
            "gcp_project_id missing: set MQAGENT_GCP_PROJECT_ID or "
            "WHENRICH_GCP_PROJECT_ID."
        )
    if loop_settings.openai_api_key is None:
        raise RuntimeError(
            "openai_api_key missing: set MQAGENT_OPENAI_API_KEY or "
            "WHENRICH_OPENAI_API_KEY."
        )

    # Init Braintrust before constructing the OpenAI client — the
    # client's __init__ pipes itself through `wrap_openai_client`, which
    # is a no-op until `configure_braintrust` has flipped tracing on.
    braintrust_key = (
        service_settings.braintrust_api_key.get_secret_value()
        if service_settings.braintrust_api_key is not None
        else None
    )
    braintrust_active = configure_braintrust(
        api_key=braintrust_key,
        project=service_settings.braintrust_project,
        enabled=True,
    )
    _log.info(
        "braintrust_configured",
        active=braintrust_active,
        project=service_settings.braintrust_project,
    )

    bq = RealBqClient.for_project(loop_settings.gcp_project_id)
    openai_client = RealOpenAIClient.for_settings(
        api_key=loop_settings.openai_api_key.get_secret_value(),
        embedding_model=loop_settings.openai_embedding_model,
        request_timeout_s=loop_settings.openai_request_timeout_s,
        max_retries=loop_settings.openai_max_retries,
    )
    prompt, prompt_hash = load_system_prompt(
        loop_settings.agent_system_prompt_path, loop_settings
    )
    cache = ResponseCache(
        max_entries=loop_settings.agent_cache_max_entries,
        max_value_bytes=loop_settings.agent_cache_max_value_bytes,
        ttl_seconds=loop_settings.agent_cache_ttl_seconds,
    )
    loop_deps = LoopDeps(
        bq=bq,
        openai_client=openai_client,
        settings=loop_settings,
        system_prompt=prompt,
        prompt_hash=prompt_hash,
        cache=cache,
        snapshot_hash_provider=make_snapshot_hash_provider(bq, loop_settings),
        system_prompt_tokens=log_prompt_gauge(
            prompt=prompt, prompt_hash=prompt_hash, settings=loop_settings
        ),
    )
    posthog_client = build_posthog_client(service_settings)
    _log.info("posthog_configured", active=posthog_client is not None)
    return AppState(
        service_settings=service_settings,
        loop_settings=loop_settings,
        loop_deps=loop_deps,
        bq=bq,
        openai_client=openai_client,
        posthog=posthog_client,
        session_spans=session_span_map_from_settings(loop_settings),
    )


def get_app_state(request: Request) -> AppState:
    state: AppState = request.app.state.deps
    return state
