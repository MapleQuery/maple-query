"""FastAPI app factory.

Wires config, CORS, auth, routes, and the process-lifetime dependency
graph. Production entrypoint is `agent_service.app:app`; tests build a
scoped app via `create_app(app_state=...)` and swap in fake clients.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_service.config import AgentServiceSettings
from agent_service.deps import AppState, build_app_state
from agent_service.routes import chat, datasets, health, sql

_log = structlog.get_logger("agent_service.app")


def create_app(
    *,
    service_settings: AgentServiceSettings | None = None,
    app_state: AppState | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `service_settings` is read from env if not provided. `app_state` is
    a hook for tests — passing one skips the real client construction so
    fakes can be swapped in without monkeypatching.
    """
    settings = service_settings or AgentServiceSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = app_state or build_app_state(settings)
        app.state.deps = state
        app.state.api_token = _resolve_api_token(settings)
        _log.info(
            "app_started",
            cors_origins=settings.parsed_cors_origins(),
            has_api_token=app.state.api_token is not None,
        )
        try:
            yield
        finally:
            _log.info("app_stopped")

    app = FastAPI(
        title="MapleQuery agent service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(sql.router)
    app.include_router(datasets.router)

    return app


def _resolve_api_token(settings: AgentServiceSettings) -> str | None:
    if settings.api_token is None:
        return None
    return settings.api_token.get_secret_value()


# Production ASGI entrypoint. `agent-service` CLI and the Cloud Run
# container both point uvicorn at `agent_service.app:app`. Tests never
# import this module-level app — they call `create_app(...)` directly
# so lifespan-owned fakes stay contained per-test.
app = create_app()
