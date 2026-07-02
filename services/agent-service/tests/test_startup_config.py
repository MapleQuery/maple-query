"""Startup config invariants: refuse to boot without a required secret.

The lifespan calls `build_app_state`, which raises RuntimeError when
the OpenAI key or GCP project is missing. Cloud Run treats a lifespan
crash as an unhealthy revision and rolls back — same behaviour as an
imported module that raises on `create_app()`.
"""
from __future__ import annotations

import pytest
from pydantic import SecretStr
from semantic_enrich.config.settings import Settings

from agent_service.config import AgentServiceSettings
from agent_service.deps import build_app_state


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank out every relevant env var AND disable the auto-loaded
    `.env` file baked into each Settings class at import time.

    The `env_file` path is captured by `find_dotenv(usecwd=True)` at
    module import — when pytest boots from the repo root that resolves
    to the developer's real `.env`, which carries real OpenAI +
    GCP secrets and defeats "missing key" assertions."""
    for var in (
        "MQAGENT_OPENAI_API_KEY",
        "WHENRICH_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "MQAGENT_GCP_PROJECT_ID",
        "WHENRICH_GCP_PROJECT_ID",
        "GCP_PROJECT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setitem(AgentServiceSettings.model_config, "env_file", None)


def test_missing_openai_key_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    settings = AgentServiceSettings(
        api_token=SecretStr("x"),
        gcp_project_id="proj",
        # openai_api_key intentionally omitted
    )
    with pytest.raises(RuntimeError, match="openai_api_key"):
        build_app_state(settings)


def test_missing_project_id_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    settings = AgentServiceSettings(
        api_token=SecretStr("x"),
        openai_api_key=SecretStr("k"),
        # gcp_project_id intentionally omitted
    )
    with pytest.raises(RuntimeError, match="gcp_project_id"):
        build_app_state(settings)


def test_cors_origins_parsing() -> None:
    settings = AgentServiceSettings(
        cors_origins="http://a.example,  http://b.example ,",
    )
    assert settings.parsed_cors_origins() == [
        "http://a.example",
        "http://b.example",
    ]
