"""Service-layer configuration.

Split from `semantic_enrich.config.settings.Settings` so the two layers
stay decoupled:

- The 5.1 loop's knobs (budgets, cache, prompts, retrieval k, guard
  caps) live in `Settings` under the `WHENRICH_` prefix.
- The service's own knobs (auth token, CORS allow-list) live here under
  the `MQAGENT_` prefix.

The service also accepts `MQAGENT_OPENAI_API_KEY` and
`MQAGENT_GCP_PROJECT_ID` and forwards them into the loop's Settings —
that way the Cloud Run manifest wires everything with one consistent
prefix even though the sibling package predates it.
"""
from __future__ import annotations

from dotenv import find_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MQAGENT_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    # Shared bearer token. Checked on every non-health route. Public by
    # construction — the FE bakes the same value into its bundle.
    api_token: SecretStr | None = None

    # CORS allow-list. Comma-separated origins; `*` disables the check
    # entirely and should only be used for local smoke tests.
    cors_origins: str = "http://localhost:3000"

    # Regex allow-list for origins whose URL varies per deploy — Vercel
    # preview URLs specifically. `None` disables regex matching; the
    # exact-match `cors_origins` list is what's used at that point.
    # Scope this to your Vercel team slug so a fork on someone else's
    # team can't hit the API.
    cors_origin_regex: str | None = None

    # OpenAI + GCP identifiers. Both accept the semantic-enrich-native
    # env names too so a single .env file (WHENRICH_OPENAI_API_KEY /
    # WHENRICH_GCP_PROJECT_ID / OPENAI_API_KEY / GCP_PROJECT_ID) still
    # works without duplication.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MQAGENT_OPENAI_API_KEY",
            "WHENRICH_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
    )
    gcp_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MQAGENT_GCP_PROJECT_ID",
            "WHENRICH_GCP_PROJECT_ID",
            "GCP_PROJECT_ID",
        ),
    )

    # HTTP surface tunables. Cloud Run picks the port through `PORT`;
    # honour that convention so `uvicorn` binds correctly when the
    # container starts.
    port: int = Field(
        default=8080,
        validation_alias=AliasChoices("MQAGENT_PORT", "PORT"),
    )

    # Datasets route defaults. Non-vector-search pagination.
    datasets_default_limit: int = 20
    datasets_max_limit: int = 100

    # ── Observability ──
    # Braintrust traces every LLM call when a key is set. Absent →
    # tracing disabled, no-op wrap around the OpenAI client. Accepts
    # the unprefixed BRAINTRUST_API_KEY so a single value works for
    # both the SDK's implicit lookup and this service.
    braintrust_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MQAGENT_BRAINTRUST_API_KEY",
            "WHENRICH_BRAINTRUST_API_KEY",
            "BRAINTRUST_API_KEY",
        ),
    )
    braintrust_project: str = Field(
        default="maplequery",
        validation_alias=AliasChoices(
            "MQAGENT_BRAINTRUST_PROJECT", "BRAINTRUST_PROJECT"
        ),
    )

    # PostHog captures product analytics from the server side (chat
    # turn finished, sql run finished). Absent → capture calls no-op.
    posthog_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MQAGENT_POSTHOG_API_KEY", "POSTHOG_API_KEY"
        ),
    )
    posthog_host: str = Field(
        default="https://us.i.posthog.com",
        validation_alias=AliasChoices(
            "MQAGENT_POSTHOG_HOST", "POSTHOG_HOST"
        ),
    )

    def parsed_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
