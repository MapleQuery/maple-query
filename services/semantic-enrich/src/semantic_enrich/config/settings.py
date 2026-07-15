"""Deploy-time configuration loaded from environment variables.

Prefix `WHENRICH_` matches the WHLOAD_ / WHINGEST_ pattern in sibling
services. Per-run intent — `--run-id`, `--limit-packages`,
`--limit-package-ids`, `--limit-orgs`, `--dry-run` — is CLI-only.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_schemas_dir() -> Path:
    """Walk up from cwd looking for `infra/terraform/schemas/`."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "infra" / "terraform" / "schemas"
        if candidate.is_dir():
            return candidate
    return Path("infra/terraform/schemas")


def _find_service_dir() -> Path:
    """Return `services/semantic-enrich/` relative to the repo root.

    Walks up from cwd looking for the service dir so any file default
    keyed off it resolves to the same absolute path whether the CLI is
    invoked from the repo root or from a subdir.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "services" / "semantic-enrich"
        if candidate.is_dir():
            return candidate
    return Path("services") / "semantic-enrich"


def _find_staging_dir() -> Path:
    """Default to `services/semantic-enrich/stage/` relative to repo root."""
    return _find_service_dir() / "stage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHENRICH_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
        populate_by_name=True,
    )

    # ── 4.3 fields ──
    generation_model: str = "Qwen/Qwen2.5-14B-Instruct"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda"

    # Optional HF cache override. Falls through to the HF default
    # (`$HF_HOME` or `~/.cache/huggingface`) when unset.
    hf_cache_dir: Path | None = None

    # ── 4.4 additions ──

    # Project-wide concept — same value across every service in this
    # repo — so we accept both the service-prefixed and the unprefixed
    # env names. WHENRICH_GCP_PROJECT_ID wins if both are set.
    gcp_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WHENRICH_GCP_PROJECT_ID", "GCP_PROJECT_ID"),
    )

    # BQ identifiers.
    bq_dataset_raw: str = "raw"
    bq_documents_table: str = "documents"
    bq_rows_table: str = "rows"
    bq_dataset_semantic: str = "semantic"
    bq_datasets_table: str = "datasets"
    bq_columns_table: str = "columns"

    # Generation tunables. `generation_max_tokens` must fit a full
    # `column_chunk_size` batch of ColumnOutput objects in one guided-JSON
    # array; at 15 columns/chunk a completion is ~2.2k tokens, so 3000
    # leaves headroom. Too small truncates the array mid-object and fails
    # the whole package (a truncated chunk is not retried).
    generation_max_tokens: int = 3000
    generation_temperature: float = 0.0
    generation_dtype: str = "bfloat16"

    # Embedding tunables for the local backend. The columns/datasets
    # subcommands embed via OpenAI and validate against the
    # `openai_embedding_*` fields below, not these.
    embedding_dim: int = 1024
    embedding_batch_size: int = 64

    # Sampling.
    sample_rows_per_package: int = 10
    sample_column_cap: int = 40

    # `datasets-extract` fan-out. Each package costs two BQ queries
    # (column-union + sample-rows); running them serially against
    # ~3,700 packages takes ~2-3 hours at ~1.5s round-trip per query.
    # 16 concurrent workers cut that to ~12 minutes and stay well
    # under BQ's per-user concurrent-jobs ceiling.
    extract_concurrency: int = 16

    # ── columns enrichment ──
    # 15 columns/chunk keeps one guided-JSON completion within
    # `generation_max_tokens` (see above); larger chunks overflow the
    # token budget and truncate the array, failing the package.
    column_chunk_size: int = 15
    # Wide-package safety belt. 140 x 15 = 2,100 columns — headroom over
    # the corpus's 1,383-column max at the smaller chunk size.
    column_chunk_max_chunks_per_package: int = 140
    # ≤10 distinct sample values per column (parent §10).
    column_sample_values_cap: int = 10
    # Column-name allowlist (§5.3). Admits identifier-ish, hyphens,
    # dots, slashes, spaces. Rejects quotes, backslashes, backticks,
    # `$`-prefixed paths, and anything ≥ 200 chars.
    column_name_allowlist_re: str = r"^[A-Za-z0-9_][A-Za-z0-9_\-./ ]{0,199}$"
    # Single-retry temperature on per-chunk invariant violation (§8.3).
    # `temperature=0` is deterministic — a retry would reproduce the
    # same failure — so a small bump samples a different completion.
    column_chunk_retry_temperature: float = 0.2

    # Staging.
    staging_dir: Path = Field(default_factory=_find_staging_dir)
    flush_every_n_packages: int = 500

    # Behaviour knobs.
    max_retries: int = 3
    dry_run: bool = False

    # Schema source of truth.
    schemas_dir: Path = Field(default_factory=_find_schemas_dir)

    # Run identity. A new UUID per process by default; operators
    # override with --run-id to resume an existing stage dir.
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── 4.7 OpenAI embedding swap ──
    # After 4.7 lands, datasets-embed / columns-embed and the reembed
    # subcommands call OpenAI text-embedding-3-small (1536-dim).
    #
    # Both the WHENRICH_-prefixed and unprefixed forms are accepted;
    # the prefixed form wins, same posture as gcp_project_id.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WHENRICH_OPENAI_API_KEY", "OPENAI_API_KEY"
        ),
    )
    openai_embedding_model: str = "text-embedding-3-small"
    # Sanity-check knob: the reembed pass and the retargeted embed
    # subcommands assert every vector matches this before writing.
    # A mismatch means the operator pointed at a different model and
    # forgot to update the dim.
    openai_embedding_dim: int = 1536
    openai_embedding_batch_size: int = 128
    openai_request_timeout_s: float = 30.0
    openai_max_retries: int = 3

    # ── 4.6 retrieval-validation harness ──
    # OpenAI generation config. The embedding config above is reused as-is.
    openai_generation_model: str = "gpt-4o"
    openai_generation_temperature: float = 0.0
    openai_generation_max_tokens: int = 1024

    # Retrieval knobs. See PRD §6.4 for the k rationale.
    eval_k_packages: int = 5
    eval_k_columns: int = 15
    # Per-package cap on document literals inlined into the SQL-gen prompt.
    # `raw.rows` is clustered by document_id; only a literal IN-list
    # prunes the clustered scan at plan time. 10 gives the model room to
    # pick "most recent" or a specific title among sibling docs without
    # bloating the prompt (5 pkgs x 10 = 50 literals in the worst case).
    eval_max_documents_per_package: int = 10

    # SQL guard knobs. Cost cap default catches hallucinated full-scan
    # queries against `raw.rows` (~50 GB unfiltered) cheaply; operators
    # raise it deliberately via --max-bytes-billed.
    eval_max_bytes_billed: int = 50 * 1024 * 1024 * 1024
    eval_query_timeout_ms: int = 30_000
    eval_dry_run_timeout_ms: int = 10_000
    eval_row_limit: int = 100
    eval_allowed_datasets: tuple[str, ...] = ("raw", "semantic")
    eval_forbidden_keywords: tuple[str, ...] = (
        "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP",
        "ALTER", "GRANT", "REVOKE", "TRUNCATE", "CALL",
    )

    # Paths. Defaults land at the committed fixture + template locations
    # relative to the repo root; walked up from cwd like staging_dir.
    eval_questions_path: Path = Field(
        default_factory=lambda: _find_service_dir() / "eval" / "questions.yaml"
    )
    eval_prompt_template: Path = Field(
        default_factory=lambda: (
            _find_service_dir() / "eval" / "prompts" / "sql_generation.j2"
        )
    )
    eval_reports_dir: Path = Field(
        default_factory=lambda: _find_service_dir() / "eval" / "reports"
    )

    # Run identity. A new UUID per process by default; separate from
    # `run_id` so an eval run inside a larger session gets its own id.
    eval_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── 5.1 agent loop ──
    # Per-turn budgets. Tool-call and SQL-execution caps are the runtime
    # tripwires. There is intentionally no per-turn dollar cap — cost
    # enforcement lives at OpenAI's daily-usage dashboard.
    agent_max_tool_calls: int = 6
    agent_max_sql_executions: int = 2
    agent_turn_timeout_seconds: int = 60
    # Parallel tool-call fan-out ceiling per assistant response.
    agent_parallel_tool_calls: int = 3

    # History compaction. `keep_turns` counts (user, assistant, tool*)
    # groups kept verbatim; anything older collapses into one rolling
    # summary system message. Hard message cap is a sanity check, not
    # a UX ceiling.
    agent_history_keep_turns: int = 4
    agent_history_max_messages: int = 200

    # In-memory response cache. TTL matches the snapshot-hash refresh
    # cadence; the replay delay makes cached hits feel progressive.
    agent_cache_max_entries: int = 1000
    agent_cache_max_value_bytes: int = 1_000_000
    agent_cache_ttl_seconds: int = 3600
    agent_cache_replay_delay_ms: int = 50
    agent_snapshot_refresh_seconds: int = 3600

    # Prompt template. Rendered once at process start; the rendered
    # bytes feed the cache key so prompt edits invalidate on redeploy.
    agent_system_prompt_path: Path = Field(
        default_factory=lambda: (
            _find_service_dir() / "agent" / "prompts" / "system.j2"
        )
    )

    # ── turn pipeline (loop v2) ──
    # Which orchestrator serves turns. Both ship in the same image;
    # flipping requires only a redeploy with this env var.
    agent_loop_impl: Literal["v1", "v2"] = "v1"
    # The v2 system prompt (leaner: tool-side enforcement replaced the
    # rule prose). Loaded only when agent_loop_impl == "v2".
    agent_prompt_v2_path: Path = Field(
        default_factory=lambda: (
            _find_service_dir() / "agent" / "prompts" / "v2" / "system.j2"
        )
    )
    # Verify-phase retry gate: how many times a verdict of "retry" may
    # re-enter research within one turn.
    agent_verify_max_retries: int = 1
    # Weak-retrieval floor surfaced in the search_datasets result: a
    # top_similarity below this suggests reformulating the query before
    # concluding the data is missing.
    agent_reformulation_threshold: float = 0.5

    # Model cost accounting (observability only; not enforced).
    # $/1K tokens; defaults match gpt-4o's published rates.
    agent_model_input_rate: float = 0.0025
    agent_model_output_rate: float = 0.010

    # sample_rows tool budgets. Kept generous — the query is
    # cluster-pruned to one package.
    agent_sample_rows_timeout_ms: int = 5_000
    agent_sample_rows_max_bytes_billed: int = 1024 * 1024 * 1024

    # ── self-enforcing tool contract ──
    # run_sql NULL-ratio advisory: columns whose NULL ratio over the
    # returned rows meets this threshold get flagged in the tool result.
    # 0.8 comes from the trace evidence (observed failure: 99/100 NULL);
    # the emitted counter gives threshold-tuning data.
    agent_null_ratio_threshold: float = 0.8
    # list_documents per-column sample values: rows read per doc, chars
    # kept per value, and columns kept per doc. All three bound the tool
    # payload the model has to read.
    agent_sample_values_rows: int = 3
    agent_sample_value_max_chars: int = 40
    agent_sample_values_max_columns: int = 30
    # Docs whose generated-header (`__col_N`) share of columns exceeds
    # this ratio get quality-flagged and demoted (never dropped —
    # legitimately wide generated-header tables exist).
    agent_generated_header_ratio: float = 0.5

    # ── Braintrust tracing ──
    # Wrapping the OpenAI client through `braintrust.wrap_openai` sends
    # every generation / embedding / tool call to the Braintrust project.
    # Disabled by default — set `braintrust_api_key` to turn it on. The
    # unprefixed `BRAINTRUST_API_KEY` is accepted too so the same value
    # works for the Braintrust CLI / SDK.
    braintrust_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WHENRICH_BRAINTRUST_API_KEY", "BRAINTRUST_API_KEY"
        ),
    )
    braintrust_project: str = "maplequery"

    # ── agent loop span tracing ──
    # Master switch for the turn/session/tool span wrappers around the
    # agent loop. With this False (or no Braintrust key), every wrapper
    # is a no-op passthrough — zero overhead beyond a boolean check —
    # and only the raw `wrap_openai` auto-instrumentation (also keyed
    # off the API key) remains.
    agent_trace_sessions: bool = True
    # In-process conversation_id → exported-session-span map. Sized for
    # demo-scale traffic; an instance restart mid-conversation starts a
    # second session root for that conversation, which is accepted —
    # the map is best-effort, not a durability layer.
    agent_trace_session_map_entries: int = 1000
    agent_trace_session_ttl_seconds: int = 86_400
