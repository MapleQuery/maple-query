# services/agent-service

FastAPI wrap around the `semantic-enrich` 5.1 agent loop. Deployed to Cloud Run in `us-central1`; exposes SSE `/chat`, a public `/sql/run`, dataset browsing endpoints, and health probes to the web app.

Almost no business logic lives here — the routes marshal HTTP↔loop, enforce the bearer-token check, and stream the loop's typed events as SSE frames. The loop, tools, cache, event schema, guard, and executor all live in the sibling `services/semantic-enrich` package and are imported directly (path dependency via `tool.uv.sources`).

## Layout

```
services/agent-service/
├── Dockerfile
├── pyproject.toml
├── src/agent_service/
│   ├── __main__.py       # local `agent-service` CLI shim
│   ├── config.py         # AgentServiceSettings (env prefix MQAGENT_)
│   ├── app.py            # create_app + module-level `app`
│   ├── deps.py           # AppState + build_app_state (lifespan wiring)
│   ├── auth.py           # bearer-token dependency
│   ├── sse.py            # sync-iterator → async byte-stream adapter
│   └── routes/
│       ├── chat.py       # POST /chat (SSE)
│       ├── sql.py        # POST /sql/run
│       ├── datasets.py   # GET /datasets, /datasets/{id}/columns
│       └── health.py     # GET /healthz, /readyz
└── tests/                # auth / cors / chat / sql / datasets / health / startup
```

## Endpoints

| Method | Path                              | Auth   | Purpose                                                                                              |
| ------ | --------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| POST   | `/chat`                           | Bearer | SSE stream of the 5.1 loop's typed events (`turn_start`…`done`). One request = one turn.             |
| POST   | `/sql/run`                        | Bearer | Public wrap around the loop's `run_sql` tool. Identical guardrails. Powers the "edit this step" UI. |
| GET    | `/datasets`                       | Bearer | With `q`: VECTOR_SEARCH over `semantic.datasets`. Without: straight scan by `generated_at DESC`.     |
| GET    | `/datasets/{package_id}/columns`  | Bearer | Per-package column list from `semantic.columns`.                                                     |
| GET    | `/healthz`                        | —      | Cloud Run liveness. Always 200.                                                                      |
| GET    | `/readyz`                         | —      | OpenAI + BQ + snapshot canary. 503 if any fails. Cloud Run startup probe.                            |

`/sql/run`'s `status` widens PRD 5.2 §2.2 to match the loop's actual return surface: `ok | guard_rejected | column_not_in_doc | budget_exceeded | execution_error`. This keeps the model-invoked path and the public endpoint behavioural identical.

## Auth

Shared bearer token, single value, `hmac.compare_digest` in `auth.py`. `MQAGENT_API_TOKEN` on the server (from Secret Manager). `NEXT_PUBLIC_MAPLEQUERY_API_TOKEN` in the FE bundle — public by construction. Rotation is a redeploy on both sides.

## CORS

Allow-list via `MQAGENT_CORS_ORIGINS` (comma-separated, exact match) plus `MQAGENT_CORS_ORIGIN_REGEX` for URLs whose host varies per deploy (Vercel previews). Default local: `http://localhost:3000`. Production adds `https://maple-query.vercel.app` in the exact list and a team-scoped regex for `maple-query-*-coles-projects-4b94bd7b.vercel.app` previews. `Access-Control-Allow-Credentials: false` — bearer token in header, no cookies.

## Configuration

Two settings layers:

- **`AgentServiceSettings`** (env prefix `MQAGENT_`) — auth token, CORS, port. Reads `MQAGENT_OPENAI_API_KEY` / `MQAGENT_GCP_PROJECT_ID` and forwards them into the loop's Settings so the Cloud Run manifest wires everything with one consistent prefix.
- **`semantic_enrich.config.settings.Settings`** (env prefix `WHENRICH_`) — the loop's own knobs: budgets, cache, retrieval `k`, guard caps, model rates. Reused verbatim.

### Observability

- **Braintrust** — when `BRAINTRUST_API_KEY` (or `MQAGENT_BRAINTRUST_API_KEY`) is set at startup, the OpenAI client is wrapped with `braintrust.wrap_openai` and every chat / embedding / tool call flows to the `MQAGENT_BRAINTRUST_PROJECT` (default `maplequery`). Missing key → no-op wrap, no traces.
- **PostHog (server-side)** — when `POSTHOG_API_KEY` (or `MQAGENT_POSTHOG_API_KEY`) is set, the service fires `chat_turn_finished` (per drained SSE stream) and `sql_run_finished` (per `POST /sql/run`) events. Host defaults to `https://us.i.posthog.com`; override via `MQAGENT_POSTHOG_HOST`. Missing key → capture calls no-op.

## Deployment

- **Region**: `us-central1` — same as the BQ US multi-region so reads are cheapest and fastest.
- **Concurrency**: 80 in-flight requests per instance (matches Cloud Run default; the loop is I/O-bound).
- **Scaling**: `min_instances=1` during the demo (avoid ~5-10s cold starts on the first question), `max_instances=10`.
- **Timeouts**: Cloud Run request 300s; loop wall-clock 60s — loop's own cap wins by design.
- **Image**: `gcr.io/<project>/agent-service:<sha>`. Multi-stage Dockerfile at repo root context; `uv sync --frozen --no-dev` in the builder, runtime image runs as non-root.

Terraform is `infra/terraform/agent_service.tf`. IAM grants: `roles/bigquery.dataViewer` on `raw` + `semantic`, `roles/bigquery.jobUser` on the project, `roles/secretmanager.secretAccessor` on `openai-api-key` and `mqagent-api-token`, `roles/logging.logWriter` for stdout → Cloud Logging.

## Local dev

```
cd services/agent-service
uv sync --extra dev
uv run agent-service                    # listens on :8080

# In another shell:
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/datasets | jq .
```

`MQAGENT_API_TOKEN`, `MQAGENT_OPENAI_API_KEY`, `MQAGENT_GCP_PROJECT_ID` must be set (or their `WHENRICH_*` / `OPENAI_API_KEY` / `GCP_PROJECT_ID` aliases).

## Tests

`uv run pytest`. All tests use fake BQ + OpenAI clients — no cloud credentials required.

- `test_auth.py` — missing / malformed / wrong / correct token, healthz bypasses auth.
- `test_cors.py` — preflight from allowed origins passes; disallowed origins get no allow-origin header.
- `test_chat_route.py` — scripted OpenAI response streams as SSE, terminates in `done`; malformed body → 422.
- `test_sql_route.py` — `SELECT 1` → `ok`; `INSERT` → `guard_rejected` with no BQ dry-run.
- `test_datasets_route.py` — scan / vector-search / column listing / 404 on unknown package.
- `test_healthz_readyz.py` — healthz always 200; readyz 503 on any canary failure.
- `test_startup_config.py` — missing `MQAGENT_OPENAI_API_KEY` / `MQAGENT_GCP_PROJECT_ID` refuses to start.

## Continuous deployment

CD is wired via `.github/workflows/deploy-agent-service.yml`. On push to `main` touching `services/agent-service/**` or `services/semantic-enrich/**`:

1. Run `ruff`, `mypy --strict`, `pytest` against the agent-service tree.
2. Build the image with the Dockerfile (context = repo root).
3. Push to Artifact Registry (`us-central1-docker.pkg.dev/<project>/agent-service/agent-service:<sha>`).
4. `gcloud run services update agent-service --image=<sha>`.
5. Poll `/readyz` for 100s to confirm the new revision is serving.

Terraform manages the *shape* of the Cloud Run service (concurrency, scaling, probes, env, secrets); the workflow only rolls the image. `lifecycle.ignore_changes = [template[0].containers[0].image]` on the service keeps Terraform from reverting an image drift when it plans.

### Bootstrap (one-time, per operator)

Auth uses Workload Identity Federation — no long-lived JSON keys. Run once from an operator's box:

```
cd infra/terraform
terraform apply \
  -var "gcp_project_id=<project>" \
  -var "admin_users=[\"you@example.com\"]" \
  -var "agent_service_image=us-central1-docker.pkg.dev/<project>/agent-service/agent-service:bootstrap"
```

Then grab the outputs and paste them into GitHub → Settings → Secrets and variables → Actions → **Variables** (not secrets — WIF resource identifiers are safe to expose):

| GitHub repo variable        | Terraform output                     |
| --------------------------- | ------------------------------------ |
| `MQAGENT_GCP_PROJECT_ID`    | your GCP project id                  |
| `MQAGENT_WIF_PROVIDER`      | `cd_workload_identity_provider`      |
| `MQAGENT_CD_SERVICE_ACCOUNT`| `cd_service_account_email`           |
| `MQAGENT_IMAGE_REPO`        | `agent_service_image_base`           |
| `MQAGENT_REGION`            | `us-central1` (optional; default)    |
| `MQAGENT_SERVICE_NAME`      | `agent-service` (optional; default)  |

The initial `terraform apply` also creates the `openai-api-key` and `mqagent-api-token` secrets *without* versions. Add the initial versions manually:

```
echo -n "$OPENAI_API_KEY"    | gcloud secrets versions add openai-api-key      --data-file=-
echo -n "$MQAGENT_API_TOKEN" | gcloud secrets versions add mqagent-api-token   --data-file=-
```

For the "first ever" deploy the Cloud Run revision needs a real image, not the `:bootstrap` placeholder. Push a manual image once:

```
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -f services/agent-service/Dockerfile -t us-central1-docker.pkg.dev/<project>/agent-service/agent-service:bootstrap .
docker push us-central1-docker.pkg.dev/<project>/agent-service/agent-service:bootstrap
```

From that point on, every push to `main` (touching the service trees) triggers the workflow and deploys the new SHA.

## Observability

- **Cloud Run stdout logs** → Cloud Logging as structlog JSON. `resource.type="cloud_run_revision"`.
- **Structured event of note**: `turn_finished` at the end of every `/chat`, carrying `{conversation_id, tool_calls, dollars, elapsed_ms, terminal_state}`. Grep this to spot-check demo sessions.
- **Cloud Run built-ins**: request count / latency / instance count. No custom metrics for M4.
