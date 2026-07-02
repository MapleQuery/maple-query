# Multi-stage build so the runtime image ships only the frozen venv and
# the two service trees. Build context = repo root; expected invocation:
#
#   docker build \
#     -f services/agent-service/Dockerfile \
#     -t gcr.io/<project>/agent-service:<sha> \
#     .
#
# The sibling `semantic-enrich` package resolves via a `tool.uv.sources`
# path dependency; both trees are copied in before `uv sync --frozen` so
# the workspace resolves offline.

FROM python:3.12-slim AS builder

# `uv` — reproducible, fast, lockfile-aware. Copied from the pinned
# distroless-style official image so we don't inherit apt bloat.
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the sibling package first so the path dependency resolves.
COPY services/semantic-enrich /app/services/semantic-enrich
COPY services/agent-service /app/services/agent-service

WORKDIR /app/services/agent-service

# `--frozen` fails the build if uv.lock is out of date, catching a
# missed lock update in CI rather than at runtime.
RUN uv sync --frozen --no-dev

# ── Runtime ──────────────────────────────────────────────────────────

FROM python:3.12-slim AS runtime

# Cloud Run's PORT contract. uvicorn honours $MQAGENT_PORT (aliased to
# $PORT in AgentServiceSettings).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/services/agent-service/.venv/bin:$PATH"

WORKDIR /app

# Non-root runtime user. Cloud Run doesn't require this, but a
# defence-in-depth posture pays off when Cloud Run ever runs alongside
# a mounted volume.
RUN groupadd --system --gid 65532 nonroot \
    && useradd --system --uid 65532 --gid nonroot --no-create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app /app

USER nonroot

EXPOSE 8080

# `--host 0.0.0.0` so the container listens on the Cloud Run-injected
# port; `--proxy-headers` so `X-Forwarded-*` is honoured behind Cloud
# Run's front end.
CMD ["python", "-m", "uvicorn", "agent_service.app:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
