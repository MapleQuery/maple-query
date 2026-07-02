"""Liveness + readiness routes.

`/healthz` returns 200 unconditionally — Cloud Run uses it to decide
whether to kill the container.

`/readyz` runs three canaries (OpenAI embed, BQ SELECT 1, semantic
snapshot query) and returns 503 if any fails — Cloud Run uses it as
the startup probe so a broken deploy never receives traffic.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from agent_service.deps import AppState, get_app_state

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(
    response: Response,
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """Ping OpenAI, BQ, and the semantic snapshot. 503 on any failure.

    Every failure is surfaced in the JSON body so an operator running
    `curl -v /readyz` can see which subsystem is unhealthy without
    having to trawl Cloud Logging."""
    checks: dict[str, Any] = {}
    ok = True

    try:
        vecs = state.openai_client.embed(["ping"])
        checks["openai"] = {"ok": bool(vecs), "dim": len(vecs[0]) if vecs else 0}
        if not vecs:
            ok = False
    except Exception as exc:
        checks["openai"] = {"ok": False, "error": str(exc)}
        ok = False

    try:
        rows = list(state.bq.query_rows("SELECT 1 AS ok"))
        checks["bq"] = {"ok": bool(rows), "row_count": len(rows)}
        if not rows:
            ok = False
    except Exception as exc:
        checks["bq"] = {"ok": False, "error": str(exc)}
        ok = False

    try:
        snap = state.loop_deps.snapshot_hash_provider()
        checks["semantic_snapshot"] = {"ok": True, "hash_prefix": snap[:12]}
    except Exception as exc:
        checks["semantic_snapshot"] = {"ok": False, "error": str(exc)}
        ok = False

    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "checks": checks}
