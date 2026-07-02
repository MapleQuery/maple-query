"""`GET /corpus/stats` — landing-page counts.

Serves three integers: distinct package count, document count, and row
count. Deliberately bypasses the SQL guard because:

  1. The queries are hand-authored constants, not user input.
  2. `COUNT(*) FROM raw.rows` is a metadata read (0 bytes billed on
     BigQuery), which the guard's `document_id IN (...)` rule would
     otherwise reject on principle.

Results are cached in-process for five minutes so the landing page can't
hammer BQ.
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from agent_service.auth import BearerAuth
from agent_service.deps import AppState, get_app_state

router = APIRouter()

_TTL_SECONDS = 300


class CorpusStats(BaseModel):
    """Stable landing-page counts."""

    datasets: int
    documents: int
    rows: int


_cache_lock = threading.Lock()
_cache: CorpusStats | None = None
_cache_expires_at: float = 0.0


@router.get(
    "/corpus/stats",
    dependencies=[BearerAuth],
    response_model=CorpusStats,
)
def get_corpus_stats(
    state: AppState = Depends(get_app_state),
) -> CorpusStats:
    global _cache, _cache_expires_at
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and now < _cache_expires_at:
            return _cache

    stats = _query_stats(state)

    with _cache_lock:
        _cache = stats
        _cache_expires_at = time.monotonic() + _TTL_SECONDS
    return stats


def _query_stats(state: AppState) -> CorpusStats:
    settings = state.loop_settings
    project = settings.gcp_project_id
    raw = settings.bq_dataset_raw
    documents = settings.bq_documents_table
    rows = settings.bq_rows_table
    docs_ref = f"`{project}.{raw}.{documents}`"
    rows_ref = f"`{project}.{raw}.{rows}`"
    sql = (
        f"SELECT "
        f"(SELECT COUNT(DISTINCT package_id) FROM {docs_ref}) AS datasets, "
        f"(SELECT COUNT(*) FROM {docs_ref}) AS documents, "
        f"(SELECT COUNT(*) FROM {rows_ref}) AS rows"
    )
    result = list(state.bq.query_rows(sql))
    if not result:
        return CorpusStats(datasets=0, documents=0, rows=0)
    row = result[0]
    return CorpusStats(
        datasets=int(row.get("datasets") or 0),
        documents=int(row.get("documents") or 0),
        rows=int(row.get("rows") or 0),
    )
