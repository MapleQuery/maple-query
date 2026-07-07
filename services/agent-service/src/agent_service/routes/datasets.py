"""`GET /datasets`, `GET /datasets/{package_id}/columns`, and
`GET /datasets/{package_id}/documents`.

Serves the landing surface and explorer's dataset picker. When `q` is
present, embeds the query via OpenAI and runs `VECTOR_SEARCH`; when
absent, returns a straight scan ordered by `generated_at DESC`. The
documents endpoint lists a package's loaded source files from
`raw.documents` for the detail page's download links.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import bigquery
from pydantic import BaseModel, Field
from semantic_enrich.core.retrieval import embed_question, retrieve_packages

from agent_service.auth import BearerAuth
from agent_service.deps import AppState, get_app_state

router = APIRouter()


class DatasetCard(BaseModel):
    """One row in the `/datasets` response. `distance` is populated only
    when `q` was present (i.e. VECTOR_SEARCH was used)."""

    package_id: str
    title: str | None = None
    summary: str = ""
    grain: str | None = None
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    date_range_start: str | None = None
    date_range_end: str | None = None
    distance: float | None = None


class DatasetsResponse(BaseModel):
    datasets: list[DatasetCard]
    total: int


class ColumnCard(BaseModel):
    column_name: str
    semantic_type: str | None = None
    description: str = ""
    sample_values: list[str] = Field(default_factory=list)


class ColumnsResponse(BaseModel):
    package_id: str
    columns: list[ColumnCard]


class DocumentCard(BaseModel):
    """One loaded `raw.documents` row for a package. `source_url` is
    the open.canada.ca download link — raw CSVs are publicly
    downloadable there, so no GCS signed URL is exposed."""

    document_id: str
    title: str | None = None
    source_url: str
    file_format: str
    language: str
    row_count: int | None = None
    published_date: str | None = None
    is_representative: bool = False


class DocumentsResponse(BaseModel):
    package_id: str
    documents: list[DocumentCard]


@router.get(
    "/datasets",
    dependencies=[BearerAuth],
    response_model=DatasetsResponse,
)
def list_datasets(
    state: AppState = Depends(get_app_state),
    q: str | None = Query(default=None),
    limit: int = Query(default=0, ge=0, le=100),
    offset: int = Query(default=0, ge=0),
) -> DatasetsResponse:
    service_settings = state.service_settings
    resolved_limit = (
        limit if limit > 0 else service_settings.datasets_default_limit
    )
    resolved_limit = min(resolved_limit, service_settings.datasets_max_limit)

    project_id = state.loop_settings.gcp_project_id
    dataset = state.loop_settings.bq_dataset_semantic
    table = state.loop_settings.bq_datasets_table
    total = _count_datasets(state, project_id, dataset, table)

    if q:
        return DatasetsResponse(
            datasets=_search_datasets(state, q, resolved_limit),
            total=total,
        )
    return DatasetsResponse(
        datasets=_scan_datasets(
            state, project_id, dataset, table, resolved_limit, offset
        ),
        total=total,
    )


@router.get(
    "/datasets/{package_id}",
    dependencies=[BearerAuth],
    response_model=DatasetCard,
)
def get_dataset(
    package_id: str,
    state: AppState = Depends(get_app_state),
) -> DatasetCard:
    """Single semantic row by exact package_id. The detail page uses this
    for the title + measures/dimensions/coverage tiles; searching by the
    UUID via `/datasets?q=` was unreliable (a UUID is not semantically
    meaningful, so VECTOR_SEARCH rarely returned the row itself)."""
    project_id = state.loop_settings.gcp_project_id
    dataset = state.loop_settings.bq_dataset_semantic
    table = state.loop_settings.bq_datasets_table
    sql = (
        f"SELECT package_id, title, summary, grain, measures, dimensions, "
        f"date_range_start, date_range_end "
        f"FROM `{project_id}.{dataset}.{table}` "
        f"WHERE package_id = @pkg LIMIT 1"
    )
    params = [bigquery.ScalarQueryParameter("pkg", "STRING", package_id)]
    for r in state.bq.query_rows(sql, params=params):
        return DatasetCard(
            package_id=str(r["package_id"]),
            title=_optional_str(r.get("title")),
            summary=str(r.get("summary") or ""),
            grain=_optional_str(r.get("grain")),
            measures=[str(v) for v in (r.get("measures") or [])],
            dimensions=[str(v) for v in (r.get("dimensions") or [])],
            date_range_start=_optional_str(r.get("date_range_start")),
            date_range_end=_optional_str(r.get("date_range_end")),
        )
    raise HTTPException(status_code=404, detail="package_not_found")


@router.get(
    "/datasets/{package_id}/columns",
    dependencies=[BearerAuth],
    response_model=ColumnsResponse,
)
def list_columns(
    package_id: str,
    state: AppState = Depends(get_app_state),
) -> ColumnsResponse:
    project_id = state.loop_settings.gcp_project_id
    dataset = state.loop_settings.bq_dataset_semantic
    table = state.loop_settings.bq_columns_table
    sql = (
        f"SELECT column_name, semantic_type, description, sample_values "
        f"FROM `{project_id}.{dataset}.{table}` "
        f"WHERE package_id = @pkg "
        f"ORDER BY column_name"
    )
    params = [bigquery.ScalarQueryParameter("pkg", "STRING", package_id)]
    rows = list(state.bq.query_rows(sql, params=params))
    # Distinguish "no columns yet" from "unknown package_id" by peeking
    # at semantic.datasets when the column query came back empty — a
    # 404 is more actionable for the explorer than an empty column list.
    if not rows and not _package_exists(state, package_id):
        raise HTTPException(status_code=404, detail="package_not_found")
    columns = [
        ColumnCard(
            column_name=str(r.get("column_name") or ""),
            semantic_type=_optional_str(r.get("semantic_type")),
            description=str(r.get("description") or ""),
            sample_values=[str(v) for v in (r.get("sample_values") or [])],
        )
        for r in rows
    ]
    return ColumnsResponse(package_id=package_id, columns=columns)


@router.get(
    "/datasets/{package_id}/documents",
    dependencies=[BearerAuth],
    response_model=DocumentsResponse,
)
def list_source_documents(
    package_id: str,
    state: AppState = Depends(get_app_state),
) -> DocumentsResponse:
    project_id = state.loop_settings.gcp_project_id
    dataset_raw = state.loop_settings.bq_dataset_raw
    documents_table = state.loop_settings.bq_documents_table
    sql = (
        f"SELECT document_id, title, source_url, file_format, language, "
        f"row_count, published_date "
        f"FROM `{project_id}.{dataset_raw}.{documents_table}` "
        f"WHERE package_id = @pkg AND load_status = 'loaded' "
        # BQ sorts NULLs last under DESC, so unloaded-count rows (should
        # not appear under load_status='loaded') can't shadow real docs.
        f"ORDER BY row_count DESC, document_id"
    )
    params = [bigquery.ScalarQueryParameter("pkg", "STRING", package_id)]
    rows = list(state.bq.query_rows(sql, params=params))
    # Same 404 posture as the columns endpoint: distinguish "no loaded
    # documents" from "unknown package_id" via semantic.datasets.
    if not rows and not _package_exists(state, package_id):
        raise HTTPException(status_code=404, detail="package_not_found")
    rep_doc_id = _representative_document_id(state, package_id) if rows else None
    documents = [
        DocumentCard(
            document_id=str(r.get("document_id") or ""),
            title=_optional_str(r.get("title")),
            source_url=str(r.get("source_url") or ""),
            file_format=str(r.get("file_format") or ""),
            language=str(r.get("language") or "unknown"),
            row_count=int(r["row_count"]) if r.get("row_count") is not None else None,
            published_date=_optional_str(r.get("published_date")),
            is_representative=(
                rep_doc_id is not None and r.get("document_id") == rep_doc_id
            ),
        )
        for r in rows
    ]
    return DocumentsResponse(package_id=package_id, documents=documents)


def _search_datasets(
    state: AppState, q: str, k: int
) -> list[DatasetCard]:
    vec = embed_question(
        openai_client=state.openai_client,
        question=q,
        settings=state.loop_settings,
    )
    scoped = state.loop_settings.model_copy(update={"eval_k_packages": k})
    packages, _latency = retrieve_packages(
        bq=state.bq, question_vec=vec, settings=scoped
    )
    return [
        DatasetCard(
            package_id=p.package_id,
            title=p.title,
            summary=p.summary,
            grain=p.grain,
            measures=list(p.measures),
            dimensions=list(p.dimensions),
            date_range_start=p.date_range_start,
            date_range_end=p.date_range_end,
            distance=p.distance,
        )
        for p in packages
    ]


def _scan_datasets(
    state: AppState,
    project_id: str | None,
    dataset: str,
    table: str,
    limit: int,
    offset: int,
) -> list[DatasetCard]:
    sql = (
        f"SELECT package_id, title, summary, grain, measures, dimensions, "
        f"date_range_start, date_range_end "
        f"FROM `{project_id}.{dataset}.{table}` "
        f"ORDER BY generated_at DESC "
        f"LIMIT @limit OFFSET @offset"
    )
    params = [
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
        bigquery.ScalarQueryParameter("offset", "INT64", offset),
    ]
    rows = state.bq.query_rows(sql, params=params)
    return [
        DatasetCard(
            package_id=str(r["package_id"]),
            title=_optional_str(r.get("title")),
            summary=str(r.get("summary") or ""),
            grain=_optional_str(r.get("grain")),
            measures=[str(v) for v in (r.get("measures") or [])],
            dimensions=[str(v) for v in (r.get("dimensions") or [])],
            date_range_start=_optional_str(r.get("date_range_start")),
            date_range_end=_optional_str(r.get("date_range_end")),
        )
        for r in rows
    ]


def _count_datasets(
    state: AppState, project_id: str | None, dataset: str, table: str
) -> int:
    sql = f"SELECT COUNT(*) AS n FROM `{project_id}.{dataset}.{table}`"
    rows = list(state.bq.query_rows(sql))
    if not rows:
        return 0
    return int(rows[0].get("n") or 0)


def _representative_document_id(
    state: AppState, package_id: str
) -> str | None:
    """The document the enrichment pass described, stamped onto
    `semantic.datasets` by the picker. None for packages enriched
    before the column was backfilled — no badge is shown then."""
    project_id = state.loop_settings.gcp_project_id
    dataset = state.loop_settings.bq_dataset_semantic
    table = state.loop_settings.bq_datasets_table
    sql = (
        f"SELECT representative_document_id "
        f"FROM `{project_id}.{dataset}.{table}` "
        f"WHERE package_id = @pkg LIMIT 1"
    )
    params = [bigquery.ScalarQueryParameter("pkg", "STRING", package_id)]
    for row in state.bq.query_rows(sql, params=params):
        return _optional_str(row.get("representative_document_id"))
    return None


def _package_exists(state: AppState, package_id: str) -> bool:
    project_id = state.loop_settings.gcp_project_id
    dataset = state.loop_settings.bq_dataset_semantic
    table = state.loop_settings.bq_datasets_table
    sql = (
        f"SELECT 1 FROM `{project_id}.{dataset}.{table}` "
        f"WHERE package_id = @pkg LIMIT 1"
    )
    params = [bigquery.ScalarQueryParameter("pkg", "STRING", package_id)]
    return any(True for _ in state.bq.query_rows(sql, params=params))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
