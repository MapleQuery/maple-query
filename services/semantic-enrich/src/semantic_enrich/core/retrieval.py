"""Question-time retrieval: OpenAI embed → BQ VECTOR_SEARCH.

Two-stage:
  1. Embed the NL question via OpenAI (same model 4.7 wrote the warehouse
     with; dim asserted).
  2. `VECTOR_SEARCH` over `semantic.datasets` → top-k package candidates,
     then a scoped `VECTOR_SEARCH` over `semantic.columns` restricted to
     those package_ids → top-k column candidates.

Naïve cosine over `ARRAY<FLOAT64>`, brute-force scan. No IVF index at
this corpus size — parent §3.4 commits to this posture; revisit when
`semantic.columns` exceeds ~500K rows.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings


@dataclass(frozen=True)
class PackageCandidate:
    """One VECTOR_SEARCH hit against `semantic.datasets`.

    Fields land verbatim in the SQL-gen prompt; the runner picks the
    top-k list up as-is and passes it through.
    """

    package_id: str
    summary: str
    grain: str | None
    measures: tuple[str, ...]
    dimensions: tuple[str, ...]
    date_range_start: str | None
    date_range_end: str | None
    distance: float


@dataclass(frozen=True)
class ColumnCandidate:
    """One VECTOR_SEARCH hit against `semantic.columns`, scoped to the
    packages surfaced by the package pass."""

    package_id: str
    column_name: str
    semantic_type: str | None
    description: str
    sample_values: tuple[str, ...]
    distance: float


@dataclass(frozen=True)
class RetrievalTiming:
    """Latencies threaded up to the runner for the per-question grade."""

    embed_ms: int
    packages_ms: int
    columns_ms: int


def embed_question(
    *, openai_client: OpenAIClient, question: str, settings: Settings
) -> list[float]:
    """Embed one NL question. Asserts the returned dim matches the
    warehouse. A mismatch is a config drift between 4.7 (warehouse) and
    4.6 (query time) — abort loudly rather than silently returning
    junk-distance neighbours."""
    vectors = openai_client.embed([question])
    if len(vectors) != 1:
        raise RuntimeError(
            f"openai.embed returned {len(vectors)} vectors for 1 input"
        )
    dim = len(vectors[0])
    if dim != settings.openai_embedding_dim:
        raise RuntimeError(
            f"openai.embed returned dim={dim}, expected "
            f"{settings.openai_embedding_dim}; check "
            "WHENRICH_OPENAI_EMBEDDING_MODEL / WHENRICH_OPENAI_EMBEDDING_DIM."
        )
    return vectors[0]


def retrieve_packages(
    *,
    bq: BqClient,
    question_vec: list[float],
    settings: Settings,
) -> tuple[list[PackageCandidate], int]:
    """Top-k package candidates by cosine distance. Returns
    `(candidates, latency_ms)`."""
    project_id = _require_project(settings)
    sql = _PACKAGE_SEARCH_SQL.format(
        project_id=project_id,
        dataset=settings.bq_dataset_semantic,
        table=settings.bq_datasets_table,
    )
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("question_vec", "FLOAT64", question_vec),
        bigquery.ScalarQueryParameter(
            "k_packages", "INT64", settings.eval_k_packages
        ),
    ]
    started = time.monotonic()
    rows = list(bq.query_rows(sql, params=params))
    latency_ms = int((time.monotonic() - started) * 1000)
    return [_package_from_row(r) for r in rows], latency_ms


def retrieve_columns(
    *,
    bq: BqClient,
    question_vec: list[float],
    scoped_packages: list[str],
    settings: Settings,
) -> tuple[list[ColumnCandidate], int]:
    """Top-k column candidates scoped to `scoped_packages`. Returns
    `(candidates, latency_ms)`."""
    project_id = _require_project(settings)
    sql = _COLUMN_SEARCH_SQL.format(
        project_id=project_id,
        dataset=settings.bq_dataset_semantic,
        table=settings.bq_columns_table,
    )
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("question_vec", "FLOAT64", question_vec),
        bigquery.ArrayQueryParameter(
            "scoped_packages", "STRING", scoped_packages
        ),
        bigquery.ScalarQueryParameter(
            "k_columns", "INT64", settings.eval_k_columns
        ),
    ]
    started = time.monotonic()
    rows = list(bq.query_rows(sql, params=params))
    latency_ms = int((time.monotonic() - started) * 1000)
    return [_column_from_row(r) for r in rows], latency_ms


def _require_project(settings: Settings) -> str:
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "the retrieval harness; VECTOR_SEARCH queries fully-qualified tables."
        )
    return settings.gcp_project_id


def _package_from_row(row: dict[str, Any]) -> PackageCandidate:
    return PackageCandidate(
        package_id=str(row["package_id"]),
        summary=str(row.get("summary") or ""),
        grain=_optional_str(row.get("grain")),
        measures=tuple(_str_list(row.get("measures"))),
        dimensions=tuple(_str_list(row.get("dimensions"))),
        date_range_start=_optional_str(row.get("date_range_start")),
        date_range_end=_optional_str(row.get("date_range_end")),
        distance=float(row.get("distance") or 0.0),
    )


def _column_from_row(row: dict[str, Any]) -> ColumnCandidate:
    return ColumnCandidate(
        package_id=str(row["package_id"]),
        column_name=str(row["column_name"]),
        semantic_type=_optional_str(row.get("semantic_type")),
        description=str(row.get("description") or ""),
        sample_values=tuple(_str_list(row.get("sample_values"))),
        distance=float(row.get("distance") or 0.0),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _str_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(v) for v in value]


# VECTOR_SEARCH exposes the base-row columns as `base.<col>` in the
# SELECT list. We flatten to the underlying column names so downstream
# consumers see the same shape as a plain SELECT.
_PACKAGE_SEARCH_SQL = """
SELECT
  base.package_id AS package_id,
  base.summary AS summary,
  base.grain AS grain,
  base.measures AS measures,
  base.dimensions AS dimensions,
  base.date_range_start AS date_range_start,
  base.date_range_end AS date_range_end,
  distance
FROM VECTOR_SEARCH(
  TABLE `{project_id}.{dataset}.{table}`,
  'embedding',
  (SELECT @question_vec AS embedding),
  top_k => @k_packages,
  distance_type => 'COSINE'
)
ORDER BY distance ASC
""".strip()


_COLUMN_SEARCH_SQL = """
SELECT
  base.package_id AS package_id,
  base.column_name AS column_name,
  base.semantic_type AS semantic_type,
  base.description AS description,
  base.sample_values AS sample_values,
  distance
FROM VECTOR_SEARCH(
  (SELECT * FROM `{project_id}.{dataset}.{table}`
   WHERE package_id IN UNNEST(@scoped_packages)),
  'embedding',
  (SELECT @question_vec AS embedding),
  top_k => @k_columns,
  distance_type => 'COSINE'
)
ORDER BY distance ASC
""".strip()
