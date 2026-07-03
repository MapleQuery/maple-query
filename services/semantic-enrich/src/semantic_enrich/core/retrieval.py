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
from dataclasses import dataclass, field
from datetime import datetime
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
class DocumentCandidate:
    """One `raw.documents` row for a candidate package.

    Threaded verbatim into the SQL-gen prompt so the model can inline a
    literal `document_id IN ('…', '…')` filter. That literal shape is
    the only one BigQuery can plan-time-prune against `raw.rows`'
    `document_id` cluster; a subquery IN — even against `raw.documents` —
    forces a full clustered scan.

    `columns` is the doc's actual JSON key set, drawn from the first
    row's parsed body. Different docs in the same CKAN package can have
    entirely disjoint column sets (bilingual pairs, header-parse
    failures collapsing to `__col_N`, resource variants). Exposing the
    per-doc set lets the model pair the right column with the right
    doc instead of picking columns from the package-wide union.

    `column_samples` maps each column name to up to 3 distinct non-empty
    sample values drawn from the first few rows of the doc. A column
    with no non-empty values across the sampled rows has an empty tuple.
    Threaded into the `list_documents` tool payload so the model can
    spot mislabeled columns before writing SQL — e.g. a "column" named
    `Canada_Mortgage_and_Housing_Corporation` whose samples are all
    `"Total"` / `"Internal Services"` is a garbage-parsed section
    header, not a corporate-name column.
    """

    document_id: str
    package_id: str
    title: str | None
    row_count: int | None
    resource_last_modified: datetime | None
    columns: tuple[str, ...]
    column_samples: dict[str, tuple[str, ...]] = field(default_factory=dict)


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


def retrieve_documents(
    *,
    bq: BqClient,
    package_ids: list[str],
    settings: Settings,
) -> tuple[list[DocumentCandidate], int]:
    """Fetch loaded `raw.documents` rows for the candidate packages.

    Filter posture:
    - `load_status = 'loaded'` — the strictest signal that rows landed
      in `raw.rows`. Sibling code (`column_inputs`, `package_grouper`,
      `sample_selector`) uses the same filter; it subsumes
      `ingestion_status='success'` and excludes 'pending', 'blob_missing',
      'parse_failed', 'skipped_non_csv'.
    - Ordered by `resource_last_modified DESC` so "most recent" questions
      get the freshest doc first when the model has to pick.
    - Capped per-package at `settings.eval_max_documents_per_package` to
      keep the literal IN-list tractable in the prompt.

    Returns `(candidates, latency_ms)`.
    """
    project_id = _require_project(settings)
    docs_sql = _DOCUMENT_SEARCH_SQL.format(
        project_id=project_id,
        dataset=settings.bq_dataset_raw,
        table=settings.bq_documents_table,
    )
    docs_params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("package_ids", "STRING", package_ids),
        bigquery.ScalarQueryParameter(
            "max_per_package",
            "INT64",
            settings.eval_max_documents_per_package,
        ),
    ]
    started = time.monotonic()
    doc_rows = list(bq.query_rows(docs_sql, params=docs_params))
    doc_ids = [str(r["document_id"]) for r in doc_rows]
    keys_by_doc, samples_by_doc = _fetch_doc_column_data(
        bq=bq, project_id=project_id, doc_ids=doc_ids, settings=settings
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return [
        _document_from_row(
            r,
            keys_by_doc.get(str(r["document_id"]), []),
            samples_by_doc.get(str(r["document_id"]), {}),
        )
        for r in doc_rows
    ], latency_ms


# The doc/column pairing check and the tool payload both need the
# per-doc column set; the payload additionally wants a few sample values
# per column so the model can catch mislabeled columns (a `Corp_Name`
# header whose real values are section labels like `"Total"`). Sampling
# 5 rows per doc gives 2-3 non-null samples per column even when the
# first row has empty cells, without inflating the query cost — the
# scan is cluster-pruned by document_id.
_SAMPLE_ROWS_PER_DOC = 5
_MAX_SAMPLES_PER_COLUMN = 3


def _fetch_doc_column_data(
    *,
    bq: BqClient,
    project_id: str,
    doc_ids: list[str],
    settings: Settings,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Fetch the ordered column set and sample values for each doc.

    Returns `(columns_by_doc, samples_by_doc)`:
    - `columns_by_doc[doc_id]` — ordered column names (first-seen order
      across the sampled rows). Used by the doc/column pairing check.
    - `samples_by_doc[doc_id][col]` — up to `_MAX_SAMPLES_PER_COLUMN`
      distinct non-empty stringified sample values for `col`. Empty
      list when every sampled row's value for that column is null or
      empty. Threaded into the tool payload.

    Reads the first `_SAMPLE_ROWS_PER_DOC` rows per doc from `raw.rows`
    via a literal IN-list on `document_id` (cluster-pruned). The row
    body is stored as a JSON-string scalar (loader double-encodes), so
    `PARSE_JSON(STRING(row))` unwraps it into a real JSON object that
    the BigQuery client hands back to Python as a dict — same shape as
    `sample_rows` uses.
    """
    if not doc_ids:
        return {}, {}
    sql = _DOC_COLUMN_DATA_SQL.format(
        project_id=project_id,
        dataset=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
    )
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("document_ids", "STRING", doc_ids),
        bigquery.ScalarQueryParameter(
            "sample_rows", "INT64", _SAMPLE_ROWS_PER_DOC
        ),
    ]

    columns_by_doc: dict[str, list[str]] = {}
    samples_by_doc: dict[str, dict[str, list[str]]] = {}
    seen_by_doc: dict[str, set[str]] = {}
    for row in bq.query_rows(sql, params=params):
        doc_id = str(row["document_id"])
        body = row.get("row_body")
        if not isinstance(body, dict):
            # PARSE_JSON returns None for invalid JSON, and non-object
            # bodies (arrays, scalars) can't contribute keys.
            continue
        seen = seen_by_doc.setdefault(doc_id, set())
        ordered = columns_by_doc.setdefault(doc_id, [])
        samples = samples_by_doc.setdefault(doc_id, {})
        for key, value in body.items():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                ordered.append(key_str)
                samples[key_str] = []
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            col_samples = samples[key_str]
            if (
                text not in col_samples
                and len(col_samples) < _MAX_SAMPLES_PER_COLUMN
            ):
                col_samples.append(text)
    return columns_by_doc, samples_by_doc


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


def _document_from_row(
    row: dict[str, Any],
    columns: list[str],
    column_samples: dict[str, list[str]],
) -> DocumentCandidate:
    row_count = row.get("row_count")
    return DocumentCandidate(
        document_id=str(row["document_id"]),
        package_id=str(row["package_id"]),
        title=_optional_str(row.get("title")),
        row_count=int(row_count) if row_count is not None else None,
        resource_last_modified=_optional_datetime(
            row.get("resource_last_modified")
        ),
        columns=tuple(columns),
        column_samples={
            col: tuple(column_samples.get(col, [])) for col in columns
        },
    )


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return None


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


_DOCUMENT_SEARCH_SQL = """
WITH ranked AS (
  SELECT
    document_id,
    package_id,
    title,
    row_count,
    resource_last_modified,
    ROW_NUMBER() OVER (
      PARTITION BY package_id
      ORDER BY resource_last_modified DESC NULLS LAST, document_id
    ) AS rn
  FROM `{project_id}.{dataset}.{table}`
  WHERE package_id IN UNNEST(@package_ids)
    AND load_status = 'loaded'
)
SELECT document_id, package_id, title, row_count, resource_last_modified
FROM ranked
WHERE rn <= @max_per_package
ORDER BY package_id, rn
""".strip()


# Two-query approach because `WHERE document_id IN (SELECT ...)` would
# not plan-time-prune the `raw.rows` cluster (same failure mode 3.3.1
# documents). The literal IN-list from the docs query is small and
# cluster-pruned in the second query. Ordering the QUALIFY window by
# `row_index` picks the leading N rows per doc; Python then flattens
# them into an ordered key set plus per-key samples.
_DOC_COLUMN_DATA_SQL = """
SELECT
  document_id,
  row_index,
  PARSE_JSON(STRING(row)) AS row_body
FROM `{project_id}.{dataset}.{rows_table}`
WHERE document_id IN UNNEST(@document_ids)
QUALIFY ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY row_index) <= @sample_rows
ORDER BY document_id, row_index
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
