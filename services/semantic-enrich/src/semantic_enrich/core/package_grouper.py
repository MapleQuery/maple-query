"""SQL builders + per-row decoders for the candidate query and the
per-package secondary queries against `raw.documents` / `raw.rows`.

Plain string builders, parameter-bound via `bigquery.ScalarQueryParameter`
/ `ArrayQueryParameter`. No SQL templating engine.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.types import PackageResource


def build_candidate_sql(
    *,
    project_id: str,
    dataset_raw: str,
    documents_table: str,
    with_limit: bool,
) -> str:
    """One row per `package_id`, with the list of its loaded resources.

    `with_limit=True` appends `LIMIT @limit_packages`. BQ rejects
    `LIMIT IF(...)`, so the clause is included conditionally rather
    than parameter-toggled.

    Parameter bindings. The Python BQ SDK serialises an empty-list
    `ArrayQueryParameter(...)` as a NULL ARRAY on the wire (not as a
    zero-length array — that's a `bq` CLI quirk only). So "no filter"
    is `@p IS NULL`, and `NOT IN UNNEST(@p)` needs an explicit NULL
    guard or it filters everything (NULL propagates through `NOT IN`).

      - @limit_orgs           ARRAY<STRING> | NULL  (NULL = no filter)
      - @limit_package_ids    ARRAY<STRING> | NULL  (NULL = no filter)
      - @already_extracted    ARRAY<STRING> | NULL  (NULL = no skip)
      - @limit_packages       INT64                 (omitted when with_limit=False)
    """
    fq = f"`{project_id}.{dataset_raw}.{documents_table}`"
    limit_clause = "\nLIMIT @limit_packages" if with_limit else ""
    return f"""
SELECT
  package_id,
  ARRAY_AGG(STRUCT(
    document_id,
    title,
    subjects,
    organization_code,
    file_format,
    resource_last_modified,
    row_count
  ) ORDER BY resource_last_modified DESC NULLS LAST) AS resources
FROM {fq}
WHERE load_status = 'loaded'
  AND package_id IS NOT NULL
  AND (@limit_orgs IS NULL
       OR organization_code IN UNNEST(@limit_orgs))
  AND (@limit_package_ids IS NULL
       OR package_id IN UNNEST(@limit_package_ids))
  AND (@already_extracted IS NULL
       OR package_id NOT IN UNNEST(@already_extracted))
GROUP BY package_id
ORDER BY package_id{limit_clause};
""".strip()


def build_candidate_params(
    *,
    limit_orgs: list[str] | None,
    limit_package_ids: list[str] | None,
    already_extracted: Iterable[str],
    limit_packages: int | None,
) -> list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter]:
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter(
            "limit_orgs", "STRING", list(limit_orgs or [])
        ),
        bigquery.ArrayQueryParameter(
            "limit_package_ids", "STRING", list(limit_package_ids or [])
        ),
        bigquery.ArrayQueryParameter(
            "already_extracted", "STRING", sorted(already_extracted)
        ),
    ]
    if limit_packages is not None:
        params.append(
            bigquery.ScalarQueryParameter(
                "limit_packages", "INT64", limit_packages
            )
        )
    return params


def decode_candidate_row(row: dict[str, Any]) -> tuple[str, tuple[PackageResource, ...]]:
    """Decode one candidate-query row into `(package_id, resources)`."""
    package_id = row["package_id"]
    raw_resources = row.get("resources") or []
    resources = tuple(
        PackageResource(
            document_id=r["document_id"],
            title=r.get("title"),
            subjects=tuple(r.get("subjects") or ()),
            organization_code=r["organization_code"],
            file_format=r["file_format"],
            resource_last_modified=r.get("resource_last_modified"),
            row_count=r.get("row_count"),
        )
        for r in raw_resources
    )
    return package_id, resources


def build_doc_columns_sql(*, project_id: str, dataset_raw: str, rows_table: str) -> str:
    """Per-document column names for a package's documents.

    One row per document is enough — keys are identical across rows of
    one document — so the `QUALIFY ROW_NUMBER()=1` cuts the scan to the
    first row of each doc. The per-document grouping (rather than a
    flattened union) is what lets the representative picker inspect
    each resource's headers; callers derive the package-wide union in
    Python via `column_union`.

    `raw.rows.row` is a native JSON object since the loader's
    double-encoding fix + full reload, so the bare `JSON_KEYS(row)`
    is the correct form. The old `PARSE_JSON(STRING(row))` unwrap
    (still present in some sibling queries) *throws* against object
    rows — `STRING()` requires a JSON string — and must not be used
    here.
    """
    fq = f"`{project_id}.{dataset_raw}.{rows_table}`"
    return f"""
SELECT
  document_id,
  JSON_KEYS(row) AS columns
FROM {fq}
WHERE document_id IN UNNEST(@document_ids)
QUALIFY ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY row_index) = 1
ORDER BY document_id;
""".strip()


def fetch_doc_columns(
    *,
    bq: BqClient,
    project_id: str,
    dataset_raw: str,
    rows_table: str,
    document_ids: list[str],
) -> dict[str, list[str]]:
    """Run the per-doc columns query for one package. Returns
    `document_id -> column names`; documents with no rows are absent."""
    sql = build_doc_columns_sql(
        project_id=project_id, dataset_raw=dataset_raw, rows_table=rows_table
    )
    params = [bigquery.ArrayQueryParameter("document_ids", "STRING", document_ids)]
    return {
        str(r["document_id"]): [str(c) for c in (r.get("columns") or [])]
        for r in bq.query_rows(sql, params=params)
    }


def column_union(columns_by_doc: dict[str, list[str]]) -> list[str]:
    """Sorted distinct column names across a package's documents. Same
    output the flattened SQL union produced (DISTINCT + ORDER BY)."""
    seen: set[str] = set()
    for cols in columns_by_doc.values():
        seen.update(cols)
    return sorted(seen)


def truncate_columns(
    *, names: list[str], cap: int
) -> tuple[tuple[str, ...], int | None]:
    """Apply `sample_column_cap`. Returns `(kept, truncated_to)`."""
    if len(names) <= cap:
        return tuple(names), None
    return tuple(names[:cap]), len(names)


def build_sample_rows_sql(
    *, project_id: str, dataset_raw: str, rows_table: str
) -> str:
    """Sample rows by index from `raw.rows`.

    Uses the preferred QUALIFY-over-row_index path (PRD decision 15).
    `raw.rows.row_index` is REQUIRED, so the OFFSET fallback is not
    needed in current state.
    """
    fq = f"`{project_id}.{dataset_raw}.{rows_table}`"
    return f"""
SELECT row_index, row
FROM {fq}
WHERE document_id = @document_id
  AND row_index IN UNNEST(@indices)
ORDER BY row_index;
""".strip()


def fetch_sample_rows(
    *,
    bq: BqClient,
    project_id: str,
    dataset_raw: str,
    rows_table: str,
    document_id: str,
    indices: list[int],
) -> Iterator[dict[str, Any]]:
    """Run the sample-rows query for one document. Yields one dict per
    row — the `row` JSON decoded into `{header: cell}`."""
    sql = build_sample_rows_sql(
        project_id=project_id, dataset_raw=dataset_raw, rows_table=rows_table
    )
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ScalarQueryParameter("document_id", "STRING", document_id),
        bigquery.ArrayQueryParameter("indices", "INT64", indices),
    ]
    for r in bq.query_rows(sql, params=params):
        raw_row = r["row"]
        # `raw.rows.row` is BQ JSON; the SDK returns it pre-decoded as
        # a dict already. Guard for the legacy string-returning path.
        if isinstance(raw_row, str):
            import json

            raw_row = json.loads(raw_row)
        yield raw_row
