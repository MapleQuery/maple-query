"""The warehouse half of the recovery report: read rows, then measure.

Kept apart from `header_recovery_report` so the measurement itself stays
pure and testable without BigQuery, and so the only thing that touches
the warehouse is the part that has to.

Cost is the design constraint. The affected corpus is ~5.3k documents
across ~791 packages, and `raw.rows` is clustered on `document_id`, so a
full pass is hundreds of gigabytes — far past what this measurement is
worth. The scan therefore **dry-runs first and samples down** until the
estimate fits the budget, then reports how many documents it actually
covered and how many it drew from. A recovery rate over a sample is a
measurement; a recovery rate over a sample reported as if it were the
whole corpus is a lie.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.header_recovery import GENERATED_COL_RE
from semantic_enrich.core.header_recovery_report import ScannedDocument
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.header_recovery_scan")

# Documents belonging to packages that carry at least one generated
# column. Ordered by a stable fingerprint so a smaller sample is a
# subset of a larger one and two runs are comparable.
_AFFECTED_DOCS_SQL = """
WITH affected AS (
  SELECT package_id
  FROM `{project_id}.{semantic}.{columns_table}`
  GROUP BY package_id
  HAVING COUNTIF(STARTS_WITH(column_name, '__col_')) > 0
)
SELECT d.document_id, d.package_id, d.title
FROM `{project_id}.{raw}.{documents_table}` d
JOIN affected USING (package_id)
WHERE d.ingestion_status = 'success' AND d.file_format = 'csv'
ORDER BY FARM_FINGERPRINT(d.document_id)
""".strip()

_ROWS_SQL = """
SELECT document_id, row_index, TO_JSON_STRING(row) AS row_json
FROM `{project_id}.{raw}.{rows_table}`
WHERE document_id IN UNNEST(@doc_ids) AND row_index < @n
""".strip()


@dataclass(frozen=True)
class ScanResult:
    documents: list[ScannedDocument]
    bytes_scanned: int
    affected_documents: int
    """The full affected population, before sampling."""


def scan_affected_documents(
    *,
    bq: BqClient,
    settings: Settings,
    max_bytes: int,
    limit: int | None = None,
) -> ScanResult:
    """Read the first `agent_header_scan_rows` rows of as many affected
    documents as `max_bytes` allows.

    Halves the document list until the dry-run estimate fits, rather than
    silently truncating or silently overspending.
    """
    project_id = settings.gcp_project_id
    if not project_id:
        raise ValueError("header recovery report requires a GCP project id")

    candidates = list(
        bq.query_rows(
            _AFFECTED_DOCS_SQL.format(
                project_id=project_id,
                semantic=settings.bq_dataset_semantic,
                columns_table=settings.bq_columns_table,
                raw=settings.bq_dataset_raw,
                documents_table=settings.bq_documents_table,
            )
        )
    )
    affected = len(candidates)
    if limit is not None:
        candidates = candidates[:limit]

    scan_rows = settings.agent_header_scan_rows
    rows_sql = _ROWS_SQL.format(
        project_id=project_id,
        raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
    )
    chosen, estimate = _fit_to_budget(
        bq=bq,
        rows_sql=rows_sql,
        candidates=candidates,
        scan_rows=scan_rows,
        max_bytes=max_bytes,
        dry_run_timeout_ms=settings.agent_sample_rows_timeout_ms,
    )
    if not chosen:
        return ScanResult(
            documents=[], bytes_scanned=0, affected_documents=affected
        )

    doc_ids = [str(r["document_id"]) for r in chosen]
    rows = bq.query_rows(
        rows_sql,
        params=[
            bigquery.ArrayQueryParameter("doc_ids", "STRING", doc_ids),
            bigquery.ScalarQueryParameter("n", "INT64", scan_rows),
        ],
    )
    bodies: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        try:
            body = json.loads(str(row.get("row_json") or ""))
        except ValueError:
            continue
        if not isinstance(body, dict):
            continue
        bodies.setdefault(str(row["document_id"]), {})[
            int(row["row_index"] or 0)
        ] = {str(k): v for k, v in body.items()}

    documents: list[ScannedDocument] = []
    for candidate in chosen:
        doc_id = str(candidate["document_id"])
        by_index = bodies.get(doc_id)
        if not by_index:
            continue
        # Positional row bodies only; a gap would shift every index and
        # make `header_row_index` an off-by-one.
        if set(by_index) != set(range(len(by_index))):
            continue
        ordered = [by_index[i] for i in range(len(by_index))]
        keys: list[str] = []
        for body in ordered:
            for key in body:
                if key not in keys:
                    keys.append(key)
        generated = [k for k in keys if GENERATED_COL_RE.fullmatch(k)]
        if not generated:
            continue
        documents.append(
            ScannedDocument(
                document_id=doc_id,
                package_id=str(candidate.get("package_id") or ""),
                title=(
                    str(candidate["title"])
                    if candidate.get("title") is not None
                    else None
                ),
                generated_columns=generated,
                rows=ordered,
                total_columns=len(keys),
            )
        )
    return ScanResult(
        documents=documents,
        bytes_scanned=estimate,
        affected_documents=affected,
    )


def _fit_to_budget(
    *,
    bq: BqClient,
    rows_sql: str,
    candidates: Sequence[dict[str, object]],
    scan_rows: int,
    max_bytes: int,
    dry_run_timeout_ms: int,
    max_attempts: int = 12,
) -> tuple[list[dict[str, object]], int]:
    """Halve the document list until the dry run fits `max_bytes`.

    Returns the chosen documents and the estimate they cost. An empty
    list means even one document did not fit, which is a budget worth
    questioning rather than a scan worth running.
    """
    chosen = list(candidates)
    for _ in range(max_attempts):
        if not chosen:
            return [], 0
        estimate = bq.dry_run_bytes(
            rows_sql,
            params=[
                bigquery.ArrayQueryParameter(
                    "doc_ids",
                    "STRING",
                    [str(r["document_id"]) for r in chosen],
                ),
                bigquery.ScalarQueryParameter("n", "INT64", scan_rows),
            ],
            timeout_ms=dry_run_timeout_ms,
        )
        if estimate <= max_bytes:
            _LOG.info(
                "header_recovery_scan_fits",
                documents=len(chosen),
                estimate_bytes=estimate,
                max_bytes=max_bytes,
            )
            return chosen, estimate
        _LOG.info(
            "header_recovery_scan_over_budget",
            documents=len(chosen),
            estimate_bytes=estimate,
            max_bytes=max_bytes,
        )
        chosen = chosen[: len(chosen) // 2]
    return [], 0
