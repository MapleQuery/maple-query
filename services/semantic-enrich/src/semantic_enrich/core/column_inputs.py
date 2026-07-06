"""`columns-extract` orchestrator.

Laptop-side. Reads `raw.documents`, `raw.rows`, and
`semantic.datasets`, materialises one `ColumnInputs` per candidate
package into `stage/<run_id>/column_inputs/*.jsonl`. The GPU box
reads that JSONL after an rsync; it never touches BQ itself.

Departures from PRD §5.4 for runtime cost:
- Sample values are gathered in **one batched query per package**
  rather than one query per (package, column). At ~150K columns
  across ~3,693 packages, the per-column path is ~150K BQ round-trips
  (~hours even at 16-way concurrency); the batched path is one query
  per package (~minutes).
- JSON paths use bracket notation `$["<col>"]` rather than dotted
  `$.<col>`. The allowlist admits hyphens/dots/slashes/spaces, but BQ
  `JSON_VALUE` only accepts those when bracket-quoted. The PRD's
  dotted form would silently miss values for any non-identifier
  column name.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from google.api_core import exceptions as gax
from google.cloud import bigquery

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import package_grouper, stage_io
from semantic_enrich.core.sample_selector import looks_like_dictionary
from semantic_enrich.providers.logging import get_logger
from semantic_enrich.types import (
    ColumnInputs,
    ColumnsExtractRunSummary,
    Counters,
)

# Cap on `column_name_dropped_by_allowlist` events emitted per package
# (the counter still tracks the rest). 10 is enough to characterise
# the drop pattern without swamping logs for a pathological CSV.
_DROP_EVENT_CAP_PER_PACKAGE = 10


@dataclass(frozen=True)
class ColumnsExtractRequest:
    run_id: str
    dry_run: bool
    limit_packages: int | None
    limit_package_ids: list[str] | None
    limit_orgs: list[str] | None


@dataclass(frozen=True)
class _PackageOutcome:
    """Per-package result returned from a worker thread back to the
    main loop, which is the only thread that touches the StageWriter
    and run-level counters."""

    kind: Literal["extracted", "empty"]
    package_id: str
    inputs: ColumnInputs | None
    dropped_count: int
    summary_present: bool


def preflight(*, settings: Settings, bq: BqClient) -> None:
    """Fail fast before the candidate query."""
    if not settings.gcp_project_id:
        raise RuntimeError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) must be set for "
            "columns-extract; this subcommand reads from BigQuery."
        )
    log = get_logger("semantic_enrich.column_inputs")
    try:
        list(bq.query_rows("SELECT 1 AS ok"))
    except gax.GoogleAPICallError as exc:
        log.error("bq_auth_failed", error=str(exc))
        raise RuntimeError(f"bq_auth_failed: {exc}") from exc
    if not settings.staging_dir.parent.exists():
        raise RuntimeError(
            f"staging dir parent does not exist: {settings.staging_dir.parent}"
        )


def run_extract(
    *,
    request: ColumnsExtractRequest,
    settings: Settings,
    bq: BqClient,
    logger: structlog.BoundLogger | None = None,
) -> ColumnsExtractRunSummary:
    """End-to-end: candidate query → per-package fan-out →
    `stage/<run_id>/column_inputs/*.jsonl`."""
    log = logger or get_logger("semantic_enrich.column_inputs")
    started = time.monotonic()

    log.info(
        "columns_extract_start",
        run_id=request.run_id,
        dry_run=request.dry_run,
        limit_packages=request.limit_packages,
        limit_package_ids=request.limit_package_ids,
        limit_orgs=request.limit_orgs,
        staging_dir=str(settings.staging_dir),
    )

    already_extracted = stage_io.read_staged_package_ids(
        run_id=request.run_id,
        artifact="column_inputs",
        staging_dir=settings.staging_dir,
    )

    project_id = settings.gcp_project_id
    assert project_id is not None

    candidate_rows = list(
        bq.query_rows(
            _build_candidate_sql(
                project_id=project_id,
                dataset_raw=settings.bq_dataset_raw,
                documents_table=settings.bq_documents_table,
                with_limit=request.limit_packages is not None,
            ),
            params=_build_candidate_params(
                limit_orgs=request.limit_orgs,
                limit_package_ids=request.limit_package_ids,
                already_extracted=already_extracted,
                limit_packages=request.limit_packages,
            ),
        )
    )
    log.info(
        "columns_extract_candidates",
        run_id=request.run_id,
        candidate_count=len(candidate_rows),
    )

    # Batched cross-pass lookup (§5.5): one query for the entire
    # candidate set, not per-package.
    summary_by_pkg = _fetch_dataset_summaries(
        bq=bq,
        project_id=project_id,
        dataset_semantic=settings.bq_dataset_semantic,
        datasets_table=settings.bq_datasets_table,
        package_ids=[r["package_id"] for r in candidate_rows],
        log=log,
    )

    allowlist = re.compile(settings.column_name_allowlist_re)
    counters = Counters()
    counters.extras["skipped_already_extracted"] = len(already_extracted)
    counters.extras["empty"] = 0
    counters.extras["summary_hit"] = 0
    counters.extras["summary_miss"] = 0
    counters.extras["dropped_by_allowlist"] = 0

    def _on_flush(path: Path, seq: int, row_count: int) -> None:
        log.info(
            "column_inputs_flush_written",
            run_id=request.run_id,
            path=str(path),
            flush_seq=seq,
            row_count=row_count,
        )

    writer = stage_io.StageWriter(
        run_id=request.run_id,
        artifact="column_inputs",
        staging_dir=settings.staging_dir,
        flush_every=settings.flush_every_n_packages,
        on_flush=_on_flush,
    )

    workers = max(1, settings.extract_concurrency)
    log.info(
        "columns_extract_pool_started", run_id=request.run_id, workers=workers
    )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _extract_one_package,
                    raw_row=row,
                    bq=bq,
                    project_id=project_id,
                    settings=settings,
                    allowlist=allowlist,
                    package_summary=summary_by_pkg.get(row["package_id"]),
                    log=log,
                )
                for row in candidate_rows
            ]
            for row, future in zip(candidate_rows, futures, strict=True):
                outcome = future.result()
                counters.extras["dropped_by_allowlist"] += outcome.dropped_count
                if outcome.summary_present:
                    counters.extras["summary_hit"] += 1
                else:
                    counters.extras["summary_miss"] += 1
                if outcome.kind == "empty":
                    counters.extras["empty"] += 1
                    continue
                assert outcome.inputs is not None
                writer.append(outcome.inputs)
                counters.generated += 1
                if request.dry_run:
                    log.bind(package_id=outcome.package_id).info(
                        "would_have_extracted_columns",
                        column_count=len(outcome.inputs.column_names),
                    )
                # Reference `row` to silence unused-loop-variable lint;
                # the future-zip is the intentional pairing.
                _ = row
    finally:
        writer.close()

    duration_ms = int((time.monotonic() - started) * 1000)

    summary = ColumnsExtractRunSummary(
        run_id=request.run_id,
        dry_run=request.dry_run,
        candidate_count=(
            len(candidate_rows) + counters.extras["skipped_already_extracted"]
        ),
        packages_extracted=counters.generated,
        packages_skipped_already_extracted=counters.extras[
            "skipped_already_extracted"
        ],
        packages_empty=counters.extras["empty"],
        packages_summary_hit=counters.extras["summary_hit"],
        packages_summary_miss=counters.extras["summary_miss"],
        columns_dropped_by_allowlist=counters.extras["dropped_by_allowlist"],
        flush_files_written=writer.files_written,
        duration_ms=duration_ms,
    )
    _assert_invariant(summary, log)
    log.info(
        "columns_extract_finish",
        run_id=request.run_id,
        duration_ms=duration_ms,
        summary=summary.__dict__,
    )
    return summary


# ── Per-package work (worker-thread-safe) ──


def _extract_one_package(
    *,
    raw_row: dict[str, object],
    bq: BqClient,
    project_id: str,
    settings: Settings,
    allowlist: re.Pattern[str],
    package_summary: str | None,
    log: structlog.BoundLogger,
) -> _PackageOutcome:
    """Per-package work: decode candidate row, scan column union, fan
    out to sample-values, assemble `ColumnInputs`."""
    package_id = str(raw_row["package_id"])
    plog = log.bind(package_id=package_id)
    raw_resources = raw_row.get("resources") or []
    if not isinstance(raw_resources, list):
        raw_resources = []
    resources: list[dict[str, object]] = list(raw_resources)
    if not resources:
        plog.warning("package_columns_inputs_empty", reason="no_resources")
        return _PackageOutcome(
            kind="empty",
            package_id=package_id,
            inputs=None,
            dropped_count=0,
            summary_present=package_summary is not None,
        )

    document_ids = [str(r["document_id"]) for r in resources]
    columns_by_doc = package_grouper.fetch_doc_columns(
        bq=bq,
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
        document_ids=document_ids,
    )

    # Representative resource: mirror 4.4's sample_selector but with
    # the candidate-query's resource list already projected.
    rep = _pick_representative(resources, columns_by_doc=columns_by_doc)
    rep_doc_id = str(rep["document_id"])
    plog.info(
        "representative_picked",
        resource_count=len(resources),
        dictionary_candidates=sum(
            1 for r in resources
            if looks_like_dictionary(
                columns_by_doc.get(str(r["document_id"]), [])
            )
        ),
        chosen_document_id=rep_doc_id,
        chosen_row_count=rep.get("row_count"),
    )
    package_title = _first_non_null(resources, "title")
    subjects: set[str] = set()
    for r in resources:
        rs = r.get("subjects") or []
        if isinstance(rs, list | tuple):
            for s in rs:
                if isinstance(s, str):
                    subjects.add(s)
    sorted_subjects = sorted(subjects)

    raw_columns = package_grouper.column_union(columns_by_doc)

    kept: list[str] = []
    dropped: list[str] = []
    drop_events_emitted = 0
    for name in raw_columns:
        if allowlist.match(name):
            kept.append(name)
        else:
            dropped.append(name)
            if drop_events_emitted < _DROP_EVENT_CAP_PER_PACKAGE:
                plog.info(
                    "column_name_dropped_by_allowlist",
                    column_name=name,
                    reason="regex_no_match",
                )
                drop_events_emitted += 1

    if not kept:
        plog.warning(
            "package_columns_inputs_empty",
            reason="json_keys_empty"
            if not raw_columns
            else "all_dropped_by_allowlist",
            dropped_count=len(dropped),
        )
        return _PackageOutcome(
            kind="empty",
            package_id=package_id,
            inputs=None,
            dropped_count=len(dropped),
            summary_present=package_summary is not None,
        )

    sample_values_by_col = _fetch_sample_values(
        bq=bq,
        project_id=project_id,
        dataset_raw=settings.bq_dataset_raw,
        rows_table=settings.bq_rows_table,
        document_id=rep_doc_id,
        column_names=kept,
        per_column_cap=settings.column_sample_values_cap,
    )

    sample_values = {
        name: tuple(sample_values_by_col.get(name, ())) for name in kept
    }

    inputs = ColumnInputs(
        package_id=package_id,
        package_title=package_title,
        package_subjects=tuple(sorted_subjects),
        package_summary=package_summary,
        representative_document_id=rep_doc_id,
        column_names=tuple(kept),
        sample_values=sample_values,
        dropped_columns=tuple(dropped),
        overflow_column_count=0,
        extracted_at=datetime.now(UTC),
    )

    plog.info(
        "package_columns_inputs_built",
        column_count=len(kept),
        representative_document_id=rep_doc_id,
        dropped_column_count=len(dropped),
        package_summary_present=package_summary is not None,
    )
    if package_summary is None:
        plog.info("package_summary_unavailable")

    return _PackageOutcome(
        kind="extracted",
        package_id=package_id,
        inputs=inputs,
        dropped_count=len(dropped),
        summary_present=package_summary is not None,
    )


def _pick_representative(
    resources: list[dict[str, object]],
    *,
    columns_by_doc: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    """Median-row-count resource, lexicographic document_id tiebreak.

    Dictionary-shaped resources (per `looks_like_dictionary` over the
    per-doc headers) are demoted out of the pool; they stay eligible
    only when every resource is dictionary-shaped. Resources with
    row_count == None sort last. Matches the semantics of 4.4's
    `sample_selector.pick_representative` while operating on the
    candidate-query's projected dicts.
    """
    pool = resources
    if columns_by_doc:
        non_dict = [
            r for r in resources
            if not looks_like_dictionary(
                columns_by_doc.get(str(r.get("document_id", "")), [])
            )
        ]
        pool = non_dict or resources

    def _key(r: dict[str, object]) -> tuple[int, int, str]:
        rc = r.get("row_count")
        doc = str(r.get("document_id", ""))
        if rc is None:
            return (1, 0, doc)
        if isinstance(rc, int):
            return (0, rc, doc)
        return (0, int(str(rc)), doc)

    pool = sorted(pool, key=_key)
    return pool[len(pool) // 2]


def _first_non_null(
    resources: list[dict[str, object]], field: str
) -> str | None:
    for r in resources:
        v = r.get(field)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v
    return None


# ── SQL builders (worker-thread-safe; pure strings) ──


def _build_candidate_sql(
    *,
    project_id: str,
    dataset_raw: str,
    documents_table: str,
    with_limit: bool,
) -> str:
    """One row per `package_id`; same shape as 4.4's candidate query."""
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


def _build_candidate_params(
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


def _fetch_dataset_summaries(
    *,
    bq: BqClient,
    project_id: str,
    dataset_semantic: str,
    datasets_table: str,
    package_ids: list[str],
    log: structlog.BoundLogger,
) -> dict[str, str]:
    """Batched `semantic.datasets.summary` lookup (§5.5). One query
    for the full candidate set instead of ~3,693 per-package round-
    trips.

    A missing or empty `semantic.datasets` table is logged as a
    warning, not a failure — the prompt's fallback path handles it.
    """
    if not package_ids:
        return {}
    fq = f"`{project_id}.{dataset_semantic}.{datasets_table}`"
    sql = f"""
SELECT package_id, summary
FROM {fq}
WHERE package_id IN UNNEST(@package_ids);
""".strip()
    params = [
        bigquery.ArrayQueryParameter("package_ids", "STRING", package_ids)
    ]
    out: dict[str, str] = {}
    try:
        for row in bq.query_rows(sql, params=params):
            pid = row.get("package_id")
            summary = row.get("summary")
            if isinstance(pid, str) and isinstance(summary, str):
                out[pid] = summary
    except gax.NotFound as exc:
        log.warning(
            "semantic_datasets_table_missing",
            project=project_id,
            dataset=dataset_semantic,
            table=datasets_table,
            error=str(exc),
        )
        return {}
    return out


def _fetch_sample_values(
    *,
    bq: BqClient,
    project_id: str,
    dataset_raw: str,
    rows_table: str,
    document_id: str,
    column_names: list[str],
    per_column_cap: int,
) -> dict[str, list[str]]:
    """One query per package: returns ≤`per_column_cap` distinct
    sample values for every column name in `column_names`, drawn from
    the representative resource.

    BQ's `JSON_VALUE(json, path)` requires `path` to be a compile-time
    string literal and does not accept bracket notation of any form
    (`$['name']`, `$["name"]` both fail). Field access on the `JSON`
    type via subscript (`json_expr[name]`) *does* accept runtime
    values and works for any key, including ones with hyphens, dots,
    slashes, and spaces. So we `PARSE_JSON` in a CTE, cross-join with
    `UNNEST(@names)`, and pull each field with `LAX_STRING(j[n])`.
    """
    if not column_names:
        return {}
    fq = f"`{project_id}.{dataset_raw}.{rows_table}`"
    sql = f"""
WITH parsed AS (
  SELECT PARSE_JSON(STRING(row)) AS j
  FROM {fq}
  WHERE document_id = @document_id
),
cells AS (
  SELECT
    n AS col_name,
    LAX_STRING(p.j[n]) AS v
  FROM parsed p, UNNEST(@names) AS n
),
distinct_cells AS (
  SELECT DISTINCT col_name, v
  FROM cells
  WHERE v IS NOT NULL AND v != ''
),
ranked AS (
  SELECT col_name, v,
         ROW_NUMBER() OVER (PARTITION BY col_name ORDER BY v) AS rn
  FROM distinct_cells
)
SELECT col_name, v
FROM ranked
WHERE rn <= @cap
ORDER BY col_name, rn;
""".strip()
    params: list[
        bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
    ] = [
        bigquery.ArrayQueryParameter("names", "STRING", column_names),
        bigquery.ScalarQueryParameter("document_id", "STRING", document_id),
        bigquery.ScalarQueryParameter("cap", "INT64", per_column_cap),
    ]
    out: dict[str, list[str]] = {name: [] for name in column_names}
    for row in bq.query_rows(sql, params=params):
        name = str(row["col_name"])
        v = row.get("v")
        if name in out and v is not None:
            out[name].append(str(v))
    return out


def _assert_invariant(
    summary: ColumnsExtractRunSummary, log: structlog.BoundLogger
) -> None:
    total = (
        summary.packages_extracted
        + summary.packages_skipped_already_extracted
        + summary.packages_empty
    )
    if total != summary.candidate_count:
        log.error(
            "run_invariant_violated",
            subcommand="columns-extract",
            summary=summary.__dict__,
        )
        raise RuntimeError(
            f"columns-extract candidates accounted-for mismatch: {summary}"
        )
