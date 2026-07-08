"""BigQuery client surface: `BqClient` Protocol + `RealBqClient` impl.

Mirrors warehouse-load/clients/bq.py — same retry policy, same
load-job patterns — but trimmed to the surface 4.4 actually uses:
ad-hoc query, parameter-bound query, append-from-file, staging-table
lifecycle. Tests implement `BqClient` directly instead of monkeypatching.

The 4.6 harness adds two surfaces the enrichment pipeline does not
need: `dry_run_bytes` (cost cap) and `run_bounded_query` (bounded
execution with per-job stats). Both bypass the retry policy — the
harness's contract is that every guard-passing SQL either succeeds or
grades as a terminal state; a silent retry would mask the report signal.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from google.api_core import exceptions as gax
from google.cloud import bigquery
from tenacity import Retrying

from semantic_enrich.providers.retry import bq_retry_policy


@dataclass(frozen=True)
class BoundedQueryResult:
    """Return shape of `BqClient.run_bounded_query`.

    Bytes-billed and slot-ms are captured for the aggregate report;
    `timed_out` distinguishes an execution-timeout grade from an
    execution-error grade. `error` carries the BQ exception message
    verbatim when execution fails; `rows` is empty in that case.
    """

    rows: list[dict[str, Any]]
    total_bytes_billed: int
    slot_ms: int
    elapsed_ms: int
    timed_out: bool
    error: str | None


@runtime_checkable
class BqClient(Protocol):
    def execute(self, sql: str) -> None:
        """Run DML to completion."""

    def execute_with_params(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> None:
        """Run parameter-bound DML to completion."""

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> Iterator[dict[str, Any]]:
        """Run a SELECT and yield rows as dicts."""

    def append_jsonl_file(
        self,
        *,
        jsonl_path: Path,
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        """Append a newline-delimited JSON file to `destination`."""

    def create_staging_table(
        self,
        *,
        table_id: str,
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> None:
        """Create a table that BQ auto-deletes after `expires_in`."""

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None: ...

    def dry_run_bytes(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
        timeout_ms: int,
    ) -> int:
        """Dry-run the SQL; return total_bytes_processed.

        Raises on parse error, auth failure, or dataset-not-found. NO
        retry — the harness treats every failure as terminal for the
        question so the operator sees the raw signal in the report.
        """

    def run_bounded_query(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
        timeout_ms: int,
        max_bytes_billed: int,
        row_limit: int,
    ) -> BoundedQueryResult:
        """Execute with the harness's guardrails. NO retry.

        `row_limit` is a client-side truncation ceiling; the guard has
        already enforced a SQL-level LIMIT, so this is defence in depth.
        Timeouts return `timed_out=True` with an empty `rows` list;
        other BQ errors return `error` populated with the raw message.
        """

    def table_num_rows(self, table_ref: str) -> int:
        """Row count from table metadata (`get_table`). Free — no query
        job, no bytes scanned. `table_ref` is `project.dataset.table`."""


class RealBqClient:
    """Concrete BqClient backed by `google.cloud.bigquery`.

    Constructed once at process start (`for_project`) and passed
    through the core/ orchestrators as a parameter so the per-package
    unit-of-work stays testable behind the `BqClient` Protocol.
    """

    def __init__(
        self, client: bigquery.Client, *, retry: Retrying | None = None
    ) -> None:
        self._client = client
        # tenacity Retrying.__iter__ calls begin() to reset state, so a
        # single instance is safe to reuse across calls.
        self._retry = retry if retry is not None else bq_retry_policy()

    @classmethod
    def for_project(cls, project_id: str) -> RealBqClient:
        return cls(bigquery.Client(project=project_id))

    def execute(self, sql: str) -> None:
        for attempt in self._retry:
            with attempt:
                self._client.query(sql).result()

    def execute_with_params(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> None:
        job_config = bigquery.QueryJobConfig(query_parameters=list(params))
        for attempt in self._retry:
            with attempt:
                self._client.query(sql, job_config=job_config).result()

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> Iterator[dict[str, Any]]:
        job_config = bigquery.QueryJobConfig(query_parameters=list(params))
        # Materialise outside the retry attempt: streaming the iterator
        # mid-retry would replay a partial result on backoff.
        rows: list[dict[str, Any]] = []
        for attempt in self._retry:
            with attempt:
                job = self._client.query(sql, job_config=job_config)
                rows = [dict(r) for r in job.result()]
        return iter(rows)

    def append_jsonl_file(
        self,
        *,
        jsonl_path: Path,
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        output_rows = 0
        for attempt in self._retry:
            with attempt:
                with jsonl_path.open("rb") as src:
                    job = self._client.load_table_from_file(
                        src,
                        destination=destination,
                        job_config=job_config,
                    )
                job.result()
                output_rows = int(job.output_rows or 0)
        return output_rows

    def create_staging_table(
        self,
        *,
        table_id: str,
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> None:
        from datetime import UTC, datetime

        table = bigquery.Table(table_id, schema=schema)
        table.expires = datetime.now(UTC) + expires_in
        # Create explicitly so `expires` is set on first creation; a
        # load job's WRITE_APPEND would create the table without it.
        for attempt in self._retry:
            with attempt:
                self._client.create_table(table, exists_ok=True)

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None:
        try:
            for attempt in self._retry:
                with attempt:
                    self._client.delete_table(table_id, not_found_ok=not_found_ok)
        except gax.NotFound:
            if not not_found_ok:
                raise

    def dry_run_bytes(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
        timeout_ms: int,
    ) -> int:
        # No retry — the harness surfaces every failure verbatim in the
        # per-question grade rather than silently retrying.
        job_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            query_parameters=list(params),
        )
        job = self._client.query(
            sql,
            job_config=job_config,
            timeout=timeout_ms / 1000.0,
        )
        return int(job.total_bytes_processed or 0)

    def run_bounded_query(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
        timeout_ms: int,
        max_bytes_billed: int,
        row_limit: int,
    ) -> BoundedQueryResult:
        import time as _time

        job_config = bigquery.QueryJobConfig(
            query_parameters=list(params),
            maximum_bytes_billed=max_bytes_billed,
        )
        started = _time.monotonic()
        timeout_s = timeout_ms / 1000.0
        try:
            job = self._client.query(sql, job_config=job_config)
            iterator = job.result(timeout=timeout_s)
            rows: list[dict[str, Any]] = []
            for row in iterator:
                rows.append(dict(row))
                if len(rows) >= row_limit:
                    break
            elapsed_ms = int((_time.monotonic() - started) * 1000)
            return BoundedQueryResult(
                rows=rows,
                total_bytes_billed=int(job.total_bytes_billed or 0),
                slot_ms=int(job.slot_millis or 0),
                elapsed_ms=elapsed_ms,
                timed_out=False,
                error=None,
            )
        except TimeoutError as exc:
            elapsed_ms = int((_time.monotonic() - started) * 1000)
            return BoundedQueryResult(
                rows=[],
                total_bytes_billed=0,
                slot_ms=0,
                elapsed_ms=elapsed_ms,
                timed_out=True,
                error=str(exc),
            )
        except gax.GoogleAPICallError as exc:
            elapsed_ms = int((_time.monotonic() - started) * 1000)
            return BoundedQueryResult(
                rows=[],
                total_bytes_billed=0,
                slot_ms=0,
                elapsed_ms=elapsed_ms,
                timed_out=False,
                error=str(exc),
            )

    def table_num_rows(self, table_ref: str) -> int:
        num_rows = 0
        for attempt in self._retry:
            with attempt:
                num_rows = int(self._client.get_table(table_ref).num_rows or 0)
        return num_rows
