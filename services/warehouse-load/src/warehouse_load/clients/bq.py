"""BigQuery client surface: `BqClient` Protocol + `RealBqClient` impl.

Surface: load JSON, run DML, count rows, create/delete a staging table,
query rows, append a local JSONL file. Tests implement `BqClient`
directly instead of monkeypatching.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from google.api_core import exceptions as gax
from google.cloud import bigquery
from tenacity import Retrying

from warehouse_load.providers.retry import bq_retry_policy


@runtime_checkable
class BqClient(Protocol):
    def load_json(
        self,
        *,
        rows: list[dict[str, Any]],
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        """Truncate-and-load `rows` into `destination`. Returns rows loaded."""

    def execute(self, sql: str) -> None:
        """Run DML to completion. No result; use `count_rows` for SELECTs."""

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = (),
    ) -> Iterator[dict[str, Any]]:
        """Run a SELECT and yield rows as dicts. Parameter-bound to keep
        user-supplied filters out of the SQL string."""

    def append_jsonl_file(
        self,
        *,
        jsonl_path: Path,
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        """Append a newline-delimited JSON file to `destination`.

        Returns rows loaded. Uses WRITE_APPEND so multiple per-doc
        writes accumulate in a shared staging table without conflict
        (BQ load jobs are atomic per file)."""

    def count_rows(self, table_id: str) -> int:
        """Returns 0 if the table is empty or missing."""

    def create_staging_table(
        self,
        *,
        table_id: str,
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> None:
        """Create a table that BQ auto-deletes after `expires_in`."""

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None: ...


class RealBqClient:

    def __init__(self, client: bigquery.Client, *, retry: Retrying | None = None) -> None:
        self._client = client
        # tenacity Retrying.__iter__ calls begin() to reset state, so a
        # single instance is safe to reuse across calls.
        self._retry = retry if retry is not None else bq_retry_policy()

    @classmethod
    def for_project(cls, project_id: str) -> RealBqClient:
        return cls(bigquery.Client(project=project_id))

    def load_json(
        self,
        *,
        rows: list[dict[str, Any]],
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        for attempt in self._retry:
            with attempt:
                job = self._client.load_table_from_json(
                    json_rows=rows,
                    destination=destination,
                    job_config=job_config,
                )
                job.result()  # block + raise on failure
        return len(rows)

    def execute(self, sql: str) -> None:
        for attempt in self._retry:
            with attempt:
                self._client.query(sql).result()

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = (),
    ) -> Iterator[dict[str, Any]]:
        job_config = bigquery.QueryJobConfig(query_parameters=list(params))
        # Materialise outside the retry attempt: streaming the iterator
        # mid-retry would replay a partial result on backoff.
        for attempt in self._retry:
            with attempt:
                job = self._client.query(sql, job_config=job_config)
                rows = [dict(r) for r in job.result()]
                break
        else:  # pragma: no cover — tenacity reraises on exhaustion
            return iter(())
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
        for attempt in self._retry:
            with attempt:
                with jsonl_path.open("rb") as src:
                    job = self._client.load_table_from_file(
                        src,
                        destination=destination,
                        job_config=job_config,
                    )
                job.result()
                # `output_rows` is the count BQ wrote; fall back to
                # 0 when missing (load job for empty file).
                return int(job.output_rows or 0)
        return 0  # pragma: no cover — tenacity reraises on exhaustion

    def count_rows(self, table_id: str) -> int:
        try:
            for attempt in self._retry:
                with attempt:
                    job = self._client.query(f"SELECT COUNT(*) AS n FROM `{table_id}`")
                    for row in job.result():
                        return int(row["n"])
        except gax.NotFound:
            return 0
        return 0

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
        # Create explicitly so `expires` is set on first creation; the
        # load job's WRITE_TRUNCATE would create the table without it.
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
