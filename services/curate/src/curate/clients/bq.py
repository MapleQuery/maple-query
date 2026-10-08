"""BigQuery surface for curate: load a staging table, run DML, read rows.

Tests implement `BqClient` directly instead of monkeypatching."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from google.cloud import bigquery


@runtime_checkable
class BqClient(Protocol):
    def load_staging(
        self,
        *,
        table_id: str,
        rows: list[dict[str, Any]],
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> int:
        """Create-or-replace `table_id` (auto-expiring) and load `rows`."""

    def execute(self, sql: str) -> None:
        """Run DML to completion."""

    def query_rows(self, sql: str) -> Iterator[dict[str, Any]]:
        """Run a SELECT and yield rows as dicts."""

    def count_rows(self, table_id: str) -> int:
        """0 when empty or missing."""


class RealBqClient:
    def __init__(self, *, project_id: str) -> None:
        self._client = bigquery.Client(project=project_id)

    def load_staging(
        self,
        *,
        table_id: str,
        rows: list[dict[str, Any]],
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> int:
        table = bigquery.Table(table_id, schema=schema)
        table.expires = datetime.now(UTC) + expires_in
        self._client.delete_table(table_id, not_found_ok=True)
        self._client.create_table(table)
        if not rows:
            return 0
        job = self._client.load_table_from_json(
            rows,
            table_id,
            job_config=bigquery.LoadJobConfig(
                schema=schema, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
            ),
        )
        job.result()
        return int(job.output_rows or 0)

    def execute(self, sql: str) -> None:
        self._client.query(sql).result()

    def query_rows(self, sql: str) -> Iterator[dict[str, Any]]:
        for row in self._client.query(sql).result():
            yield dict(row.items())

    def count_rows(self, table_id: str) -> int:
        try:
            return int(self._client.get_table(table_id).num_rows or 0)
        except Exception as exc:
            if exc.__class__.__name__ == "NotFound":
                return 0
            raise
