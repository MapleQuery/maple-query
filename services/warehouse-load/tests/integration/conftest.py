"""Integration-test fakes for BQ and GCS.

`FakeBqClient` implements the `BqClient` Protocol; tests pre-populate
its `target_rows` to simulate prior loads, then inspect
`load_calls` / `query_calls` after the run.

`FakeGcsClient` implements `GcsClient` for the documents-loader
intersection tests: pre-populate `existing` with the gs:// URIs the
bucket should be claimed to contain.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from google.cloud import bigquery


@dataclass
class FakeBqClient:
    """Hand-rolled BQ stand-in.

    Captures every load and execute for assertion. The `target_rows`
    map simulates the contents of `raw.documents` between runs —
    `count_rows` reads from it, and a MERGE-style execute upserts
    from the most recent load payload into it.
    """

    target_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    load_calls: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)
    query_calls: list[str] = field(default_factory=list)
    create_calls: list[str] = field(default_factory=list)
    delete_calls: list[str] = field(default_factory=list)

    def load_json(
        self,
        *,
        rows: list[dict[str, Any]],
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        self.load_calls.append((destination, list(rows)))
        return len(rows)

    def execute(self, sql: str) -> None:
        self.query_calls.append(sql)
        # Simulate a MERGE by upserting from the most recent load_call
        # into target_rows, keyed by document_id.
        if "MERGE INTO" in sql and self.load_calls:
            for row in self.load_calls[-1][1]:
                self.target_rows[row["document_id"]] = row

    def count_rows(self, table_id: str) -> int:
        return len(self.target_rows)

    def create_staging_table(
        self,
        *,
        table_id: str,
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> None:
        self.create_calls.append(table_id)

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None:
        self.delete_calls.append(table_id)


@dataclass
class FakeGcsClient:
    """Hand-rolled GCS stand-in.

    `existing` is the set of full `gs://bucket/object` URIs the bucket
    is claimed to contain. `list_existing_calls` records the prefixes
    requested, so tests can assert the call happens (or doesn't).
    `list_jsonl_pages` lets tests feed the runlog reader if needed.
    """

    existing: set[str] = field(default_factory=set)
    list_existing_calls: list[str] = field(default_factory=list)
    list_jsonl_pages: list[tuple[str, list[str]]] = field(default_factory=list)

    def list_jsonl(self, gcs_prefix: str) -> Iterator[tuple[str, Iterator[str]]]:
        for source, lines in self.list_jsonl_pages:
            yield source, iter(lines)

    def list_existing(self, gcs_prefix: str) -> set[str]:
        self.list_existing_calls.append(gcs_prefix)
        return set(self.existing)
