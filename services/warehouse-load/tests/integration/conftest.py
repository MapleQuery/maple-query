"""Integration-test fakes for BQ and GCS.

`FakeBqClient` implements the `BqClient` Protocol; tests pre-populate
its `target_rows` to simulate prior loads, then inspect
`load_calls` / `query_calls` after the run.

`FakeGcsClient` implements `GcsClient` for the documents-loader
intersection tests: pre-populate `existing` with the gs:// URIs the
bucket should be claimed to contain.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
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

    # Rows-loader surface. Defaults are inert so existing
    # documents-loader tests don't have to care.
    query_results: list[list[dict[str, Any]]] = field(default_factory=list)
    query_rows_calls: list[tuple[str, list[Any]]] = field(default_factory=list)
    append_calls: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[Any] = (),
    ) -> Iterator[dict[str, Any]]:
        self.query_rows_calls.append((sql, list(params)))
        # FIFO: each call pops the next pre-seeded result. Tests that
        # don't seed anything see empty results.
        if self.query_results:
            return iter(self.query_results.pop(0))
        return iter(())

    def append_jsonl_file(
        self,
        *,
        jsonl_path: Path,
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        with jsonl_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.append_calls.append((destination, rows))
        return len(rows)


@dataclass
class FakeGcsClient:
    """Hand-rolled GCS stand-in.

    `existing` is the set of full `gs://bucket/object` URIs the bucket
    is claimed to contain. `list_existing_calls` records the prefixes
    requested, so tests can assert the call happens (or doesn't).
    `list_jsonl_pages` lets tests feed the runlog reader if needed.
    """

    existing: set[str] = field(default_factory=set)
    # When set, `blob_exists` checks this set instead of `existing`. Lets
    # tests simulate the format-drift case where `list_existing` and a
    # per-URI HEAD disagree (the smoking gun for ingest/listing URI drift).
    head_existing: set[str] | None = None
    list_existing_calls: list[str] = field(default_factory=list)
    blob_exists_calls: list[str] = field(default_factory=list)
    list_jsonl_pages: list[tuple[str, list[str]]] = field(default_factory=list)

    def list_jsonl(self, gcs_prefix: str) -> Iterator[tuple[str, Iterator[str]]]:
        for source, lines in self.list_jsonl_pages:
            yield source, iter(lines)

    def list_existing(self, gcs_prefix: str) -> set[str]:
        self.list_existing_calls.append(gcs_prefix)
        return set(self.existing)

    def blob_exists(self, gcs_uri: str) -> bool:
        self.blob_exists_calls.append(gcs_uri)
        source = self.head_existing if self.head_existing is not None else self.existing
        return gcs_uri in source
