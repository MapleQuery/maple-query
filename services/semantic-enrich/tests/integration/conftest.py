"""Integration-test fixtures.

`FakeBqClient` implements the `BqClient` Protocol and records every
call so tests can assert on SQL fingerprints, parameter bindings, and
call counts. `fake_generate_json` / `fake_embed_batch` are pure-Python
stand-ins for the GPU calls — injected into the runners via parameter
rather than monkeypatched at the torch level, so the tests stay
GPU-free without coupling to outlines/sentence-transformers internals.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.cloud import bigquery

from semantic_enrich.clients.bq import BoundedQueryResult


class FakeBqClient:
    """In-memory BqClient. Tests register canned responses keyed by a
    substring of the SQL; each call records its full SQL + params."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, list[list[dict[str, Any]]]] = {}
        self._dml_responses: dict[str, int] = {}
        self.staging_tables: dict[str, dict[str, Any]] = {}
        self.deleted_tables: list[str] = []
        self.loaded_files: list[dict[str, Any]] = []
        # 4.6 harness surfaces. `dry_run_bytes_value` is returned by
        # every `dry_run_bytes` call unless `dry_run_bytes_exc` is set.
        # `bounded_query_result` is returned by `run_bounded_query`
        # unless the test wires a per-SQL-fragment lookup via
        # `register_bounded_query`.
        self.dry_run_bytes_value: int = 100_000_000
        self.dry_run_bytes_exc: Exception | None = None
        self.dry_run_calls: list[str] = []
        self._bounded_by_fragment: dict[str, BoundedQueryResult] = {}
        self.bounded_default: BoundedQueryResult = BoundedQueryResult(
            rows=[], total_bytes_billed=0, slot_ms=0,
            elapsed_ms=0, timed_out=False, error=None,
        )
        self.bounded_calls: list[str] = []
        # 6.2 describe_corpus surface: table_ref → num_rows metadata.
        self.table_num_rows_by_ref: dict[str, int] = {}
        self.table_num_rows_calls: list[str] = []

    # ── Test-side setup ──

    def register_query(
        self, sql_fragment: str, rows: list[dict[str, Any]]
    ) -> None:
        """Append a canned response. Consumed FIFO on each call whose SQL
        contains `sql_fragment`."""
        self._responses.setdefault(sql_fragment, []).append(rows)

    # ── BqClient Protocol ──

    def execute(self, sql: str) -> None:
        self.calls.append({"kind": "execute", "sql": sql})

    def execute_with_params(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> None:
        self.calls.append(
            {"kind": "execute_with_params", "sql": sql, "params": list(params)}
        )

    def query_rows(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
    ) -> Iterator[dict[str, Any]]:
        self.calls.append(
            {"kind": "query_rows", "sql": sql, "params": list(params)}
        )
        for fragment, queue in self._responses.items():
            if fragment in sql and queue:
                return iter(queue.pop(0))
        return iter([])

    def append_jsonl_file(
        self,
        *,
        jsonl_path: Path,
        destination: str,
        schema: list[bigquery.SchemaField],
    ) -> int:
        rows = 0
        if jsonl_path.exists():
            with jsonl_path.open() as f:
                rows = sum(1 for line in f if line.strip())
        self.loaded_files.append(
            {
                "jsonl_path": str(jsonl_path),
                "destination": destination,
                "schema_len": len(schema),
                "rows": rows,
            }
        )
        return rows

    def create_staging_table(
        self,
        *,
        table_id: str,
        schema: list[bigquery.SchemaField],
        expires_in: timedelta,
    ) -> None:
        self.staging_tables[table_id] = {
            "schema_len": len(schema),
            "expires_in_s": expires_in.total_seconds(),
        }

    def delete_table(self, table_id: str, *, not_found_ok: bool = True) -> None:
        self.deleted_tables.append(table_id)

    # ── 4.6 harness surfaces ──

    def register_bounded_query(
        self, sql_fragment: str, result: BoundedQueryResult
    ) -> None:
        self._bounded_by_fragment[sql_fragment] = result

    def dry_run_bytes(
        self,
        sql: str,
        *,
        params: Iterable[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = (),
        timeout_ms: int,
    ) -> int:
        self.dry_run_calls.append(sql)
        if self.dry_run_bytes_exc is not None:
            raise self.dry_run_bytes_exc
        return self.dry_run_bytes_value

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
        self.bounded_calls.append(sql)
        for fragment, result in self._bounded_by_fragment.items():
            if fragment in sql:
                return result
        return self.bounded_default

    def table_num_rows(self, table_ref: str) -> int:
        self.table_num_rows_calls.append(table_ref)
        return self.table_num_rows_by_ref.get(table_ref, 0)


def fake_generate_json_factory(
    *, response_by_package: dict[str, dict[str, Any]] | None = None,
    default: dict[str, Any] | None = None,
):
    """Return a `generate_json`-shaped callable that ignores the model
    + temperature and returns canned JSON per `package_id`.

    The package_id is extracted from the prompt by a simple substring
    match against `"package_id: <id>"` — same format the dataset_prompt
    template emits.
    """
    response_by_package = response_by_package or {}

    def _fn(
        prompt: str,
        schema: object,
        *,
        model: object,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        # Extract package_id from the prompt. The template puts it in
        # both the metadata block and the trailing instructions.
        marker = "package_id: "
        idx = prompt.find(marker)
        package_id = ""
        if idx >= 0:
            tail = prompt[idx + len(marker) :]
            package_id = tail.split("\n", 1)[0].strip()
        if package_id in response_by_package:
            return response_by_package[package_id]
        if default is not None:
            return {**default, "package_id": package_id or default.get("package_id", "")}
        return {
            "package_id": package_id,
            "summary": (
                "A canned fake summary of the dataset, padded to satisfy the "
                "minimum length so the pydantic validation passes."
            ),
            "grain": "row",
            "measures": ["count"],
            "dimensions": ["year"],
            "date_range_start": None,
            "date_range_end": None,
        }

    return _fn


def fake_generate_json_list_factory(
    *,
    response_for: dict[tuple[str, int], list[dict[str, Any]]] | None = None,
    default_description: str = "A canned column description, padded to satisfy "
    "the minimum-length validation that pydantic enforces.",
):
    """Return a `generate_json_list`-shaped callable.

    `response_for` maps `(package_id, chunk_index)` to a canned
    response list. When unset, the fake echoes back one entry per
    requested column with `default_description` and the first
    available sample value.

    The fake parses the prompt for a `batch <N> of <M>` marker plus
    a `package_id` line. The columns generator's prompt template emits
    those phrases verbatim so the fake can identify which chunk is
    being asked for.
    """
    response_for = response_for or {}
    import re as _re

    def _parse_chunk_metadata(prompt: str) -> tuple[str, int, list[str]]:
        # The columns prompt template puts a `batch N of M` line and a
        # `Columns:` block with `- name: <col>` entries. We parse both.
        chunk_index = 0
        m = _re.search(r"batch (\d+) of (\d+)", prompt)
        if m:
            chunk_index = int(m.group(1)) - 1
        # No package_id line in the columns prompt; the fake_generate
        # caller passes the package_id by name match-back. For tests,
        # we read the package_title placeholder (set by the test
        # fixture to encode the package_id) when present.
        pid_match = _re.search(r"- Title: ([^\n]+)", prompt)
        package_id = pid_match.group(1).strip() if pid_match else ""
        # Column names: lines starting with `- name: `.
        column_names = [
            line.strip().removeprefix("- name: ").strip()
            for line in prompt.splitlines()
            if line.lstrip().startswith("- name: ")
        ]
        return package_id, chunk_index, column_names

    def _fn(
        prompt: str,
        schema: object,
        *,
        model: object,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> list[dict[str, Any]]:
        package_id, chunk_index, column_names = _parse_chunk_metadata(prompt)
        canned = response_for.get((package_id, chunk_index))
        if canned is not None:
            return canned
        return [
            {
                "column_name": name,
                "semantic_type": "text",
                "description": default_description,
                "sample_values": [],
            }
            for name in column_names
        ]

    return _fn


def fake_embed_batch_factory(*, dim: int = 1024):
    """Return a `embed_batch`-shaped callable that produces unit-norm
    vectors of `dim` entries, deterministic per input text."""
    import hashlib
    import math

    def _fn(texts: list[str], *, model: object, batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int(
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16
            )
            # Spread the seed across the dim coords so vectors aren't
            # identical — but stay cheap to compute.
            raw = [
                ((seed >> (i % 60)) & 0xFF) / 255.0 + 0.001
                for i in range(dim)
            ]
            norm = math.sqrt(sum(x * x for x in raw))
            out.append([x / norm for x in raw])
        return out

    return _fn
