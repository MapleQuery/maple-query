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
