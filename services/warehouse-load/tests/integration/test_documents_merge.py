"""Integration tests for the MERGE pipeline against a FakeBqClient."""
from __future__ import annotations

import re
from pathlib import Path

from google.cloud import bigquery

from tests.conftest import make_row
from tests.integration.conftest import FakeBqClient
from warehouse_load.core.documents_merge import (
    DOCUMENTS_OWNED_BY_CONTENT_LOADER,
    merge_documents,
)
from warehouse_load.core.schema_loader import load_schema


def _schema(schemas_dir: Path) -> list[bigquery.SchemaField]:
    return load_schema(schemas_dir / "raw_documents.json")


def test_merge_emits_correct_staging_payload_shape(schemas_dir: Path) -> None:
    bq = FakeBqClient()
    row = make_row(document_id="a" * 64, source_url="https://example.org/a.csv")

    merge_documents(
        bq=bq,
        rows=[row],
        project_id="proj",
        dataset="raw",
        table="documents",
        schema=_schema(schemas_dir),
        run_id_short="abcdef12",
    )

    assert len(bq.load_calls) == 1
    destination, payload = bq.load_calls[0]
    assert destination == "proj.raw._documents_staging_abcdef12"
    assert len(payload) == 1

    sent = payload[0]
    schema_field_names = {f.name for f in _schema(schemas_dir)}
    payload_keys = set(sent.keys())
    assert payload_keys == schema_field_names, (
        f"staging payload columns must match raw_documents.json. "
        f"missing={schema_field_names - payload_keys}, "
        f"extra={payload_keys - schema_field_names}"
    )
    # Content-loader columns are present, set to their initial values.
    assert sent["load_status"] == "pending"
    assert sent["row_count"] is None
    assert sent["load_attempted_at"] is None


def test_merge_update_clause_omits_content_loader_columns(schemas_dir: Path) -> None:
    """The single most important property of the MERGE statement."""
    bq = FakeBqClient()
    row = make_row()

    merge_documents(
        bq=bq,
        rows=[row],
        project_id="proj",
        dataset="raw",
        table="documents",
        schema=_schema(schemas_dir),
        run_id_short="abcdef12",
    )

    # Find the MERGE query and pull its UPDATE clause.
    merge_sqls = [q for q in bq.query_calls if "MERGE INTO" in q]
    assert len(merge_sqls) == 1
    sql = merge_sqls[0]

    update_clause_match = re.search(
        r"THEN UPDATE SET\s+(.+?)WHEN NOT MATCHED",
        sql,
        re.DOTALL,
    )
    assert update_clause_match, f"could not locate UPDATE clause in:\n{sql}"
    update_clause = update_clause_match.group(1)

    for forbidden in DOCUMENTS_OWNED_BY_CONTENT_LOADER:
        assert forbidden not in update_clause, (
            f"Content-loader column {forbidden!r} must NOT appear in the MERGE UPDATE clause. "
            f"Full clause:\n{update_clause}"
        )


def test_merge_re_run_against_same_payload_inserts_zero(schemas_dir: Path) -> None:
    """Idempotence: re-running against the same payload inserts zero rows."""
    bq = FakeBqClient()
    row = make_row(document_id="a" * 64, source_url="https://example.org/a.csv")

    first = merge_documents(
        bq=bq,
        rows=[row],
        project_id="proj",
        dataset="raw",
        table="documents",
        schema=_schema(schemas_dir),
        run_id_short="run-aaaa",
    )
    assert first.rows_inserted == 1

    second = merge_documents(
        bq=bq,
        rows=[row],
        project_id="proj",
        dataset="raw",
        table="documents",
        schema=_schema(schemas_dir),
        run_id_short="run-bbbb",
    )
    assert second.rows_inserted == 0
