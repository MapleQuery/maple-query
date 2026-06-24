"""SQL generation for raw.documents UPDATEs owned by the rows loader."""
from __future__ import annotations

import re

import pytest

from tests.integration.conftest import FakeBqClient
from warehouse_load.core import document_status

_DOCS_TABLE = "proj.raw.documents"
_DOC_ID = "a" * 64


def test_mark_in_flight_updates_status_attempt_and_clears_error() -> None:
    bq = FakeBqClient()
    document_status.mark_in_flight(
        bq=bq, documents_table=_DOCS_TABLE, document_id=_DOC_ID,
    )
    assert len(bq.query_calls) == 1
    sql = bq.query_calls[0]
    assert "load_status = 'pending'" in sql
    assert "load_attempted_at = CURRENT_TIMESTAMP()" in sql
    assert "load_error = NULL" in sql
    assert f"'{_DOC_ID}'" in sql


def test_record_load_outcome_loaded() -> None:
    bq = FakeBqClient()
    document_status.record_load_outcome(
        bq=bq, documents_table=_DOCS_TABLE,
        document_id=_DOC_ID,
        load_status="loaded", load_error=None,
        preamble_rows=(("Title",), ("Date",)),
        header_confidence="single", row_count=42,
    )
    sql = bq.query_calls[0]
    assert "load_status = 'loaded'" in sql
    assert "header_confidence = 'single'" in sql
    assert "row_count = 42" in sql
    assert "PARSE_JSON(" in sql  # preamble JSON literal


def test_record_load_outcome_blob_missing() -> None:
    bq = FakeBqClient()
    document_status.record_load_outcome(
        bq=bq, documents_table=_DOCS_TABLE,
        document_id=_DOC_ID,
        load_status="blob_missing", load_error="gcs 404",
        preamble_rows=None, header_confidence=None, row_count=None,
    )
    sql = bq.query_calls[0]
    assert "load_status = 'blob_missing'" in sql
    assert "load_error = 'gcs 404'" in sql
    assert "row_count = NULL" in sql


def test_record_load_outcome_escapes_single_quotes() -> None:
    bq = FakeBqClient()
    document_status.record_load_outcome(
        bq=bq, documents_table=_DOCS_TABLE,
        document_id=_DOC_ID,
        load_status="parse_failed",
        load_error="o'reilly malformed; ok",
        preamble_rows=None, header_confidence=None, row_count=None,
    )
    sql = bq.query_calls[0]
    # Single quote doubled; no stray injection point.
    assert "load_error = 'o''reilly malformed; ok'" in sql


def test_mark_in_flight_rejects_bad_document_id() -> None:
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="invalid document_id"):
        document_status.mark_in_flight(
            bq=bq, documents_table=_DOCS_TABLE, document_id="abc'; DROP TABLE x; --",
        )


def test_mark_in_flight_rejects_bad_table_id() -> None:
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="invalid BQ identifier"):
        document_status.mark_in_flight(
            bq=bq,
            documents_table="proj`; DROP TABLE x; --.raw.documents",
            document_id=_DOC_ID,
        )


def test_preamble_serialised_as_json_array_of_arrays() -> None:
    bq = FakeBqClient()
    document_status.record_load_outcome(
        bq=bq, documents_table=_DOCS_TABLE,
        document_id=_DOC_ID,
        load_status="loaded", load_error=None,
        preamble_rows=(("Title: Foo",), ("Date: 2026",)),
        header_confidence="single", row_count=1,
    )
    sql = bq.query_calls[0]
    # PARSE_JSON('[["Title: Foo"], ["Date: 2026"]]') — but with single
    # quotes doubled inside the literal.
    match = re.search(r"PARSE_JSON\('(.+?)'\)", sql)
    assert match is not None
    assert '[["Title: Foo"], ["Date: 2026"]]' in match.group(1)
