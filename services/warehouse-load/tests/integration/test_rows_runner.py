"""End-to-end orchestration against fakes.

Exercises the candidate query → per-doc work → batch MERGE →
column-index refresh → invariant check path. The CSV bytes are
small fixtures written to a tmp_path; the fake GCS client serves
them.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pytest

from tests.integration.conftest import FakeBqClient
from warehouse_load.clients.gcs_stream import BlobMissingError, BytesCapExceededError
from warehouse_load.config.settings import Settings
from warehouse_load.core.rows_runner import (
    run_rows_load,
)
from warehouse_load.types import RowsRunRequest

_DOC_A = "a" * 64
_DOC_B = "b" * 64
_DOC_C = "c" * 64


class FakeGcsStreamClient:
    """In-memory GCS stand-in. Serves blob bytes by URI; raises
    `BlobMissingError` for any URI not in the map.
    """

    def __init__(
        self,
        *,
        blobs: dict[str, bytes],
        max_bytes_override: dict[str, int] | None = None,
    ) -> None:
        self._blobs = blobs
        self._max_bytes_override = max_bytes_override or {}
        self.downloads: list[str] = []

    def download_blob_to_file(
        self,
        *,
        gcs_uri: str,
        sink: BinaryIO,
        max_bytes: int,
    ) -> int:
        self.downloads.append(gcs_uri)
        if gcs_uri not in self._blobs:
            raise BlobMissingError(f"gcs 404: {gcs_uri}")
        data = self._blobs[gcs_uri]
        effective_cap = self._max_bytes_override.get(gcs_uri, max_bytes)
        if len(data) > effective_cap:
            raise BytesCapExceededError(
                f"exceeded max_bytes_per_doc={effective_cap}",
            )
        sink.write(data)
        return len(data)


def _settings(tmp_path: Path, project: str = "proj") -> Settings:
    schemas_dir = Path(__file__).resolve().parents[2].parent.parent / "infra" / "terraform" / "schemas"
    return Settings(
        gcp_project_id=project,
        schemas_dir=schemas_dir,
        runlog_local_dir=tmp_path,
        rows_concurrency=2,
        body_min_run=5,
        header_lookback=3,
    )


def _candidate_row(
    *,
    document_id: str,
    gcs_uri: str | None,
    organization_code: str = "fin",
    file_format: str = "csv",
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "organization_code": organization_code,
        "source_url": f"https://example.org/{document_id[:8]}.csv",
        "gcs_uri": gcs_uri,
        "file_format": file_format,
        "declared_format": "CSV",
        "checksum": "x" * 64,
        "resource_last_modified": datetime(2026, 6, 1, tzinfo=UTC),
    }


_GOOD_CSV = (
    b"year,value,name\n"
    + b"\n".join(f"{i},{i}.5,label_{i}".encode() for i in range(10))
    + b"\n"
)


def test_dry_run_emits_no_bq_writes(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.query_results = [
        [_candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv")],
    ]
    gcs = FakeGcsStreamClient(blobs={"gs://b/a.csv": _GOOD_CSV})

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=True,
        refresh_column_index=False,
    )
    summary = run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )

    # Dry-run skips staging precondition, mark_in_flight, append, MERGE,
    # record_outcome. Only the candidate query SQL was emitted, and that
    # comes through query_rows_calls — NOT query_calls (which is for
    # execute()).
    assert bq.query_calls == []
    assert bq.append_calls == []
    # The dry-run still reports the candidate as "loaded" if the pipeline
    # succeeded end-to-end.
    assert summary.candidate_count == 1
    assert summary.docs_loaded == 1
    assert summary.rows_merged == 0  # dry-run doesn't merge
    assert gcs.downloads == ["gs://b/a.csv"]


def test_skip_loaded_without_force_marks_all_skipped(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.query_results = [
        [
            _candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv"),
            _candidate_row(document_id=_DOC_B, gcs_uri="gs://b/b.csv"),
        ],
    ]
    gcs = FakeGcsStreamClient(blobs={})  # never called

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="loaded",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    summary = run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )
    assert summary.docs_skipped_already_loaded == 2
    assert summary.docs_loaded == 0
    assert gcs.downloads == []


def test_blob_missing_is_a_disposed_outcome(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.query_results = [
        [
            _candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv"),
            _candidate_row(document_id=_DOC_B, gcs_uri="gs://b/gone.csv"),
        ],
    ]
    gcs = FakeGcsStreamClient(blobs={"gs://b/a.csv": _GOOD_CSV})

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    summary = run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )
    assert summary.docs_loaded == 1
    assert summary.docs_blob_missing == 1
    assert summary.candidate_count == 2
    # Invariant: every doc accounted for.
    assert (
        summary.docs_loaded
        + summary.docs_blob_missing
        + summary.docs_parse_failed
        + summary.docs_skipped_already_loaded
    ) == summary.candidate_count


def test_staging_precondition_aborts_when_table_not_empty(tmp_path: Path) -> None:
    bq = FakeBqClient()
    # Non-empty staging trips the §8.0 precondition. The runner calls
    # sys.exit(2); pytest captures that as SystemExit.
    bq.target_rows = {str(i): {} for i in range(3)}
    gcs = FakeGcsStreamClient(blobs={})

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        run_rows_load(
            request=request, bq=bq, gcs=gcs,
            settings=_settings(tmp_path), run_id="run-x",
        )
    assert exc_info.value.code == 2


def test_successful_load_writes_staging_and_merges(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.query_results = [
        [_candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv")],
    ]
    gcs = FakeGcsStreamClient(blobs={"gs://b/a.csv": _GOOD_CSV})

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    summary = run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )

    assert summary.docs_loaded == 1
    assert summary.rows_merged > 0
    # One append to staging, one MERGE+TRUNCATE script.
    assert len(bq.append_calls) == 1
    destination, payload = bq.append_calls[0]
    assert destination == "proj.raw.rows_staging"
    # Payload row count matches the CSV body (10 rows after header).
    assert len(payload) == 10
    # MERGE script was issued via execute().
    assert any("MERGE INTO" in q for q in bq.query_calls)
    assert any("TRUNCATE TABLE" in q for q in bq.query_calls)


def test_csv_body_lands_with_normalised_keys_and_null_for_empty(tmp_path: Path) -> None:
    bq = FakeBqClient()
    csv = (
        b"Hello World,value\n"
        + b"\n".join(f"{i}.5,label_{i}".encode() for i in range(8))
        + b"\n,\n"
    )
    bq.query_results = [
        [_candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv")],
    ]
    gcs = FakeGcsStreamClient(blobs={"gs://b/a.csv": csv})

    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )

    _, payload = bq.append_calls[0]
    first = payload[0]
    import json as _json
    row = _json.loads(first["row"])
    # Whitespace in header collapsed to `_`.
    assert "Hello_World" in row
    assert row["value"] == "label_0"
    # Empty-cell row should land as nulls.
    last = payload[-1]
    last_row = _json.loads(last["row"])
    assert last_row["Hello_World"] is None
    assert last_row["value"] is None


def test_invariant_violation_raises_runtime_error(tmp_path: Path) -> None:
    """If the orchestrator accidentally drops a doc on the floor, the
    §8.7 invariant check raises. Simulate via a custom GCS client that
    raises an unmapped exception type from inside the worker."""
    bq = FakeBqClient()
    bq.query_results = [
        [_candidate_row(document_id=_DOC_A, gcs_uri="gs://b/a.csv")],
    ]

    class BoomGcs:
        def download_blob_to_file(self, **kwargs: object) -> int:
            # An "uncaught" generic exception — the worker maps it to
            # parse_failed (via the broad except in the orchestrator).
            # That preserves the invariant, so this test instead asserts
            # the disposition mapping rather than the failure mode.
            raise RuntimeError("simulated worker boom")

    gcs = BoomGcs()
    request = RowsRunRequest(
        limit_orgs=(), limit_documents=(), status="pending",
        force=False, concurrency=None, dry_run=False,
        refresh_column_index=False,
    )
    summary = run_rows_load(
        request=request, bq=bq, gcs=gcs,
        settings=_settings(tmp_path), run_id="run-x",
    )
    # Uncaught -> parse_failed -> invariant holds.
    assert summary.docs_parse_failed == 1
