"""Runner-level integration tests for the bucket-existence intersection.

Wires `run_documents_load` against `FakeBqClient` + `FakeGcsClient` and
asserts the staging-load payload reflects the intersection — i.e.
rows with a `gcs_uri` absent from the bucket-truth set are NOT sent to
BigQuery.
"""
from __future__ import annotations

from pathlib import Path

from tests.conftest import make_row
from tests.integration.conftest import FakeBqClient, FakeGcsClient
from warehouse_load.core.runner import RunRequest, run_documents_load
from warehouse_load.types import RawRunlogRow

PROJECT = "proj"
DATASET = "raw"
TABLE = "documents"
RUN_ID = "00000000-0000-0000-0000-000000000000"
BUCKET_PREFIX = "gs://maplequery-raw/raw/"


def _runlog_jsonl(rows: list[RawRunlogRow]) -> list[str]:
    return [r.model_dump_json() for r in rows]


def _request_with_bucket(
    *,
    dry_run: bool = False,
    no_bucket_check: bool = False,
    bucket_prefix: str | None = BUCKET_PREFIX,
) -> RunRequest:
    return RunRequest(
        local_dir=None,
        gcs_prefix="gs://maplequery-raw/runlog/",
        since=None,
        dry_run=dry_run,
        limit_orgs=(),
        bucket_prefix=bucket_prefix,
        no_bucket_check=no_bucket_check,
    )


def _row_with_uri(
    *,
    document_id: str,
    source_url: str,
    gcs_uri: str | None,
) -> RawRunlogRow:
    return make_row(
        source_url=source_url,
        document_id=document_id,
    ).model_copy(update={"gcs_uri": gcs_uri})


def test_intersection_filters_zombie_before_merge(schemas_dir: Path) -> None:
    """A row whose gcs_uri is missing from the bucket never reaches BQ."""
    alive = _row_with_uri(
        document_id="a" * 64,
        source_url="https://example.org/alive.csv",
        gcs_uri=f"{BUCKET_PREFIX}ca/x/y/2026-01-01/csv/alive.csv",
    )
    zombie = _row_with_uri(
        document_id="b" * 64,
        source_url="https://example.org/zombie.csv",
        gcs_uri=f"{BUCKET_PREFIX}ca/x/y/2026-01-01/csv/zombie.csv",
    )

    bq = FakeBqClient()
    gcs = FakeGcsClient(
        existing={alive.gcs_uri or ""},
        list_jsonl_pages=[("gs://run/runlog/a.jsonl", _runlog_jsonl([alive, zombie]))],
    )

    summary = run_documents_load(
        request=_request_with_bucket(),
        bq=bq,
        gcs=gcs,
        project_id=PROJECT,
        dataset=DATASET,
        table=TABLE,
        schemas_dir=schemas_dir,
        run_id=RUN_ID,
    )

    assert gcs.list_existing_calls == [BUCKET_PREFIX]
    assert summary.rows_filtered_blob_missing == 1
    assert summary.bucket_check_skipped is False
    assert summary.rows_kept == 1

    assert len(bq.load_calls) == 1
    _, payload = bq.load_calls[0]
    sent_doc_ids = {p["document_id"] for p in payload}
    assert sent_doc_ids == {alive.document_id}
    assert zombie.document_id not in sent_doc_ids


def test_no_bucket_check_skips_intersection_and_writes_everything(schemas_dir: Path) -> None:
    """`--no-bucket-check` skips the listing entirely — even zombies merge."""
    zombie = _row_with_uri(
        document_id="b" * 64,
        source_url="https://example.org/zombie.csv",
        gcs_uri=f"{BUCKET_PREFIX}ca/x/y/2026-01-01/csv/zombie.csv",
    )

    bq = FakeBqClient()
    gcs = FakeGcsClient(
        existing=set(),  # bucket says nothing exists, but we don't ask
        list_jsonl_pages=[("gs://run/runlog/a.jsonl", _runlog_jsonl([zombie]))],
    )

    summary = run_documents_load(
        request=_request_with_bucket(no_bucket_check=True),
        bq=bq,
        gcs=gcs,
        project_id=PROJECT,
        dataset=DATASET,
        table=TABLE,
        schemas_dir=schemas_dir,
        run_id=RUN_ID,
    )

    assert gcs.list_existing_calls == [], "list_existing must not be called when bucket-check is off"
    assert summary.bucket_check_skipped is True
    assert summary.rows_filtered_blob_missing == 0
    assert summary.rows_kept == 1


def test_summary_invariant_holds_with_blob_missing(schemas_dir: Path) -> None:
    """kept = seen - not_csv - not_success - deduped - blob_missing."""
    alive = _row_with_uri(
        document_id="a" * 64,
        source_url="https://example.org/alive.csv",
        gcs_uri=f"{BUCKET_PREFIX}alive.csv",
    )
    zombie = _row_with_uri(
        document_id="b" * 64,
        source_url="https://example.org/zombie.csv",
        gcs_uri=f"{BUCKET_PREFIX}zombie.csv",
    )
    quarantined = make_row(
        source_url="https://example.org/quarantined.csv",
        document_id="c" * 64,
        ingestion_status="quarantined",
    )
    non_csv = make_row(
        source_url="https://example.org/data.xlsx",
        document_id="d" * 64,
        file_format="xlsx",
    )

    bq = FakeBqClient()
    gcs = FakeGcsClient(
        existing={alive.gcs_uri or ""},
        list_jsonl_pages=[
            (
                "gs://run/runlog/a.jsonl",
                _runlog_jsonl([alive, zombie, quarantined, non_csv]),
            ),
        ],
    )

    summary = run_documents_load(
        request=_request_with_bucket(),
        bq=bq,
        gcs=gcs,
        project_id=PROJECT,
        dataset=DATASET,
        table=TABLE,
        schemas_dir=schemas_dir,
        run_id=RUN_ID,
    )

    expected_kept = (
        summary.runlog_rows_seen
        - summary.rows_filtered_not_csv
        - summary.rows_filtered_not_success
        - summary.rows_deduped
        - summary.rows_filtered_blob_missing
    )
    assert summary.rows_kept == expected_kept


def test_real_run_without_bucket_fails_loud(schemas_dir: Path) -> None:
    """Silent skipping would pollute the warehouse. The run must fail."""
    import pytest

    alive = _row_with_uri(
        document_id="a" * 64,
        source_url="https://example.org/alive.csv",
        gcs_uri=f"{BUCKET_PREFIX}alive.csv",
    )

    bq = FakeBqClient()
    gcs = FakeGcsClient(
        list_jsonl_pages=[("gs://run/runlog/a.jsonl", _runlog_jsonl([alive]))],
    )

    request = _request_with_bucket(bucket_prefix=None)

    with pytest.raises(ValueError, match="bucket-intersection"):
        run_documents_load(
            request=request,
            bq=bq,
            gcs=gcs,
            project_id=PROJECT,
            dataset=DATASET,
            table=TABLE,
            schemas_dir=schemas_dir,
            run_id=RUN_ID,
        )


def test_intersection_skipped_in_dry_run_without_bucket(schemas_dir: Path) -> None:
    """Dry-run without a bucket: skip, don't fail. Local-dev affordance."""
    alive = _row_with_uri(
        document_id="a" * 64,
        source_url="https://example.org/alive.csv",
        gcs_uri=f"{BUCKET_PREFIX}alive.csv",
    )

    gcs = FakeGcsClient(
        list_jsonl_pages=[("gs://run/runlog/a.jsonl", _runlog_jsonl([alive]))],
    )

    summary = run_documents_load(
        request=_request_with_bucket(dry_run=True, bucket_prefix=None),
        bq=None,
        gcs=gcs,
        project_id=PROJECT,
        dataset=DATASET,
        table=TABLE,
        schemas_dir=schemas_dir,
        run_id=RUN_ID,
    )

    assert summary.bucket_check_skipped is True
    assert summary.rows_filtered_blob_missing == 0
    assert summary.rows_kept == 1


def test_intersection_runs_after_dedupe(schemas_dir: Path) -> None:
    """A row that loses dedupe is never sent to the bucket check.

    Two rows share a `source_url`; the older one points at a `gcs_uri`
    that IS in the bucket. The newer one points at a `gcs_uri` that is
    NOT in the bucket. If intersection ran before dedupe, the older
    row would survive. The correct (dedupe-first) order drops both:
    dedupe picks the newer (its gcs_uri is the zombie), intersection
    then drops it.
    """
    from datetime import UTC, datetime

    url = "https://example.org/shared.csv"
    older_alive = _row_with_uri(
        document_id="1" + "a" * 63,
        source_url=url,
        gcs_uri=f"{BUCKET_PREFIX}older.csv",
    ).model_copy(update={"ingested_at": datetime(2026, 6, 1, tzinfo=UTC)})
    newer_zombie = _row_with_uri(
        document_id="2" + "a" * 63,
        source_url=url,
        gcs_uri=f"{BUCKET_PREFIX}newer.csv",
    ).model_copy(update={"ingested_at": datetime(2026, 6, 2, tzinfo=UTC)})

    bq = FakeBqClient()
    gcs = FakeGcsClient(
        existing={older_alive.gcs_uri or ""},  # newer is the zombie
        list_jsonl_pages=[
            ("gs://run/runlog/a.jsonl", _runlog_jsonl([older_alive, newer_zombie])),
        ],
    )

    summary = run_documents_load(
        request=_request_with_bucket(),
        bq=bq,
        gcs=gcs,
        project_id=PROJECT,
        dataset=DATASET,
        table=TABLE,
        schemas_dir=schemas_dir,
        run_id=RUN_ID,
    )

    assert summary.rows_deduped == 1
    assert summary.rows_filtered_blob_missing == 1
    assert summary.rows_kept == 0
    assert bq.load_calls == [] or bq.load_calls[0][1] == []
