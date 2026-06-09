"""§12.3 — dry-run against the real services/ingest/runlog/ directory.

Asserts the run completes, the summary's stage counts are
self-consistent, and no parse errors appear on real data. If parse
errors start showing up here, the runlog row shape has drifted from
types.RawRunlogRow.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_load.core.runner import RunRequest, run_documents_load


def test_dry_run_against_real_runlogs(ingest_runlog_dir: Path, schemas_dir: Path) -> None:
    if not ingest_runlog_dir.is_dir() or not list(ingest_runlog_dir.glob("*.jsonl")):
        pytest.skip(f"no runlog JSONLs at {ingest_runlog_dir}")

    request = RunRequest(
        local_dir=ingest_runlog_dir,
        gcs_prefix=None,
        since=None,
        dry_run=True,
        limit_orgs=(),
    )

    summary = run_documents_load(
        request=request,
        bq=None,
        gcs=None,
        project_id="dry-run-project",
        dataset="raw",
        table="documents",
        schemas_dir=schemas_dir,
        run_id="00000000-0000-0000-0000-000000000000",
    )

    # Stage counts self-check (§13.4):
    # kept = seen - parse_errors - filtered_not_csv - filtered_not_success - deduped
    expected_kept = (
        summary.runlog_rows_seen
        - summary.rows_filtered_not_csv
        - summary.rows_filtered_not_success
        - summary.rows_deduped
    )
    assert summary.rows_kept == expected_kept, (
        f"stage counts inconsistent: kept={summary.rows_kept} "
        f"expected={expected_kept} from summary={summary}"
    )

    # Dry-run never writes to BQ.
    assert summary.documents_inserted == 0
    assert summary.documents_updated == 0
    assert summary.documents_unchanged == 0

    # Real runlogs should parse without errors — if this starts
    # failing, the runlog shape has drifted from RawRunlogRow.
    assert summary.runlog_parse_errors == 0, (
        f"runlog parse errors on real data: {summary.runlog_parse_errors}. "
        "Inspect tests/unit/test_runlog_reader.py output and update the model."
    )

    # Sanity: at least one row survived the filter on a non-empty
    # runlog directory.
    assert summary.rows_kept > 0, (
        "expected at least one kept row from real runlogs; got 0. "
        "Either the filter is too aggressive or the runlogs are empty."
    )
