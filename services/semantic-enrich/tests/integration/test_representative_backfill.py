"""End-to-end datasets-backfill-representative against a FakeBqClient."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.representative_backfill import (
    RepresentativeBackfillRequest,
    run_backfill,
)

from .conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        # Single-thread for deterministic FakeBqClient response FIFO.
        extract_concurrency=1,
    )


def _resource(doc_id: str, row_count: int) -> dict:
    return {
        "document_id": doc_id,
        "title": "T",
        "subjects": ["s1"],
        "organization_code": "org",
        "file_format": "csv",
        "resource_last_modified": None,
        "row_count": row_count,
    }


def _register_common(bq: FakeBqClient) -> None:
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    # Rows currently in semantic.datasets (column not yet populated).
    bq.register_query(
        "FROM `proj.semantic.datasets`",
        [{"package_id": "pkg-a", "representative_document_id": None}],
    )
    # Candidate query over raw.documents: dictionary CSV wins the
    # naive median; the backfill must demote it.
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "package_id": "pkg-a",
                "resources": [
                    _resource("doc-data-big", 100),
                    _resource("doc-dict", 24),
                    _resource("doc-data-small", 10),
                ],
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [
            {"document_id": "doc-data-big", "columns": ["permit_id", "area_ha"]},
            {
                "document_id": "doc-dict",
                "columns": ["COLUMN_NAME", "DATA_TYPE", "DESCRIPTION"],
            },
            {"document_id": "doc-data-small", "columns": ["permit_id", "area_ha"]},
        ],
    )


def test_backfill_merges_picked_representative(tmp_path: Path) -> None:
    bq = FakeBqClient()
    _register_common(bq)

    summary = run_backfill(
        request=RepresentativeBackfillRequest(
            run_id="r1", dry_run=False, limit_package_ids=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.packages_in_target == 1
    assert summary.packages_picked == 1
    assert summary.packages_no_resources == 0
    # Stored value was NULL → counts as changed.
    assert summary.picks_changed == 1
    assert summary.rows_merged == 1

    # Payload carries the demoted-dictionary pick.
    payload = (
        tmp_path / "r1" / "_representative_backfill_payload.jsonl"
    )
    rows = [json.loads(line) for line in payload.read_text().splitlines()]
    assert rows == [
        {
            "package_id": "pkg-a",
            "representative_document_id": "doc-data-big",
        }
    ]

    # MERGE updates representative_document_id only — never the
    # always-newer-wins generated_at clock.
    merge_calls = [
        c for c in bq.calls
        if c["kind"] == "execute" and "MERGE INTO" in c["sql"]
    ]
    assert len(merge_calls) == 1
    assert "representative_document_id" in merge_calls[0]["sql"]
    assert "generated_at" not in merge_calls[0]["sql"]
    assert bq.deleted_tables  # staging table cleaned up


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    bq = FakeBqClient()
    _register_common(bq)

    summary = run_backfill(
        request=RepresentativeBackfillRequest(
            run_id="r1", dry_run=True, limit_package_ids=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.dry_run is True
    assert summary.packages_picked == 1
    assert summary.rows_merged == 0
    assert bq.staging_tables == {}
    assert not any(
        c["kind"] == "execute" and "MERGE INTO" in c["sql"] for c in bq.calls
    )


def test_backfill_counts_unchanged_pick(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query(
        "FROM `proj.semantic.datasets`",
        [
            {
                "package_id": "pkg-a",
                "representative_document_id": "doc-only",
            }
        ],
    )
    bq.register_query(
        "load_status = 'loaded'",
        [{"package_id": "pkg-a", "resources": [_resource("doc-only", 10)]}],
    )
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [{"document_id": "doc-only", "columns": ["year"]}],
    )

    summary = run_backfill(
        request=RepresentativeBackfillRequest(
            run_id="r1", dry_run=False, limit_package_ids=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.picks_changed == 0
    assert summary.picks_unchanged == 1
    # Unchanged picks still merge (idempotent write, simpler than a diff).
    assert summary.rows_merged == 1


def test_backfill_raises_when_target_empty(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.datasets`", [])
    with pytest.raises(RuntimeError, match="nothing to backfill"):
        run_backfill(
            request=RepresentativeBackfillRequest(
                run_id="r1", dry_run=False, limit_package_ids=None
            ),
            settings=_settings(tmp_path),
            bq=bq,
        )
