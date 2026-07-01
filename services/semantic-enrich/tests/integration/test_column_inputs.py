"""End-to-end `columns-extract` against a FakeBqClient."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_inputs import (
    ColumnsExtractRequest,
    run_extract,
)

from .conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        flush_every_n_packages=10,
        # Single-thread for deterministic FakeBqClient response FIFO.
        extract_concurrency=1,
    )


def _candidate_row(
    package_id: str,
    doc_id: str,
    *,
    title: str | None = None,
    row_count: int = 5,
    subjects: list[str] | None = None,
) -> dict:
    return {
        "package_id": package_id,
        "resources": [
            {
                "document_id": doc_id,
                "title": title or f"Title for {package_id}",
                "subjects": subjects or ["s1"],
                "organization_code": "org",
                "file_format": "csv",
                "resource_last_modified": None,
                "row_count": row_count,
            }
        ],
    }


def test_extract_three_packages_with_summary_lookup(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            _candidate_row("pkg-a", "doc-a"),
            _candidate_row("pkg-b", "doc-b"),
            _candidate_row("pkg-c", "doc-c"),
        ],
    )
    # Cross-pass summary lookup (one batched query).
    bq.register_query(
        "FROM `proj.semantic.datasets`",
        [
            {"package_id": "pkg-a", "summary": "summary-a"},
            # pkg-b missing → fallback path
            {"package_id": "pkg-c", "summary": "summary-c"},
        ],
    )
    # Per-package: column-union then sample-values. 3 packages x 2
    # responses each = 6 canned responses.
    for _ in range(3):
        bq.register_query(
            "JSON_KEYS(PARSE_JSON(STRING(row)))",
            [{"col_name": "year"}, {"col_name": "amt_cad"}],
        )
        bq.register_query(
            "ranked AS",
            [
                {"col_name": "year", "v": "2024"},
                {"col_name": "year", "v": "2025"},
                {"col_name": "amt_cad", "v": "100.00"},
            ],
        )

    request = ColumnsExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)

    assert summary.packages_extracted == 3
    assert summary.packages_summary_hit == 2
    assert summary.packages_summary_miss == 1
    assert summary.packages_empty == 0

    out = tmp_path / "r1" / "column_inputs" / "000.jsonl"
    lines = [json.loads(line) for line in out.read_text().splitlines() if line]
    pids = {row["package_id"] for row in lines}
    assert pids == {"pkg-a", "pkg-b", "pkg-c"}
    summary_by_pid = {row["package_id"]: row["package_summary"] for row in lines}
    assert summary_by_pid["pkg-a"] == "summary-a"
    assert summary_by_pid["pkg-b"] is None
    assert summary_by_pid["pkg-c"] == "summary-c"

    # Per-column sample values present, capped at 10.
    for row in lines:
        for col, samples in row["sample_values"].items():
            assert isinstance(samples, list)
            assert len(samples) <= 10
            assert col in row["column_names"]


def test_extract_resume_skips_already_extracted(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [_candidate_row("pkg-b", "doc-b")],
    )
    bq.register_query("FROM `proj.semantic.datasets`", [])
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))", [{"col_name": "x"}]
    )
    bq.register_query("ranked AS", [{"col_name": "x", "v": "1"}])

    # Pre-populate stage with pkg-a.
    inputs = tmp_path / "r1" / "column_inputs"
    inputs.mkdir(parents=True)
    (inputs / "000.jsonl").write_text(
        '{"package_id":"pkg-a","package_title":null,'
        '"package_subjects":[],"package_summary":null,'
        '"representative_document_id":"doc","column_names":["a"],'
        '"sample_values":{"a":[]},"dropped_columns":[],'
        '"overflow_column_count":0,"extracted_at":"2026-01-01T00:00:00+00:00"}\n'
    )

    request = ColumnsExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)
    assert summary.packages_extracted == 1
    assert summary.packages_skipped_already_extracted == 1


def test_extract_filters_allowlist_violations(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'", [_candidate_row("pkg-a", "doc-a")]
    )
    bq.register_query("FROM `proj.semantic.datasets`", [])
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [
            {"col_name": "year"},
            {"col_name": '"injection_attempt"'},  # allowlist drops
        ],
    )
    bq.register_query(
        "ranked AS", [{"col_name": "year", "v": "2024"}]
    )

    request = ColumnsExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)
    assert summary.columns_dropped_by_allowlist == 1

    out = tmp_path / "r1" / "column_inputs" / "000.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert rows[0]["column_names"] == ["year"]
    assert rows[0]["dropped_columns"] == ['"injection_attempt"']


def test_extract_empty_after_allowlist_skipped(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'", [_candidate_row("pkg-a", "doc-a")]
    )
    bq.register_query("FROM `proj.semantic.datasets`", [])
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [{"col_name": '"all-bad"'}],
    )

    request = ColumnsExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)
    assert summary.packages_empty == 1
    assert summary.packages_extracted == 0


def test_extract_summary_table_missing_falls_back(tmp_path: Path) -> None:
    """If semantic.datasets isn't populated yet, every package gets
    a None summary — the prompt fallback handles it."""
    from google.api_core import exceptions as gax

    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'", [_candidate_row("pkg-a", "doc-a")]
    )

    # Override FakeBqClient.query_rows for the datasets lookup only
    # to raise NotFound, then resume normal behaviour.
    original_query_rows = bq.query_rows

    def query_rows_with_not_found(sql, *, params=()):  # type: ignore[no-untyped-def]
        if "FROM `proj.semantic.datasets`" in sql:
            raise gax.NotFound("table not found")
        return original_query_rows(sql, params=params)

    bq.query_rows = query_rows_with_not_found  # type: ignore[assignment]

    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))", [{"col_name": "year"}]
    )
    bq.register_query(
        "ranked AS", [{"col_name": "year", "v": "2024"}]
    )

    request = ColumnsExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)
    assert summary.packages_extracted == 1
    assert summary.packages_summary_miss == 1
    assert summary.packages_summary_hit == 0
