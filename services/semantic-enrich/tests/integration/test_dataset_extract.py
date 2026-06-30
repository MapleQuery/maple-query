"""End-to-end datasets-extract against a FakeBqClient."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.dataset_extract import ExtractRequest, run_extract

from .conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        flush_every_n_packages=10,
        sample_rows_per_package=2,
        # Single-threaded so the FakeBqClient's FIFO response queue
        # stays deterministic; concurrency is exercised by the live
        # smoke ladder, not unit tests.
        extract_concurrency=1,
    )


def _candidate_row(package_id: str, doc_id: str, row_count: int) -> dict:
    return {
        "package_id": package_id,
        "resources": [
            {
                "document_id": doc_id,
                "title": f"Title for {package_id}",
                "subjects": ["s1"],
                "organization_code": "org",
                "file_format": "csv",
                "resource_last_modified": None,
                "row_count": row_count,
            }
        ],
    }


def test_extract_three_packages(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            _candidate_row("pkg-a", "doc-a", 5),
            _candidate_row("pkg-b", "doc-b", 5),
            _candidate_row("pkg-c", "doc-c", 5),
        ],
    )
    # Per-package column union + sample rows. Three packages, two
    # secondary queries each = six canned responses.
    for _ in range(3):
        bq.register_query(
            "JSON_KEYS(row)", [{"col_name": "x"}, {"col_name": "y"}]
        )
        bq.register_query(
            "WHERE document_id = @document_id",
            [
                {"row_index": 0, "row": {"x": "1", "y": "2"}},
                {"row_index": 1, "row": {"x": "3", "y": "4"}},
            ],
        )

    request = ExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)

    assert summary.packages_extracted == 3
    assert summary.packages_failed == 0
    assert summary.packages_skipped_already_extracted == 0

    # 3 rows landed in stage/r1/inputs/000.jsonl.
    out = tmp_path / "r1" / "inputs" / "000.jsonl"
    assert out.exists()
    lines = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert {row["package_id"] for row in lines} == {"pkg-a", "pkg-b", "pkg-c"}
    assert all(row["representative_document_id"] for row in lines)


def test_extract_resume_skips_already_extracted(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        # Second run: candidate query already excludes pkg-a, so the
        # response only has pkg-b. The orchestrator uses the prior
        # extraction count for the candidate-set total.
        [_candidate_row("pkg-b", "doc-b", 5)],
    )
    bq.register_query("JSON_KEYS(row)", [{"col_name": "x"}])
    bq.register_query(
        "WHERE document_id = @document_id",
        [{"row_index": 0, "row": {"x": "1"}}],
    )

    # Pre-populate stage with pkg-a (the "already extracted" file).
    inputs = tmp_path / "r1" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "000.jsonl").write_text(
        '{"package_id":"pkg-a","resources":[],"column_names":[],'
        '"column_names_truncated_to":null,"representative_document_id":"x",'
        '"sample_rows":[]}\n'
    )

    request = ExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)

    assert summary.packages_extracted == 1
    assert summary.packages_skipped_already_extracted == 1


def test_extract_hard_fails_on_no_rows(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [_candidate_row("pkg-empty", "doc-x", 0)],
    )
    request = ExtractRequest(
        run_id="r1",
        dry_run=False,
        limit_packages=None,
        limit_package_ids=None,
        limit_orgs=None,
    )
    summary = run_extract(request=request, settings=_settings(tmp_path), bq=bq)
    assert summary.packages_failed == 1
    assert summary.packages_extracted == 0


def test_extract_preflight_missing_project_id(tmp_path: Path) -> None:
    from semantic_enrich.core.dataset_extract import preflight

    bq = FakeBqClient()
    s = Settings(staging_dir=tmp_path)
    s = s.model_copy(update={"gcp_project_id": None})
    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        preflight(settings=s, bq=bq)
