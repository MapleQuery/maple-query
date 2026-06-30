"""End-to-end dry-run wiring across extract → generate.

3 packages, hand-crafted `raw.documents` / `raw.rows` responses,
dry-run on both stages, no model loads, no BQ writes. Asserts the
JSONL files land in the right shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.dataset_extract import ExtractRequest, run_extract
from semantic_enrich.core.dataset_generator import GenerateRequest, run_generate

from ..integration.conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        flush_every_n_packages=10,
        sample_rows_per_package=2,
        extract_concurrency=1,
    )


def test_dry_run_chain(tmp_path: Path) -> None:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "package_id": f"pkg-{i}",
                "resources": [
                    {
                        "document_id": f"doc-{i}",
                        "title": f"Title-{i}",
                        "subjects": ["s"],
                        "organization_code": "org",
                        "file_format": "csv",
                        "resource_last_modified": None,
                        "row_count": 5,
                    }
                ],
            }
            for i in range(3)
        ],
    )
    for _ in range(3):
        bq.register_query("JSON_KEYS(row)", [{"col_name": "x"}])
        bq.register_query(
            "WHERE document_id = @document_id",
            [{"row_index": 0, "row": {"x": "1"}}, {"row_index": 1, "row": {"x": "2"}}],
        )

    extract_summary = run_extract(
        request=ExtractRequest(
            run_id="dry-3",
            dry_run=True,
            limit_packages=None,
            limit_package_ids=None,
            limit_orgs=None,
        ),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert extract_summary.packages_extracted == 3
    assert extract_summary.candidate_count == 3
    assert extract_summary.packages_failed == 0

    inputs_path = tmp_path / "dry-3" / "inputs" / "000.jsonl"
    assert inputs_path.exists()
    input_rows = [
        json.loads(line) for line in inputs_path.read_text().splitlines() if line
    ]
    assert len(input_rows) == 3

    generate_summary = run_generate(
        request=GenerateRequest(run_id="dry-3", dry_run=True),
        settings=_settings(tmp_path),
    )
    assert generate_summary.packages_generated == 3
    assert generate_summary.input_row_count == 3

    datasets_path = tmp_path / "dry-3" / "datasets" / "000.jsonl"
    dataset_rows = [
        json.loads(line) for line in datasets_path.read_text().splitlines() if line
    ]
    assert len(dataset_rows) == 3
    assert all(row["dry_run"] is True for row in dataset_rows)
    assert all(row["embedding"] is None for row in dataset_rows)
