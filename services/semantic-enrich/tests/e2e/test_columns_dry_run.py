"""End-to-end dry-run chain for the columns pipeline.

3 packages, hand-crafted `raw.documents` / `raw.rows` /
`semantic.datasets` responses, dry-run on every stage, no model loads,
no BQ writes. Asserts the JSONL files land in the right shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_generator import (
    ColumnsGenerateRequest,
    run_generate,
)
from semantic_enrich.core.column_inputs import (
    ColumnsExtractRequest,
    run_extract,
)

from ..integration.conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        flush_every_n_packages=10,
        column_chunk_size=100,
        extract_concurrency=1,
    )


def test_dry_run_chain(tmp_path: Path) -> None:
    bq = FakeBqClient()
    # Three packages — one will produce >100 columns to force chunking
    # at columns-generate time.
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "package_id": f"pkg-{i}",
                "resources": [
                    {
                        "document_id": f"doc-{i}",
                        "title": f"pkg-{i}",  # encode pid in title
                        "description": "d",
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
    # Cross-pass summary lookup — empty (smoke testing without 4.4).
    bq.register_query("FROM `proj.semantic.datasets`", [])
    # Three packages: pkg-0 → 5 cols, pkg-1 → 5 cols, pkg-2 → 150 cols.
    pkg_col_counts = [5, 5, 150]
    for pkg_i, col_count in enumerate(pkg_col_counts):
        bq.register_query(
            "JSON_KEYS(row)",
            [
                {
                    "document_id": f"doc-{pkg_i}",
                    "columns": [f"c{i}" for i in range(col_count)],
                }
            ],
        )
        bq.register_query(
            "ranked AS",
            [
                {"col_name": f"c{i}", "v": str(i)}
                for i in range(min(col_count, 10))
            ],
        )

    extract_summary = run_extract(
        request=ColumnsExtractRequest(
            run_id="dry",
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
    assert extract_summary.packages_empty == 0

    col_inputs_path = tmp_path / "dry" / "column_inputs" / "000.jsonl"
    rows = [json.loads(line) for line in col_inputs_path.read_text().splitlines() if line]
    by_pid = {r["package_id"]: r for r in rows}
    assert len(by_pid["pkg-0"]["column_names"]) == 5
    assert len(by_pid["pkg-2"]["column_names"]) == 150

    generate_summary = run_generate(
        request=ColumnsGenerateRequest(run_id="dry", dry_run=True, chunk_size=100),
        settings=_settings(tmp_path),
    )
    assert generate_summary.packages_generated == 3
    # pkg-2 with 150 columns at chunk_size=100 → 2 chunks; others → 1
    # each. Total = 4.
    assert generate_summary.chunks_total == 4
    assert generate_summary.columns_generated == 5 + 5 + 150

    columns_path = tmp_path / "dry" / "columns" / "000.jsonl"
    column_rows = [
        json.loads(line) for line in columns_path.read_text().splitlines() if line
    ]
    assert len(column_rows) == 160
    assert all(r["dry_run"] for r in column_rows)
    assert all(r["embedding"] is None for r in column_rows)

    # Order is preserved across chunks for the wide package. The
    # extract emits the column union sorted lexicographically (the
    # fake BQ used to mask this by ignoring the SQL ORDER BY).
    pkg_2_rows = [r for r in column_rows if r["package_id"] == "pkg-2"]
    assert [r["column_name"] for r in pkg_2_rows] == sorted(
        f"c{i}" for i in range(150)
    )
