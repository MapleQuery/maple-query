"""End-to-end columns-reembed against `FakeBqClient` + `FakeOpenAIClient`."""
from __future__ import annotations

import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.reembed import (
    ColumnsReembedRequest,
    run_columns_reembed,
)

from .conftest import FakeBqClient
from .openai_fakes import FakeOpenAIClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        openai_api_key="sk-test",  # type: ignore[arg-type]
        openai_embedding_batch_size=3,
        openai_embedding_dim=1536,
    )


def _register_target(bq: FakeBqClient, rows: list[dict[str, object]]) -> None:
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("COUNT(*) AS n", [{"n": len(rows)}])
    bq.register_query(
        "SELECT package_id, column_name, description FROM `proj.semantic.columns`",
        rows,
    )


def _unit_vec(dim: int) -> list[float]:
    return [1.0 / math.sqrt(dim)] * dim


def test_columns_reembed_updates_five_rows(tmp_path: Path) -> None:
    seed = [
        {
            "package_id": "pkg-a",
            "column_name": f"c{i}",
            "description": f"description {i}",
        }
        for i in range(5)
    ]
    bq = FakeBqClient()
    _register_target(bq, seed)
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(1536))

    summary = run_columns_reembed(
        request=ColumnsReembedRequest(
            run_id="rex-cols", dry_run=False, batch_size=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
        openai_client=client,
    )
    assert summary.rows_read == 5
    assert summary.rows_embedded == 5
    assert summary.rows_merged == 5
    assert any(
        name.startswith("proj.semantic._columns_reembed_")
        for name in bq.staging_tables
    )
    assert bq.deleted_tables
    merge = next(
        c for c in bq.calls if c["kind"] == "execute" and "MERGE INTO" in c["sql"]
    )
    assert "AND t.column_name = s.column_name" in merge["sql"]
    assert "embedding = s.embedding" in merge["sql"]
    assert "description" not in merge["sql"]


def test_columns_reembed_dry_run(tmp_path: Path) -> None:
    seed = [
        {
            "package_id": "pkg-a",
            "column_name": "c1",
            "description": "some description",
        }
    ]
    bq = FakeBqClient()
    _register_target(bq, seed)
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(1536))

    summary = run_columns_reembed(
        request=ColumnsReembedRequest(
            run_id="rex-cols-dry", dry_run=True, batch_size=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
        openai_client=client,
    )
    assert summary.dry_run is True
    assert summary.rows_read == 1
    assert summary.rows_embedded == 0
    assert bq.staging_tables == {}
    assert client.calls == []
