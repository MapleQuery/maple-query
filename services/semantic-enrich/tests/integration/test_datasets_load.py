"""End-to-end datasets-load against a FakeBqClient."""
from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.datasets_load import (
    LoadRequest,
    _build_merge_sql,
    run_load,
)

from .conftest import FakeBqClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
    )


def _seed_loadable(tmp_path: Path, run_id: str, ids: list[str]) -> None:
    datasets = tmp_path / run_id / "datasets"
    datasets.mkdir(parents=True)
    with (datasets / "000.jsonl").open("w") as f:
        for pid in ids:
            f.write(
                json.dumps(
                    {
                        "package_id": pid,
                        "summary": (
                            "A canned summary for the load-pass test, "
                            "padded to satisfy the minimum-length rule."
                        ),
                        "grain": None,
                        "measures": [],
                        "dimensions": [],
                        "date_range_start": None,
                        "date_range_end": None,
                        "embedding": [0.1] * 1024,
                        "generated_at": "2026-01-01T00:00:00+00:00",
                        "generation_model": "fake",
                        "generation_model_commit": None,
                        "generation_run_id": run_id,
                        "dry_run": False,
                    }
                )
                + "\n"
            )


def test_load_three_packages_inserts(tmp_path: Path) -> None:
    _seed_loadable(tmp_path, "r1", ["pkg-a", "pkg-b", "pkg-c"])
    bq = FakeBqClient()
    # Preflight `SELECT 1` succeeds (FakeBqClient returns []).
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    # `before` snapshot — no rows present → all inserts.
    bq.register_query("FROM `proj.semantic.datasets`", [])
    summary = run_load(
        request=LoadRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.rows_inserted == 3
    assert summary.rows_updated == 0
    assert summary.rows_unchanged == 0
    # Staging table created, load + MERGE executed, table dropped.
    assert any(name.startswith("proj.semantic._datasets_staging_")
               for name in bq.staging_tables)
    assert bq.deleted_tables
    assert any(call["kind"] == "execute" and "MERGE INTO" in call["sql"]
               for call in bq.calls)


def test_load_excludes_embedding_null(tmp_path: Path) -> None:
    # Two rows: one loadable, one not.
    datasets = tmp_path / "r1" / "datasets"
    datasets.mkdir(parents=True)
    with (datasets / "000.jsonl").open("w") as f:
        for pid, embedding in (("pkg-a", [0.1] * 1024), ("pkg-b", None)):
            f.write(
                json.dumps(
                    {
                        "package_id": pid,
                        "summary": (
                            "A canned summary for the load-pass test, "
                            "padded to satisfy the minimum-length rule."
                        ),
                        "grain": None,
                        "measures": [],
                        "dimensions": [],
                        "date_range_start": None,
                        "date_range_end": None,
                        "embedding": embedding,
                        "generated_at": "2026-01-01T00:00:00+00:00",
                        "generation_model": "fake",
                        "generation_model_commit": None,
                        "generation_run_id": "r1",
                        "dry_run": False,
                    }
                )
                + "\n"
            )
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.datasets`", [])
    summary = run_load(
        request=LoadRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.coalesced_row_count == 2
    assert summary.embedding_null_count == 1
    assert summary.rows_staged == 1
    assert summary.rows_inserted == 1


def test_load_dry_run(tmp_path: Path) -> None:
    _seed_loadable(tmp_path, "r1", ["pkg-a"])
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("FROM `proj.semantic.datasets`", [])
    summary = run_load(
        request=LoadRequest(run_id="r1", dry_run=True),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.dry_run is True
    assert summary.rows_inserted == 1
    # Dry-run should NOT touch the staging table.
    assert bq.staging_tables == {}
    assert bq.deleted_tables == []


def test_load_rejects_dry_run_placeholder(tmp_path: Path) -> None:
    """A dry-run placeholder summary is < 50 chars; the loader refuses."""
    datasets = tmp_path / "r1" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "000.jsonl").write_text(
        json.dumps(
            {
                "package_id": "pkg-a",
                "summary": "DRY_RUN_PLACEHOLDER",
                "grain": None,
                "measures": [],
                "dimensions": [],
                "date_range_start": None,
                "date_range_end": None,
                "embedding": [0.1] * 1024,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generation_model": "fake",
                "generation_model_commit": None,
                "generation_run_id": "r1",
                "dry_run": True,
            }
        )
        + "\n"
    )
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    with pytest.raises(RuntimeError, match="dry-run placeholder"):
        run_load(
            request=LoadRequest(run_id="r1", dry_run=False),
            settings=_settings(tmp_path),
            bq=bq,
        )


def test_merge_sql_shape() -> None:
    sql = _build_merge_sql(target="t.semantic.datasets", staging="t.semantic._stg")
    assert "MERGE INTO `t.semantic.datasets` t" in sql
    assert "USING `t.semantic._stg` s" in sql
    assert "ON t.package_id = s.package_id" in sql
    assert "WHEN MATCHED AND s.generated_at > t.generated_at" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    # No DELETE clause.
    assert "DELETE" not in sql
    # The 8 non-key columns appear in the UPDATE SET.
    for col in (
        "summary", "grain", "measures", "dimensions",
        "date_range_start", "date_range_end", "embedding", "generated_at",
    ):
        assert f"{col}" in sql


def test_load_idempotent_reload(tmp_path: Path) -> None:
    """Second load against the same generated_at: zero updates."""
    _seed_loadable(tmp_path, "r1", ["pkg-a"])
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    # Target already has pkg-a at the same generated_at.
    from datetime import datetime

    bq.register_query(
        "FROM `proj.semantic.datasets`",
        [
            {
                "package_id": "pkg-a",
                "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        ],
    )
    summary = run_load(
        request=LoadRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        bq=bq,
    )
    assert summary.rows_inserted == 0
    assert summary.rows_updated == 0
    assert summary.rows_unchanged == 1
