"""End-to-end datasets-reembed against `FakeBqClient` + `FakeOpenAIClient`.

Covers §4.7 acceptance §12.1 (dry-run row-count + zero side effects)
and §12.2 (real-run all-updates + staging cleanup).
"""
from __future__ import annotations

import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.reembed import (
    DatasetsReembedRequest,
    run_datasets_reembed,
)

from .conftest import FakeBqClient
from .openai_fakes import FakeOpenAIClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        openai_api_key="sk-test",  # type: ignore[arg-type]
        openai_embedding_batch_size=2,
        openai_embedding_dim=1536,
    )


def _register_target(bq: FakeBqClient, rows: list[dict[str, object]]) -> None:
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("COUNT(*) AS n", [{"n": len(rows)}])
    bq.register_query(
        "SELECT package_id, summary FROM `proj.semantic.datasets`", rows
    )


def _unit_vec(dim: int) -> list[float]:
    return [1.0 / math.sqrt(dim)] * dim


def test_datasets_reembed_updates_five_rows(tmp_path: Path) -> None:
    seed = [
        {
            "package_id": f"pkg-{i}",
            "summary": f"summary text {i}",
        }
        for i in range(5)
    ]
    bq = FakeBqClient()
    _register_target(bq, seed)
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(1536))

    summary = run_datasets_reembed(
        request=DatasetsReembedRequest(
            run_id="rex-1", dry_run=False, batch_size=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
        openai_client=client,
    )

    assert summary.rows_read == 5
    assert summary.rows_embedded == 5
    assert summary.rows_failed == 0
    assert summary.rows_merged == 5
    # Staging table lifecycle.
    assert any(
        name.startswith("proj.semantic._datasets_reembed_")
        for name in bq.staging_tables
    )
    assert bq.deleted_tables
    # A MERGE that only touches embedding + generated_at.
    merge = next(
        c for c in bq.calls if c["kind"] == "execute" and "MERGE INTO" in c["sql"]
    )
    assert "embedding = s.embedding" in merge["sql"]
    assert "generated_at = CURRENT_TIMESTAMP()" in merge["sql"]
    assert "summary" not in merge["sql"]
    # OpenAI got a preflight ping + 3 batches of 2 (batch_size=2).
    assert client.calls[0] == ["ping"]
    assert len(client.calls) == 1 + 3  # ping + 5 rows / 2 per batch → 3 batches


def test_datasets_reembed_dry_run_no_side_effects(tmp_path: Path) -> None:
    seed = [{"package_id": "pkg-a", "summary": "one"}]
    bq = FakeBqClient()
    _register_target(bq, seed)
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(1536))

    summary = run_datasets_reembed(
        request=DatasetsReembedRequest(
            run_id="rex-dry", dry_run=True, batch_size=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
        openai_client=client,
    )
    assert summary.dry_run is True
    assert summary.rows_read == 1
    assert summary.rows_embedded == 0
    assert summary.rows_merged == 0
    # No staging + no MERGE.
    assert bq.staging_tables == {}
    assert not any(
        c["kind"] == "execute" and "MERGE INTO" in c["sql"] for c in bq.calls
    )
    # And no OpenAI calls at all — dry-run must not touch the vendor.
    assert client.calls == []


def test_datasets_reembed_wrong_dim_fails_preflight(tmp_path: Path) -> None:
    """Preflight ping catches a wrong-dim vector before any per-row work."""
    import pytest

    seed = [{"package_id": "pkg-a", "summary": "one"}]
    bq = FakeBqClient()
    _register_target(bq, seed)
    client = FakeOpenAIClient(vector_factory=lambda _: [0.1] * 4)

    with pytest.raises(RuntimeError, match="unexpected shape"):
        run_datasets_reembed(
            request=DatasetsReembedRequest(
                run_id="rex-bad", dry_run=False, batch_size=None
            ),
            settings=_settings(tmp_path),
            bq=bq,
            openai_client=client,
        )
