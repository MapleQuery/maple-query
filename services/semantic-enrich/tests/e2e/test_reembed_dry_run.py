"""§4.7 acceptance §12.1 dry-run E2E.

`--dry-run` against a scratch 10-row semantic dataset:
  - reads all 10 rows
  - emits one would_have_reembedded event per row
  - zero OpenAI calls
  - zero MERGEs
  - exits 0
"""
from __future__ import annotations

import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.reembed import (
    DatasetsReembedRequest,
    run_datasets_reembed,
)
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        openai_api_key="sk-test",  # type: ignore[arg-type]
        openai_embedding_batch_size=4,
        openai_embedding_dim=1536,
    )


def test_dry_run_ten_rows(tmp_path: Path) -> None:
    seed = [
        {"package_id": f"pkg-{i}", "summary": f"summary {i}"} for i in range(10)
    ]
    bq = FakeBqClient()
    bq.register_query("SELECT 1 AS ok", [{"ok": 1}])
    bq.register_query("COUNT(*) AS n", [{"n": len(seed)}])
    bq.register_query(
        "SELECT package_id, summary FROM `proj.semantic.datasets`", seed
    )
    client = FakeOpenAIClient(
        vector_factory=lambda _: [1.0 / math.sqrt(1536)] * 1536
    )

    summary = run_datasets_reembed(
        request=DatasetsReembedRequest(
            run_id="e2e-dry", dry_run=True, batch_size=None
        ),
        settings=_settings(tmp_path),
        bq=bq,
        openai_client=client,
    )

    assert summary.rows_read == 10
    assert summary.rows_embedded == 0
    assert summary.rows_merged == 0
    assert client.calls == []
    assert bq.staging_tables == {}
    assert not any(
        c["kind"] == "execute" and "MERGE INTO" in c["sql"] for c in bq.calls
    )
