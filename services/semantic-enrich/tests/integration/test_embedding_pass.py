"""End-to-end datasets-embed with deterministic fake OpenAI client."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.embedding_pass import EmbedRequest, run_embed
from semantic_enrich.types import StagedDatasetCard

from .openai_fakes import FakeOpenAIClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        openai_embedding_batch_size=4,
        openai_embedding_dim=8,
    )


def _seed_datasets(
    tmp_path: Path,
    run_id: str,
    package_ids: list[str],
    pre_embedded_ids: set[str] | None = None,
) -> Path:
    pre_embedded_ids = pre_embedded_ids or set()
    datasets = tmp_path / run_id / "datasets"
    datasets.mkdir(parents=True)
    path = datasets / "000.jsonl"
    with path.open("w") as f:
        for pid in package_ids:
            embedding = [0.1] * 8 if pid in pre_embedded_ids else None
            f.write(
                json.dumps(
                    {
                        "package_id": pid,
                        "summary": f"Summary for {pid}, padded to satisfy "
                                   "the minimum length requirement of the "
                                   "DatasetCard schema in the test suite.",
                        "grain": None,
                        "measures": [],
                        "dimensions": [],
                        "date_range_start": None,
                        "date_range_end": None,
                        "embedding": embedding,
                        "generated_at": "2026-01-01T00:00:00+00:00",
                        "generation_model": "fake",
                        "generation_model_commit": None,
                        "generation_run_id": run_id,
                        "dry_run": False,
                    }
                )
                + "\n"
            )
    return path


def _unit_vec(dim: int) -> list[float]:
    return [1.0 / math.sqrt(dim)] * dim


def test_embed_three_packages(tmp_path: Path) -> None:
    path = _seed_datasets(tmp_path, "r1", ["pkg-a", "pkg-b", "pkg-c"])
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(8))
    summary = run_embed(
        request=EmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        openai_client=client,
    )
    assert summary.embeddings_written == 3
    assert summary.embeddings_failed == 0

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert all(r["embedding"] is not None for r in rows)
    assert all(len(r["embedding"]) == 8 for r in rows)


def test_embed_resume_skips_already_embedded(tmp_path: Path) -> None:
    _seed_datasets(
        tmp_path,
        "r1",
        ["pkg-a", "pkg-b", "pkg-c"],
        pre_embedded_ids={"pkg-a", "pkg-c"},
    )
    client = FakeOpenAIClient(vector_factory=lambda _: _unit_vec(8))
    summary = run_embed(
        request=EmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        openai_client=client,
    )
    assert summary.embeddings_skipped_already_embedded == 2
    assert summary.embeddings_written == 1
    # Only pkg-b's summary should be sent to the embedder.
    flat = [t for batch in client.calls for t in batch]
    assert any("pkg-b" in t for t in flat)
    assert not any("pkg-a" in t for t in flat)


def test_embed_wrong_dim_failure(tmp_path: Path) -> None:
    _seed_datasets(tmp_path, "r1", ["pkg-a"])
    client = FakeOpenAIClient(vector_factory=lambda _: [0.1] * 4)
    summary = run_embed(
        request=EmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        openai_client=client,
    )
    assert summary.embeddings_failed == 1
    assert summary.embeddings_written == 0


def test_embed_nan_failure(tmp_path: Path) -> None:
    _seed_datasets(tmp_path, "r1", ["pkg-a"])
    client = FakeOpenAIClient(vector_factory=lambda _: [float("nan")] * 8)
    summary = run_embed(
        request=EmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        openai_client=client,
    )
    assert summary.embeddings_failed == 1


def test_staged_dataset_card_extra_field_rejected() -> None:
    """`extra="forbid"` belt-and-suspenders."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        StagedDatasetCard.model_validate(
            {
                "package_id": "p",
                "summary": "A" * 60,
                "grain": None,
                "measures": [],
                "dimensions": [],
                "date_range_start": None,
                "date_range_end": None,
                "embedding": None,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generation_model": "x",
                "generation_model_commit": None,
                "generation_run_id": "r1",
                "secret_field": "nope",
            }
        )
