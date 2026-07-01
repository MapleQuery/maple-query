"""End-to-end `columns-embed` with deterministic fake embedder."""
from __future__ import annotations

import json
import math
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.embedding_pass import (
    ColumnsEmbedRequest,
    run_columns_embed,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        embedding_batch_size=4,
        embedding_dim=8,
    )


def _seed_columns(
    tmp_path: Path,
    run_id: str,
    *,
    keys: list[tuple[str, str]],
    pre_embedded_keys: set[tuple[str, str]] | None = None,
    failure_keys: set[tuple[str, str]] | None = None,
) -> Path:
    pre_embedded_keys = pre_embedded_keys or set()
    failure_keys = failure_keys or set()
    columns_dir = tmp_path / run_id / "columns"
    columns_dir.mkdir(parents=True)
    path = columns_dir / "000.jsonl"
    with path.open("w") as f:
        for pid, col in keys:
            embedding = [0.1] * 8 if (pid, col) in pre_embedded_keys else None
            failed = (pid, col) in failure_keys
            f.write(
                json.dumps(
                    {
                        "package_id": pid,
                        "column_name": col,
                        "semantic_type": "text",
                        "description": "A canned column description, padded to "
                        "satisfy the 20-char minimum-length rule.",
                        "sample_values": [],
                        "embedding": embedding,
                        "generated_at": "2026-01-01T00:00:00+00:00",
                        "generation_model": "fake",
                        "generation_model_commit": None,
                        "generation_run_id": run_id,
                        "generation_failed": failed,
                        "failure_reason": "x" if failed else None,
                        "dry_run": False,
                    }
                )
                + "\n"
            )
    return path


def _good_embed(dim: int):
    def _fn(texts, *, model, batch_size=64):  # type: ignore[no-untyped-def]
        return [[1.0 / math.sqrt(dim)] * dim for _ in texts]
    return _fn


def test_embed_three_columns(tmp_path: Path) -> None:
    path = _seed_columns(
        tmp_path,
        "r1",
        keys=[("pkg-a", "c1"), ("pkg-a", "c2"), ("pkg-b", "c1")],
    )
    summary = run_columns_embed(
        request=ColumnsEmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        load_embedding_model=lambda *a, **k: object(),
        embed_batch=_good_embed(8),
    )
    assert summary.embeddings_written == 3
    assert summary.embeddings_failed == 0

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert all(r["embedding"] is not None for r in rows)
    assert all(len(r["embedding"]) == 8 for r in rows)


def test_embed_skips_failure_markers(tmp_path: Path) -> None:
    """Failure markers should be counted under skipped and NOT
    embedded — the load pass filters them out at coalesce time."""
    keys = [
        ("pkg-a", "c1"),  # good, embed
        ("pkg-b", "__failure_marker__"),  # failure marker
    ]
    _seed_columns(
        tmp_path,
        "r1",
        keys=keys,
        failure_keys={("pkg-b", "__failure_marker__")},
    )
    summary = run_columns_embed(
        request=ColumnsEmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        load_embedding_model=lambda *a, **k: object(),
        embed_batch=_good_embed(8),
    )
    assert summary.embeddings_written == 1
    assert summary.embeddings_skipped_already_embedded == 1
    assert summary.embeddings_failed == 0


def test_embed_resume_skips_already_embedded(tmp_path: Path) -> None:
    _seed_columns(
        tmp_path,
        "r1",
        keys=[("pkg-a", "c1"), ("pkg-a", "c2")],
        pre_embedded_keys={("pkg-a", "c1")},
    )
    summary = run_columns_embed(
        request=ColumnsEmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        load_embedding_model=lambda *a, **k: object(),
        embed_batch=_good_embed(8),
    )
    assert summary.embeddings_skipped_already_embedded == 1
    assert summary.embeddings_written == 1


def test_embed_wrong_dim_fails(tmp_path: Path) -> None:
    _seed_columns(tmp_path, "r1", keys=[("pkg-a", "c1")])

    def bad(texts, *, model, batch_size=64):  # type: ignore[no-untyped-def]
        return [[0.1] * 4 for _ in texts]

    summary = run_columns_embed(
        request=ColumnsEmbedRequest(run_id="r1", dry_run=False, batch_size=None),
        settings=_settings(tmp_path),
        load_embedding_model=lambda *a, **k: object(),
        embed_batch=bad,
    )
    assert summary.embeddings_failed == 1
    assert summary.embeddings_written == 0
