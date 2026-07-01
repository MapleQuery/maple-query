"""End-to-end `columns-generate` with a deterministic fake."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_generator import (
    ColumnsGenerateRequest,
    run_generate,
)

from .conftest import fake_generate_json_list_factory


def _make_tokenizer_sentinel():
    """A model sentinel whose tokenizer concatenates the user message
    so the fake generate_json_list can parse the per-chunk metadata
    back out of the rendered prompt."""

    class _Tok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return "\n".join(m["content"] for m in messages)

    class _Sentinel:
        tokenizer = _Tok()

    return _Sentinel()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        staging_dir=tmp_path,
        column_chunk_size=100,
        flush_every_n_packages=10,
    )


def _seed_column_inputs(
    tmp_path: Path,
    run_id: str,
    *,
    package_id: str,
    column_count: int,
    title: str | None = None,
) -> None:
    inputs_dir = tmp_path / run_id / "column_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"c{i}" for i in range(column_count)]
    payload = {
        "package_id": package_id,
        # Encode the package_id in the title so the fake can extract
        # it from the rendered prompt.
        "package_title": title or package_id,
        "package_subjects": [],
        "package_summary": None,
        "representative_document_id": "doc",
        "column_names": columns,
        "sample_values": {c: ["v"] for c in columns},
        "dropped_columns": [],
        "overflow_column_count": 0,
        "extracted_at": "2026-01-01T00:00:00+00:00",
    }
    with (inputs_dir / "000.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")


def test_generate_one_package_one_chunk_dry_run(tmp_path: Path) -> None:
    _seed_column_inputs(tmp_path, "r1", package_id="pkg-a", column_count=5)
    summary = run_generate(
        request=ColumnsGenerateRequest(run_id="r1", dry_run=True, chunk_size=None),
        settings=_settings(tmp_path),
    )
    assert summary.packages_generated == 1
    assert summary.columns_generated == 5
    assert summary.chunks_total == 1

    out = tmp_path / "r1" / "columns" / "000.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 5
    assert all(r["dry_run"] for r in rows)


def test_generate_wide_package_chunks_correctly(tmp_path: Path) -> None:
    _seed_column_inputs(tmp_path, "r1", package_id="pkg-a", column_count=250)
    summary = run_generate(
        request=ColumnsGenerateRequest(run_id="r1", dry_run=False, chunk_size=100),
        settings=_settings(tmp_path),
        load_generation_model=lambda *a, **k: _make_tokenizer_sentinel(),
        generate_json_list=fake_generate_json_list_factory(),
    )
    # 250 columns / 100 chunk_size = 3 chunks (100, 100, 50).
    assert summary.chunks_total == 3
    assert summary.columns_generated == 250
    assert summary.packages_generated == 1
    assert summary.packages_failed == 0

    out = tmp_path / "r1" / "columns" / "000.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 250
    assert [r["column_name"] for r in rows] == [f"c{i}" for i in range(250)]


def test_generate_retries_on_invariant_violation(tmp_path: Path) -> None:
    """First-chunk response is short-by-one; retry returns the full
    list. The package succeeds with chunks_retried=1."""
    _seed_column_inputs(tmp_path, "r1", package_id="pkg-a", column_count=3)

    # The fake's call count drives the bad-then-good behaviour.
    call_count = {"n": 0}

    def fn(prompt: str, schema: object, *, model: object,
           max_tokens: int = 1500, temperature: float = 0.0) -> list[dict[str, Any]]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Length mismatch — only 2 entries for 3 columns.
            return [
                {"column_name": "c0", "description": "x" * 40, "sample_values": []},
                {"column_name": "c1", "description": "y" * 40, "sample_values": []},
            ]
        return [
            {"column_name": f"c{i}", "description": "z" * 40, "sample_values": []}
            for i in range(3)
        ]

    summary = run_generate(
        request=ColumnsGenerateRequest(run_id="r1", dry_run=False, chunk_size=100),
        settings=_settings(tmp_path),
        load_generation_model=lambda *a, **k: _make_tokenizer_sentinel(),
        generate_json_list=fn,
    )
    assert summary.packages_generated == 1
    assert summary.chunks_retried == 1
    assert call_count["n"] == 2


def test_generate_fails_package_when_retry_also_fails(tmp_path: Path) -> None:
    _seed_column_inputs(tmp_path, "r1", package_id="pkg-a", column_count=3)

    def fn(prompt: str, schema: object, *, model: object,
           max_tokens: int = 1500, temperature: float = 0.0) -> list[dict[str, Any]]:
        # Always short-by-one.
        return [
            {"column_name": "c0", "description": "x" * 40, "sample_values": []},
            {"column_name": "c1", "description": "y" * 40, "sample_values": []},
        ]

    summary = run_generate(
        request=ColumnsGenerateRequest(run_id="r1", dry_run=False, chunk_size=100),
        settings=_settings(tmp_path),
        load_generation_model=lambda *a, **k: _make_tokenizer_sentinel(),
        generate_json_list=fn,
    )
    assert summary.packages_failed == 1
    assert summary.packages_generated == 0

    # Failure-marker line is staged for gap-fill.
    out = tmp_path / "r1" / "columns" / "000.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["generation_failed"] is True
    assert rows[0]["failure_reason"] == "chunk_invariant_violation_after_retry"


def test_generate_resume_skips_already_staged(tmp_path: Path) -> None:
    _seed_column_inputs(tmp_path, "r1", package_id="pkg-a", column_count=2)

    # Pre-seed columns/ with pkg-a already done.
    columns_dir = tmp_path / "r1" / "columns"
    columns_dir.mkdir(parents=True)
    (columns_dir / "000.jsonl").write_text(
        json.dumps(
            {
                "package_id": "pkg-a",
                "column_name": "c0",
                "semantic_type": "text",
                "description": "x" * 40,
                "sample_values": [],
                "embedding": None,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generation_model": "fake",
                "generation_model_commit": None,
                "generation_run_id": "r1",
                "generation_failed": False,
                "failure_reason": None,
                "dry_run": False,
            }
        )
        + "\n"
    )

    calls = {"n": 0}

    def fn(*a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise AssertionError("should not be called for already-staged package")

    summary = run_generate(
        request=ColumnsGenerateRequest(run_id="r1", dry_run=False, chunk_size=100),
        settings=_settings(tmp_path),
        load_generation_model=lambda *a, **k: _make_tokenizer_sentinel(),
        generate_json_list=fn,
    )
    assert summary.packages_skipped_already_staged == 1
    assert calls["n"] == 0
