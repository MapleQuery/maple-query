"""End-to-end datasets-generate with deterministic fake `generate_json`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.dataset_generator import (
    GenerateRequest,
    preflight,
    run_generate,
)

from .conftest import fake_generate_json_factory


def _make_sentinel_with_tokenizer():
    """A model sentinel whose tokenizer concatenates the user message
    into the rendered prompt, so a downstream fake `generate_json` can
    parse the package_id back out."""

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
        flush_every_n_packages=10,
    )


def _seed_inputs(tmp_path: Path, run_id: str, package_ids: list[str]) -> None:
    inputs = tmp_path / run_id / "inputs"
    inputs.mkdir(parents=True)
    with (inputs / "000.jsonl").open("w") as f:
        for pid in package_ids:
            f.write(
                json.dumps(
                    {
                        "package_id": pid,
                        "resources": [
                            {
                                "document_id": f"doc-{pid}",
                                "title": "T",
                                "subjects": [],
                                "organization_code": "org",
                                "file_format": "csv",
                                "resource_last_modified": None,
                                "row_count": 10,
                            }
                        ],
                        "column_names": ["a"],
                        "column_names_truncated_to": None,
                        "representative_document_id": f"doc-{pid}",
                        "sample_rows": [{"a": "1"}],
                    }
                )
                + "\n"
            )


def test_generate_three_packages_dry_run(tmp_path: Path) -> None:
    _seed_inputs(tmp_path, "r1", ["pkg-a", "pkg-b", "pkg-c"])
    request = GenerateRequest(run_id="r1", dry_run=True)
    summary = run_generate(request=request, settings=_settings(tmp_path))
    assert summary.packages_generated == 3
    out = tmp_path / "r1" / "datasets" / "000.jsonl"
    assert out.exists()
    lines = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert all(row["dry_run"] is True for row in lines)
    assert all(row["embedding"] is None for row in lines)


def test_generate_three_packages_with_fake_model(tmp_path: Path) -> None:
    _seed_inputs(tmp_path, "r1", ["pkg-a", "pkg-b", "pkg-c"])
    fake = fake_generate_json_factory()
    request = GenerateRequest(run_id="r1", dry_run=False)

    sentinel = _make_sentinel_with_tokenizer()
    summary = run_generate(
        request=request,
        settings=_settings(tmp_path),
        load_generation_model=lambda *args, **kwargs: sentinel,
        generate_json=fake,
    )
    assert summary.packages_generated == 3
    assert summary.packages_failed == 0
    out = tmp_path / "r1" / "datasets" / "000.jsonl"
    lines = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert {row["package_id"] for row in lines} == {"pkg-a", "pkg-b", "pkg-c"}


def test_generate_resume_skips_already_staged(tmp_path: Path) -> None:
    _seed_inputs(tmp_path, "r1", ["pkg-a", "pkg-b", "pkg-c"])
    # Pre-populate datasets/ with one row.
    datasets = tmp_path / "r1" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "000.jsonl").write_text(
        json.dumps(
            {
                "package_id": "pkg-a",
                "summary": "A" * 60,
                "grain": None,
                "measures": [],
                "dimensions": [],
                "date_range_start": None,
                "date_range_end": None,
                "embedding": None,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generation_model": "fake",
                "generation_model_commit": None,
                "generation_run_id": "r1",
                "dry_run": False,
            }
        )
        + "\n"
    )

    request = GenerateRequest(run_id="r1", dry_run=True)
    summary = run_generate(request=request, settings=_settings(tmp_path))
    assert summary.packages_skipped_already_staged == 1
    assert summary.packages_generated == 2


def test_generate_validation_failure_marks_failed(tmp_path: Path) -> None:
    _seed_inputs(tmp_path, "r1", ["pkg-a"])
    # Returns a summary that's too short → DatasetCard.validate raises.
    bad_fake = fake_generate_json_factory(
        default={
            "package_id": "pkg-a",
            "summary": "short",
            "grain": "",
            "measures": [],
            "dimensions": [],
            "date_range_start": None,
            "date_range_end": None,
        }
    )

    summary = run_generate(
        request=GenerateRequest(run_id="r1", dry_run=False),
        settings=_settings(tmp_path),
        load_generation_model=lambda *a, **k: _make_sentinel_with_tokenizer(),
        generate_json=bad_fake,
    )
    assert summary.packages_failed == 1
    assert summary.packages_generated == 0


def test_generate_preflight_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="inputs_dir_missing"):
        preflight(
            settings=_settings(tmp_path),
            request=GenerateRequest(run_id="missing", dry_run=False),
        )
