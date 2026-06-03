from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ingest.core.runlog import default_runlog_path

GENERATED_UUID = "3f9a2b1c-7d4e-4f8a-9c1d-2e3f4a5b6c7d"
STARTED = datetime(2026, 5, 25, 14, 30, 15, tzinfo=UTC)


def test_default_uuid_produces_descriptive_filename() -> None:
    p = default_runlog_path(
        run_id=GENERATED_UUID,
        subject="government_and_politics",
        started_at=STARTED,
    )
    assert p == Path("runlog/2026-05-25T14-30-15Z-government_and_politics-3f9a2b1c.jsonl")


def test_explicit_run_id_used_verbatim() -> None:
    # User-set INGEST_RUN_ID — descriptive pattern is bypassed so multiple
    # invocations can append to the same consolidated file.
    p = default_runlog_path(
        run_id="backfill-2026-05-25",
        subject="government_and_politics",
        started_at=STARTED,
    )
    assert p == Path("runlog/backfill-2026-05-25.jsonl")


def test_override_dir_applies_to_both_patterns(tmp_path: Path) -> None:
    p1 = default_runlog_path(
        run_id=GENERATED_UUID, subject="x", started_at=STARTED, override_dir=tmp_path
    )
    p2 = default_runlog_path(
        run_id="custom", subject="x", started_at=STARTED, override_dir=tmp_path
    )
    assert p1.parent == tmp_path
    assert p2.parent == tmp_path


def test_subject_sanitised() -> None:
    p = default_runlog_path(
        run_id=GENERATED_UUID,
        subject="WEIRD/Subject With Spaces!",
        started_at=STARTED,
    )
    # Lowercased; non-[a-z0-9_-] runs collapsed to single dash; edges trimmed.
    assert p.name == "2026-05-25T14-30-15Z-weird-subject-with-spaces-3f9a2b1c.jsonl"


def test_subject_that_sanitises_to_empty_falls_back() -> None:
    p = default_runlog_path(
        run_id=GENERATED_UUID, subject="!!!", started_at=STARTED
    )
    assert p.name == "2026-05-25T14-30-15Z-subject-3f9a2b1c.jsonl"
