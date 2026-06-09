"""§11.3 — closed-enum value drift on real runlog files.

§11.2 catches new keys; this catches new values on `language`,
`ingestion_status`, `file_format`. Surfaces drift as a clear "update
the Literal" failure rather than an opaque parse-error spike.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from warehouse_load.types import FILE_FORMATS, INGESTION_STATUSES, LANGUAGES

CLOSED_ENUMS: dict[str, frozenset[str]] = {
    "language": LANGUAGES,
    "ingestion_status": INGESTION_STATUSES,
    "file_format": FILE_FORMATS,
}


def test_runlog_closed_enum_values_known_to_model(ingest_runlog_dir: Path) -> None:
    if not ingest_runlog_dir.is_dir():
        pytest.skip(f"no runlog directory at {ingest_runlog_dir}")

    runlog_files = list(ingest_runlog_dir.glob("*.jsonl"))
    if not runlog_files:
        pytest.skip("no *.jsonl files in runlog directory")

    seen: dict[str, set[str]] = {k: set() for k in CLOSED_ENUMS}

    for path in runlog_files:
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                for field in CLOSED_ENUMS:
                    val = row.get(field)
                    if val is None:
                        continue
                    seen[field].add(val)

    unknown_summary: dict[str, list[str]] = {}
    for field, expected in CLOSED_ENUMS.items():
        unknown = seen[field] - expected
        if unknown:
            unknown_summary[field] = sorted(unknown)

    assert not unknown_summary, (
        f"Runlog files contain closed-enum values unknown to the model: "
        f"{unknown_summary}. Update the Literal in RawRunlogRow (warehouse_load/types.py) "
        "and the matching set constant (LANGUAGES/INGESTION_STATUSES/FILE_FORMATS) before this lands."
    )
