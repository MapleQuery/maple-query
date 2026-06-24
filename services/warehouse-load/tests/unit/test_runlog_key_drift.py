"""Every top-level key in real runlog JSONLs is known to RawRunlogRow.

Forward-compat guard. If ingest starts emitting a new field, this
test fails loudly so the field gets added to the model (and possibly
`raw_documents.json`) before it silently disappears under
`extra="ignore"`.
"""
from __future__ import annotations

import json
from pathlib import Path

from warehouse_load.types import RawRunlogRow


def test_runlog_keys_known_to_model(ingest_runlog_dir: Path) -> None:
    if not ingest_runlog_dir.is_dir():
        # No runlogs on disk in this checkout (e.g. fresh clone in
        # CI without ingest history). Skip rather than fail; the
        # check runs in the developer's normal environment where
        # runlogs exist.
        import pytest
        pytest.skip(f"no runlog directory at {ingest_runlog_dir}")

    seen_keys: set[str] = set()
    runlog_files = list(ingest_runlog_dir.glob("*.jsonl"))
    if not runlog_files:
        import pytest
        pytest.skip("no *.jsonl files in runlog directory")

    for path in runlog_files:
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                seen_keys.update(json.loads(stripped).keys())

    model_keys = set(RawRunlogRow.model_fields.keys())
    unknown = seen_keys - model_keys
    assert not unknown, (
        f"Runlog files contain keys unknown to RawRunlogRow: {sorted(unknown)}. "
        "Update the model (and raw.documents schema if needed) before this lands."
    )
