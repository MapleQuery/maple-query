"""Per-run JSONL log of every ingested resource.

Phase A1 doesn't write to BigQuery yet. Instead, the pipeline appends
one JSON record per resource (success / quarantined / failed) to a
local file. When Phase A2 (the BQ catalog task) lands, that JSONL is
the input — load it into `raw.documents`, no need to re-fetch from
CKAN or re-derive metadata from GCS paths.

File layout:
- Default: `runlog/<run_id>.jsonl` relative to the operator's cwd.
- Override with `INGEST_RUNLOG_DIR=/path/to/dir`.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingest.types import DocumentRow


class RunLogWriter:
    """Append-only JSONL writer. Opens the file lazily on first write."""

    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._fh: Any = None

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> RunLogWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write_row(self, row: DocumentRow) -> None:
        if self._fh is None:
            raise RuntimeError("RunLogWriter must be used as a context manager")
        self._fh.write(json.dumps(_row_to_dict(row), separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()


def _row_to_dict(row: DocumentRow) -> dict[str, Any]:
    """Convert DocumentRow to a JSON-serialisable dict.

    Datetimes/dates become ISO strings — same shape Phase A2 will hand
    to `bigquery.Client.insert_rows_json`, so no transformation step
    needed at load time.
    """
    d = asdict(row)
    for k in (
        "http_last_modified",
        "resource_last_modified",
        "metadata_modified",
        "ingested_at",
    ):
        v = d.get(k)
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    pub = d.get("published_date")
    if isinstance(pub, date):
        d["published_date"] = pub.isoformat()
    return d


def default_runlog_path(*, run_id: str, override_dir: Path | None = None) -> Path:
    base = override_dir or Path("runlog")
    return base / f"{run_id}.jsonl"
