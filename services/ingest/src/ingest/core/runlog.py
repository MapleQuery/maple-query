"""Per-run JSONL log of every ingested resource.

The pipeline doesn't write to BigQuery yet. Instead, it appends one
JSON record per resource (success / quarantined / failed) to a local
file. A follow-up loader reads the JSONL into `raw.documents` — no
need to re-fetch from CKAN or re-derive metadata from GCS paths.

File layout:
- Default: `runlog/<timestamp>-<subject>-<short-uuid>.jsonl` relative
  to the operator's cwd — sortable by time, identifiable at a glance.
- If `INGEST_RUN_ID` is set explicitly (i.e. not a generated UUID),
  the filename becomes `<run_id>.jsonl` and multiple invocations with
  the same `INGEST_RUN_ID` append to the same file.
- Override the parent directory with `INGEST_RUNLOG_DIR=/path/to/dir`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingest.types import DocumentRow

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_UNSAFE_SUBJECT_CHARS = re.compile(r"[^a-z0-9_-]+")


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

    Datetimes/dates become ISO strings — same shape the eventual BQ
    loader will hand to `bigquery.Client.insert_rows_json`, so no
    transformation step needed at load time.
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


def default_runlog_path(
    *,
    run_id: str,
    subject: str,
    started_at: datetime,
    override_dir: Path | None = None,
) -> Path:
    """Build the JSONL path for this run.

    - If `run_id` looks like a generated UUID (the default), the
      filename is `<timestamp>-<subject>-<short-uuid>.jsonl` so a
      bare `ls runlog/` tells you what each file is.
    - If `run_id` is an explicit user override (e.g.
      `INGEST_RUN_ID=backfill-2026-05-25`), the filename is
      `<run_id>.jsonl` verbatim — multiple invocations with the same
      `run_id` append to the same file.
    """
    base = override_dir or Path("runlog")

    if _UUID_RE.match(run_id):
        ts = started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
        safe_subject = _UNSAFE_SUBJECT_CHARS.sub("-", subject.lower()).strip("-")
        if not safe_subject:
            safe_subject = "subject"
        short = run_id[:8]
        return base / f"{ts}-{safe_subject}-{short}.jsonl"

    return base / f"{run_id}.jsonl"
