"""Read JSONL runlog files from local disk or GCS.

Yields `RawRunlogRow` instances. Parse failures are surfaced via the
`ParseError` dataclass rather than raised — the loader counts and
logs them, then continues, because the runlog is immutable history
and one corrupted line shouldn't halt the load.

Local-disk reads are line-streamed. GCS reads pull each blob into
memory before iterating (per-blob bounded, not per-directory) — fine
at current per-blob sizes (largest seen <10 MB), revisit if a single
runlog blob grows past ~100 MB.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from warehouse_load.clients.gcs import GcsClient
from warehouse_load.types import RawRunlogRow


@dataclass(frozen=True)
class ParseError:
    """A line that didn't parse. Surfaces via `RunlogReadEvent`."""

    source: str  # path or gs:// object name
    line_number: int  # 1-based
    error: str


@dataclass(frozen=True)
class RunlogReadEvent:
    """One yield from the reader: either a row or a parse error.

    Using a sum type instead of two iterators lets the caller maintain
    a single counter loop while preserving per-source/line context for
    structured logging.
    """

    source: str
    row: RawRunlogRow | None
    error: ParseError | None

    @classmethod
    def ok(cls, source: str, row: RawRunlogRow) -> RunlogReadEvent:
        return cls(source=source, row=row, error=None)

    @classmethod
    def fail(cls, error: ParseError) -> RunlogReadEvent:
        return cls(source=error.source, row=None, error=error)


def iter_runlog_rows(
    *,
    local_dir: Path | None,
    gcs_prefix: str | None,
    gcs_client: GcsClient | None = None,
    since: datetime | None = None,
) -> Iterator[RunlogReadEvent]:
    """Iterate every runlog row from local disk then GCS.

    Precondition: at least one of (local_dir, gcs_prefix) is set.
    Both can be set; local is read first, then GCS.

    `since` is an optional `ingested_at` cutoff applied after parsing
    — it's a CLI-level convenience for ad-hoc reloads, not a
    watermark.
    """
    if local_dir is None and gcs_prefix is None:
        raise ValueError("at least one of local_dir, gcs_prefix must be set")

    if local_dir is not None and local_dir.is_dir():
        for path in sorted(local_dir.glob("*.jsonl")):
            yield from _iter_lines(
                source=str(path),
                lines=_open_lines(path),
                since=since,
            )

    if gcs_prefix is not None:
        if gcs_client is None:
            raise ValueError("gcs_client must be provided when gcs_prefix is set")
        for source, lines in gcs_client.list_jsonl(gcs_prefix):
            yield from _iter_lines(source=source, lines=lines, since=since)


def _open_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        yield from f


def _iter_lines(
    *,
    source: str,
    lines: Iterator[str],
    since: datetime | None,
) -> Iterator[RunlogReadEvent]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            yield RunlogReadEvent.fail(
                ParseError(source=source, line_number=line_number, error=f"json: {exc}"),
            )
            continue

        try:
            row = RawRunlogRow.model_validate(payload)
        except ValidationError as exc:
            yield RunlogReadEvent.fail(
                ParseError(source=source, line_number=line_number, error=f"schema: {exc}"),
            )
            continue

        if since is not None and row.ingested_at < since:
            continue

        yield RunlogReadEvent.ok(source, row)
