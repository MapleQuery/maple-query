"""JSONL flush/read/scan helpers for `stage/<run_id>/<artifact>/`.

Parameterised on the artifact name so the same buffer/flush manager
handles every per-pass JSONL bucket. Files are named
`<flush_seq:03d>.jsonl`; the writer seeds itself by scanning the dir,
so a crashed run resumes flushing into the next sequence without
colliding with prior files.

The reader returns `(path, line_index, row)` tuples so callers can
both rewrite a file in place (the embed pass, §8.2) and reference a
specific row in log events.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Artifact = Literal["inputs", "datasets", "column_inputs", "columns"]

FlushCallback = Callable[[Path, int, int], None]
"""Signature: `(path, flush_seq, row_count) -> None`."""


def stage_path(*, staging_dir: Path, run_id: str, artifact: Artifact) -> Path:
    """Resolve the `<staging_dir>/<run_id>/<artifact>/` directory."""
    return staging_dir / run_id / artifact


class StageWriter:
    """Buffered JSONL writer with periodic flushes.

    One instance per (run_id, artifact) pair. The buffer flushes every
    `flush_every` rows, on `close()`, and on explicit `flush()`. Files
    are named `<flush_seq:03d>.jsonl`; the seq monotonically increases
    per (run_id, artifact), so a crash-resume picks up at the next
    seq without colliding with the prior run's files.
    """

    def __init__(
        self,
        *,
        run_id: str,
        artifact: Artifact,
        staging_dir: Path,
        flush_every: int,
        on_flush: FlushCallback | None = None,
    ) -> None:
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1; got {flush_every}")
        self._dir = stage_path(
            staging_dir=staging_dir, run_id=run_id, artifact=artifact
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[BaseModel] = []
        self._flush_every = flush_every
        self._flush_seq = self._next_flush_seq()
        self._files_written = 0
        self._on_flush = on_flush

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def files_written(self) -> int:
        return self._files_written

    def _next_flush_seq(self) -> int:
        """Scan for existing files, return the next sequence number.

        A crashed run that wrote `000.jsonl` and `001.jsonl` resumes
        with `_flush_seq=2`. New files never overwrite old ones.
        """
        max_seen = -1
        for path in self._dir.glob("*.jsonl"):
            stem = path.stem
            try:
                max_seen = max(max_seen, int(stem))
            except ValueError:
                # Files not matching the `<int>.jsonl` shape (e.g. a
                # tempfile, an operator note) are ignored. The seq
                # advances past them too.
                continue
        return max_seen + 1

    def append(self, row: BaseModel) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self) -> Path | None:
        """Write the buffer to the next sequence file. Returns the path,
        or None if the buffer was empty (no file is created)."""
        if not self._buffer:
            return None
        seq = self._flush_seq
        path = self._dir / f"{seq:03d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in self._buffer:
                f.write(row.model_dump_json())
                f.write("\n")
        tmp.replace(path)
        row_count = len(self._buffer)
        self._buffer.clear()
        self._flush_seq += 1
        self._files_written += 1
        if self._on_flush is not None:
            self._on_flush(path, seq, row_count)
        return path

    def close(self) -> None:
        """Flush any partial buffer. Idempotent."""
        self.flush()


def iter_staged_rows[T: BaseModel](
    *,
    run_id: str,
    artifact: Artifact,
    staging_dir: Path,
    row_type: type[T],
) -> Iterator[tuple[Path, int, T]]:
    """Yield `(file_path, line_index, row)` per line.

    Iteration order is file-path ascending, then within-file line
    ascending. Each line is parsed through `row_type.model_validate_json`
    so the caller gets back a validated pydantic model.
    """
    dir_ = stage_path(staging_dir=staging_dir, run_id=run_id, artifact=artifact)
    if not dir_.is_dir():
        return
    for path in sorted(dir_.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield path, line_index, row_type.model_validate_json(line)


def read_staged_package_ids(
    *,
    run_id: str,
    artifact: Artifact,
    staging_dir: Path,
) -> set[str]:
    """Collect the set of `package_id`s already present under
    `<staging_dir>/<run_id>/<artifact>/`. Used by both
    `datasets-extract` (resume) and `datasets-generate` (skip
    already-staged packages without burning a generation call)."""
    dir_ = stage_path(staging_dir=staging_dir, run_id=run_id, artifact=artifact)
    ids: set[str] = set()
    if not dir_.is_dir():
        return ids
    for path in sorted(dir_.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse cheaply — we only need package_id. A full
                # pydantic validate per line would burn CPU on a
                # 3.7K-row resume scan.
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = obj.get("package_id")
                if isinstance(pid, str):
                    ids.add(pid)
    return ids


def atomic_rewrite_file(
    *, path: Path, rows: list[BaseModel]
) -> None:
    """Write `rows` to `path` via tempfile + atomic rename.

    Used by the embed pass to rewrite a JSONL file in place with
    `embedding` populated. The original is never partially overwritten:
    a crash mid-write leaves the `.tmp` file behind but the original is
    still complete and re-readable on resume.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)
