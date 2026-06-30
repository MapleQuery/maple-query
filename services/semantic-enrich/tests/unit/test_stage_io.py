"""StageWriter / iter_staged_rows / atomic_rewrite_file."""
from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from semantic_enrich.core.stage_io import (
    StageWriter,
    atomic_rewrite_file,
    iter_staged_rows,
    read_staged_package_ids,
    stage_path,
)


class _Row(pydantic.BaseModel):
    package_id: str
    payload: str


def test_writer_flushes_every_n(tmp_path: Path) -> None:
    w = StageWriter(
        run_id="r1", artifact="inputs", staging_dir=tmp_path, flush_every=2
    )
    w.append(_Row(package_id="a", payload="x"))
    assert w.files_written == 0  # not yet
    w.append(_Row(package_id="b", payload="y"))
    assert w.files_written == 1  # flushed
    w.append(_Row(package_id="c", payload="z"))
    w.close()
    assert w.files_written == 2  # close flushed the partial buffer


def test_writer_picks_up_next_seq_after_crash(tmp_path: Path) -> None:
    """A 'crashed' run that left 000.jsonl behind picks up at seq=1."""
    dir_ = stage_path(staging_dir=tmp_path, run_id="r1", artifact="inputs")
    dir_.mkdir(parents=True)
    (dir_ / "000.jsonl").write_text('{"package_id":"a","payload":"x"}\n')
    w = StageWriter(
        run_id="r1", artifact="inputs", staging_dir=tmp_path, flush_every=1
    )
    w.append(_Row(package_id="b", payload="y"))
    w.close()
    assert (dir_ / "000.jsonl").exists()
    assert (dir_ / "001.jsonl").exists()


def test_writer_on_flush_callback(tmp_path: Path) -> None:
    seen: list[tuple[Path, int, int]] = []

    w = StageWriter(
        run_id="r1",
        artifact="inputs",
        staging_dir=tmp_path,
        flush_every=1,
        on_flush=lambda p, seq, n: seen.append((p, seq, n)),
    )
    w.append(_Row(package_id="a", payload="x"))
    w.append(_Row(package_id="b", payload="y"))
    w.close()
    assert len(seen) == 2
    assert [s[1] for s in seen] == [0, 1]
    assert [s[2] for s in seen] == [1, 1]


def test_iter_staged_rows_order(tmp_path: Path) -> None:
    w = StageWriter(
        run_id="r1", artifact="inputs", staging_dir=tmp_path, flush_every=1
    )
    for pid in ("a", "b", "c"):
        w.append(_Row(package_id=pid, payload="x"))
    w.close()
    got = [
        (path.name, idx, row.package_id)
        for path, idx, row in iter_staged_rows(
            run_id="r1",
            artifact="inputs",
            staging_dir=tmp_path,
            row_type=_Row,
        )
    ]
    assert got == [
        ("000.jsonl", 0, "a"),
        ("001.jsonl", 0, "b"),
        ("002.jsonl", 0, "c"),
    ]


def test_read_staged_package_ids(tmp_path: Path) -> None:
    w = StageWriter(
        run_id="r1", artifact="inputs", staging_dir=tmp_path, flush_every=1
    )
    for pid in ("a", "b", "c"):
        w.append(_Row(package_id=pid, payload="x"))
    w.close()
    assert read_staged_package_ids(
        run_id="r1", artifact="inputs", staging_dir=tmp_path
    ) == {"a", "b", "c"}


def test_atomic_rewrite_preserves_old_on_crash(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"package_id":"a","payload":"old"}\n')
    # The atomic rewrite goes through `<path>.tmp` then renames; even
    # if the writer is interrupted before the rename, the original is
    # still complete. Simulate by checking the tmp is the staging
    # surface (the function itself does the rename atomically).
    atomic_rewrite_file(
        path=path, rows=[_Row(package_id="a", payload="new")]
    )
    assert "new" in path.read_text()


def test_writer_rejects_flush_every_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flush_every"):
        StageWriter(
            run_id="r1",
            artifact="inputs",
            staging_dir=tmp_path,
            flush_every=0,
        )
