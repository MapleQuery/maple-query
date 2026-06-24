"""Body row streaming + cell normalisation."""
from __future__ import annotations

from pathlib import Path

from warehouse_load.core.row_stream import (
    iter_lookahead_rows,
    needs_utf8_conversion,
    prepare_utf8_copy,
    stream_body_rows,
)
from warehouse_load.types import HeaderResult, SniffResult


def _write_csv(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(text, encoding="utf-8")
    return p


def _sniff(encoding: str = "utf-8") -> SniffResult:
    return SniffResult(delimiter=",", encoding=encoding, sniff_bytes=0)


def test_empty_string_becomes_null(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, "a,b,c\n1,,3\n")
    header = HeaderResult(
        body_start_index=1,
        header_rows=(("a", "b", "c"),),
        preamble_rows=(),
        confidence="single",
        keys=("a", "b", "c"),
    )
    rows = list(stream_body_rows(path=csv, sniff=_sniff(), header=header))
    assert len(rows) == 1
    assert rows[0].row == {"a": "1", "b": None, "c": "3"}


def test_sentinel_values_kept_verbatim(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, "a,b\nn.a.,N/A\n<MDL,NULL\n")
    header = HeaderResult(
        body_start_index=1,
        header_rows=(("a", "b"),),
        preamble_rows=(),
        confidence="single",
        keys=("a", "b"),
    )
    rows = list(stream_body_rows(path=csv, sniff=_sniff(), header=header))
    assert rows[0].row == {"a": "n.a.", "b": "N/A"}
    assert rows[1].row == {"a": "<MDL", "b": "NULL"}


def test_ragged_short_rows_pad_with_none(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, "a,b,c\n1\n2,3\n")
    header = HeaderResult(
        body_start_index=1,
        header_rows=(("a", "b", "c"),),
        preamble_rows=(),
        confidence="single",
        keys=("a", "b", "c"),
    )
    rows = list(stream_body_rows(path=csv, sniff=_sniff(), header=header))
    assert rows[0].row == {"a": "1", "b": None, "c": None}
    assert rows[1].row == {"a": "2", "b": "3", "c": None}


def test_nul_bytes_stripped_and_flagged(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_bytes(b"a,b\nfoo\x00bar,baz\n")
    header = HeaderResult(
        body_start_index=1,
        header_rows=(("a", "b"),),
        preamble_rows=(),
        confidence="single",
        keys=("a", "b"),
    )
    rows = list(stream_body_rows(path=csv_path, sniff=_sniff(), header=header))
    assert rows[0].row == {"a": "foobar", "b": "baz"}
    assert rows[0].nul_stripped is True


def test_lookahead_returns_first_n_rows(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, "a,b\n1,2\n3,4\n5,6\n7,8\n")
    rows = list(iter_lookahead_rows(path=csv, sniff=_sniff(), max_rows=3))
    assert len(rows) == 3
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", "2"]


def test_needs_utf8_conversion_only_for_non_utf8() -> None:
    assert needs_utf8_conversion("utf-8") is False
    assert needs_utf8_conversion("UTF-8") is False
    assert needs_utf8_conversion("utf8") is False
    assert needs_utf8_conversion("utf-8-sig") is True
    assert needs_utf8_conversion("latin-1") is True
    assert needs_utf8_conversion("cp1252") is True


def test_prepare_utf8_copy_strips_bom(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_bytes(b"\xef\xbb\xbfcol,val\n1,2\n")
    dst = tmp_path / "dst.csv"
    prepare_utf8_copy(source_path=src, encoding="utf-8-sig", dest_path=dst)
    assert dst.read_bytes() == b"col,val\n1,2\n"


def test_prepare_utf8_copy_decodes_latin1(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_bytes(b"r\xe9sum\xe9,col\n1,2\n")
    dst = tmp_path / "dst.csv"
    prepare_utf8_copy(source_path=src, encoding="latin-1", dest_path=dst)
    decoded = dst.read_text(encoding="utf-8")
    assert decoded.startswith("résumé,col\n")
