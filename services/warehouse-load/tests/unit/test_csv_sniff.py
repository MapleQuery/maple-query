"""Sniff: delimiter + encoding."""
from __future__ import annotations

from warehouse_load.core.csv_sniff import sniff_csv


def test_comma_delimited_utf8() -> None:
    head = b"col1,col2,col3\n1,2,3\n"
    result = sniff_csv(head)
    assert result.delimiter == ","
    assert result.encoding == "utf-8"


def test_tab_delimited_utf8() -> None:
    head = b"col1\tcol2\tcol3\n1\t2\t3\n"
    result = sniff_csv(head)
    assert result.delimiter == "\t"
    assert result.encoding == "utf-8"


def test_tab_wins_when_more_tabs_than_commas() -> None:
    # One comma in a quoted cell, three tabs as separators — tabs win.
    head = b"col1\tcol2\tcol,3\tcol4\nx\ty\tz\tw\n"
    assert sniff_csv(head).delimiter == "\t"


def test_single_column_defaults_to_comma() -> None:
    head = b"only_one\nvalue\nvalue2\n"
    assert sniff_csv(head).delimiter == ","


def test_comma_wins_on_tie() -> None:
    # Same count of each → comma wins per §5.1.
    head = b"a,b\tc,d\te\n"
    assert sniff_csv(head).delimiter == ","


def test_utf8_bom_detected() -> None:
    head = b"\xef\xbb\xbfcol1,col2\n1,2\n"
    result = sniff_csv(head)
    assert result.encoding == "utf-8-sig"


def test_latin1_fallback_when_utf8_fails() -> None:
    # 0xE9 is `é` in latin-1 but invalid utf-8 (continuation byte).
    head = b"r\xe9sum\xe9,col2\n1,2\n"
    result = sniff_csv(head)
    assert result.encoding != "utf-8"  # charset_normalizer picks something


def test_sniff_bytes_reflects_input_length() -> None:
    head = b"a,b\n1,2\n"
    assert sniff_csv(head).sniff_bytes == len(head)
