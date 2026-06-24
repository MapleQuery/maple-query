"""Header-name normalisation."""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from warehouse_load.core.header_normalise import normalise_keys


def test_whitespace_collapsed() -> None:
    assert normalise_keys(["  hello  world  "]) == ["hello_world"]


def test_case_preserved() -> None:
    assert normalise_keys(["EN_YEAR", "FR_ANNEE", "Year"]) == [
        "EN_YEAR", "FR_ANNEE", "Year",
    ]


def test_empty_becomes_indexed_synthetic() -> None:
    assert normalise_keys(["a", "", "c"]) == ["a", "__col_1", "c"]


def test_duplicate_suffix_in_order_of_appearance() -> None:
    assert normalise_keys(["x", "x", "x"]) == ["x", "x__2", "x__3"]


def test_tab_newline_treated_as_whitespace() -> None:
    assert normalise_keys(["foo\tbar\nbaz"]) == ["foo_bar_baz"]


def test_blank_after_strip_falls_back() -> None:
    assert normalise_keys(["   "]) == ["__col_0"]


def test_synthetic_indices_match_input_position() -> None:
    out = normalise_keys(["", "", ""])
    assert out == ["__col_0", "__col_1", "__col_2"]


# ---------- property: deterministic, length-preserving, no empty keys ----------


@given(st.lists(st.text(), min_size=0, max_size=20))
def test_property_length_preserved(raw: list[str]) -> None:
    out = normalise_keys(raw)
    assert len(out) == len(raw)


@given(st.lists(st.text(), min_size=0, max_size=20))
def test_property_never_empty(raw: list[str]) -> None:
    out = normalise_keys(raw)
    assert all(key for key in out)


@given(st.lists(st.text(), min_size=0, max_size=20))
def test_property_keys_unique(raw: list[str]) -> None:
    out = normalise_keys(raw)
    assert len(set(out)) == len(out)
