"""Sample selector — pure functions, property-tested."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from semantic_enrich.core.sample_selector import (
    derive_indices,
    pick_representative,
    truncate_cell,
)
from semantic_enrich.types import PackageResource


def _res(doc_id: str, row_count: int | None) -> PackageResource:
    return PackageResource(
        document_id=doc_id,
        title=None,
        description=None,
        subjects=(),
        organization_code="org",
        file_format="csv",
        resource_last_modified=None,
        row_count=row_count,
    )


def test_pick_representative_median() -> None:
    resources = [
        _res("a", 100),
        _res("b", 5000),
        _res("c", 200),
        _res("d", 1_000_000),
    ]
    rep = pick_representative(resources)
    # Sorted by row_count: [100, 200, 5000, 1_000_000]; median index = 2 → 5000.
    assert rep.document_id == "b"
    assert rep.row_count == 5000


def test_pick_representative_lex_tiebreak() -> None:
    resources = [_res("zebra", 100), _res("alpha", 100), _res("bravo", 100)]
    rep = pick_representative(resources)
    # All have row_count=100; sorted by document_id: alpha, bravo, zebra;
    # median index 1 → bravo.
    assert rep.document_id == "bravo"


def test_pick_representative_one_resource() -> None:
    resources = [_res("only", 42)]
    assert pick_representative(resources).document_id == "only"


def test_pick_representative_empty_raises() -> None:
    with pytest.raises(ValueError):
        pick_representative([])


def test_derive_indices_deterministic() -> None:
    a = derive_indices(document_id="doc-x", row_count=1_000, k=10)
    b = derive_indices(document_id="doc-x", row_count=1_000, k=10)
    assert a == b


def test_derive_indices_n_below_k() -> None:
    out = derive_indices(document_id="doc-x", row_count=3, k=10)
    assert out == [0, 1, 2]


def test_derive_indices_rejects_zero_rows() -> None:
    with pytest.raises(ValueError):
        derive_indices(document_id="doc-x", row_count=0, k=10)


def test_truncate_cell_short() -> None:
    assert truncate_cell("hi") == "hi"


def test_truncate_cell_long() -> None:
    out = truncate_cell("x" * 300)
    assert out is not None
    assert len(out) == 200
    assert out.endswith("…")


def test_truncate_cell_none() -> None:
    assert truncate_cell(None) is None


# ── Property tests ──


@given(
    document_id=st.text(min_size=1, max_size=50),
    row_count=st.integers(min_value=1, max_value=10_000),
    k=st.integers(min_value=1, max_value=100),
)
def test_derive_indices_properties(
    document_id: str, row_count: int, k: int
) -> None:
    out = derive_indices(document_id=document_id, row_count=row_count, k=k)
    assert out == sorted(out)
    assert len(out) == len(set(out))
    assert len(out) == min(k, row_count)
    assert all(0 <= i < row_count for i in out)
    # Determinism: re-running with the same inputs yields the same.
    assert (
        derive_indices(document_id=document_id, row_count=row_count, k=k) == out
    )
