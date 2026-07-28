"""Recovered names reaching `LoopState`, for the SQL path to read.

Same side-effect pattern `doc_columns` / `doc_package` / `doc_title`
already use: the tool that fetched the data writes it down, and whatever
needs it later reads it without paying for a second query.
"""
from __future__ import annotations

from typing import Any

from tests.unit.test_header_recovery_payload import (
    CLEAN_ROWS,
    DECLINING,
    RECOVERABLE,
    _bq_for,
    _fixture,
    _list,
)


def _state(prefix: str, *, recovery: bool = True) -> Any:
    doc = _fixture(prefix)
    return _list(_bq_for([("doc-1", doc["rows"])]), recovery=recovery)["_state"]


def test_recovered_names_land_on_state_for_the_right_document() -> None:
    state = _state(RECOVERABLE)
    assert state.doc_recovered_names == {
        "doc-1": {
            "__col_1": "Number of Unique Applicants",
            "__col_2": "Total Amount ($000)",
        }
    }
    assert state.doc_header_row == {"doc-1": 1}


def test_state_still_carries_the_positional_keys() -> None:
    """`doc_columns` is what the pairing check validates against, and it
    keeps holding what the stored row holds."""
    state = _state(RECOVERABLE)
    assert "__col_1" in state.doc_columns["doc-1"]
    assert "Total Amount ($000)" not in state.doc_columns["doc-1"]


def test_a_declined_document_writes_nothing() -> None:
    state = _state(DECLINING)
    assert state.doc_recovered_names == {}
    assert state.doc_header_row == {}


def test_a_clean_document_writes_nothing() -> None:
    state = _list(_bq_for([("doc-1", CLEAN_ROWS)]), recovery=True)["_state"]
    assert state.doc_recovered_names == {}
    assert state.doc_header_row == {}


def test_kill_switch_leaves_state_empty() -> None:
    """With recovery off there are no recovered names anywhere, which is
    what makes the SQL translation a no-op by construction rather than
    by a second flag."""
    state = _state(RECOVERABLE, recovery=False)
    assert state.doc_recovered_names == {}
    assert state.doc_header_row == {}


def test_only_the_recovered_document_is_recorded() -> None:
    bq = _bq_for(
        [
            ("doc-good", _fixture(RECOVERABLE)["rows"]),
            ("doc-bad", _fixture(DECLINING)["rows"]),
            ("doc-clean", CLEAN_ROWS),
        ]
    )
    state = _list(bq, recovery=True)["_state"]
    assert set(state.doc_recovered_names) == {"doc-good"}
    assert set(state.doc_header_row) == {"doc-good"}
    # Every listed doc is still known to the pairing check.
    assert {"doc-good", "doc-bad", "doc-clean"} <= state.known_document_ids
