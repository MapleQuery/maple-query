"""Provenance capture: source packages/titles and the source-row
estimate that the magnitude floor gate keys on.
"""
from __future__ import annotations

from semantic_enrich.core.agent.derivation import build_derivation
from semantic_enrich.core.agent_tools import LoopState


def _state() -> LoopState:
    state = LoopState(conversation_id="c", turn_id="t", question="q")
    state.doc_package.update({"doc-a": "pkg-2024", "doc-b": "pkg-2025"})
    state.doc_title.update(
        {
            "doc-a": "Main Estimates 2024-25",
            "doc-b": "Main Estimates 2025-26",
        }
    )
    state.doc_row_count.update({"doc-a": 900, "doc-b": 512})
    return state


_TWO_DOC_SUM = (
    "SELECT SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
    "FROM raw.rows WHERE document_id IN ('doc-a', 'doc-b')"
)


def test_two_packages_are_both_captured() -> None:
    deriv = build_derivation(
        sql=_TWO_DOC_SUM,
        result={"row_count": 1, "rows": [{"total": 9.0e11}]},
        state=_state(),
    )[0]
    assert set(deriv.source_documents) == {"doc-a", "doc-b"}
    assert set(deriv.source_packages) == {"pkg-2024", "pkg-2025"}
    assert set(deriv.dataset_titles) == {
        "Main Estimates 2024-25",
        "Main Estimates 2025-26",
    }


def test_source_row_estimate_sums_source_docs() -> None:
    deriv = build_derivation(
        sql=_TWO_DOC_SUM,
        result={"row_count": 1, "rows": [{"total": 9.0e11}]},
        state=_state(),
    )[0]
    # This is the input-row proxy the absurd-floor gate uses; the
    # scalar aggregate's own row_count is 1.
    assert deriv.source_row_estimate == 1412
    assert deriv.row_count == 1


def test_missing_row_count_metadata_defaults_to_zero() -> None:
    state = LoopState(conversation_id="c", turn_id="t", question="q")
    state.doc_package["doc-a"] = "pkg-2024"
    # no doc_row_count entry
    deriv = build_derivation(
        sql="SELECT SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS t "
        "FROM raw.rows WHERE document_id IN ('doc-a')",
        result={"row_count": 1, "rows": [{"t": 5.0}]},
        state=state,
    )[0]
    assert deriv.source_row_estimate == 0
    assert deriv.source_packages == ("pkg-2024",)
