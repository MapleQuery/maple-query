"""The measurements behind a detection, and the invariants that hold
whatever the input.

Signals exist so a recovery-rate report can say *why* a document was
accepted or declined without re-running the detector, and so moving a
threshold is an argument about a distribution rather than about
intuition. They are kept on declines for exactly that reason: the
near-misses are what a threshold change would move.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_enrich.core.header_recovery import (
    GENERATED_COL_RE,
    detect_header,
    explain_header,
    generated_header_ratio,
)

_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "header_recovery_documents.json"
)
DOCUMENTS: list[dict[str, Any]] = json.loads(_FIXTURE.read_text(encoding="utf-8"))

_SIGNAL_NAMES = {
    "positional",
    "all_text",
    "density",
    "distinctness",
    "contrast",
}


def _ids() -> list[str]:
    return [doc["document_id"][:10] for doc in DOCUMENTS]


@pytest.mark.parametrize("doc", DOCUMENTS, ids=_ids())
def test_every_real_document_reports_a_reason(doc: dict[str, Any]) -> None:
    report = explain_header(doc["rows"], doc["generated_columns"])
    assert report.reason
    if report.recovery is not None:
        assert report.reason in {"accepted", "accepted_composed"}


@pytest.mark.parametrize("doc", DOCUMENTS, ids=_ids())
def test_signals_are_populated_whenever_a_candidate_was_scored(
    doc: dict[str, Any],
) -> None:
    """A document only lacks signals when there was no candidate row to
    measure — no rows above the data at all, or nothing to name."""
    report = explain_header(doc["rows"], doc["generated_columns"])
    if report.candidate_row_index is None:
        assert report.signals == {}
        return
    assert set(report.signals) == _SIGNAL_NAMES
    assert all(0.0 <= value <= 1.0 for value in report.signals.values())


def test_accepted_signals_clear_every_gate() -> None:
    """The gate is conjunctive: a weighted score would let a very dense
    preamble outvote a failed contrast test, so each signal keeps a veto."""
    accepted = [
        explain_header(doc["rows"], doc["generated_columns"])
        for doc in DOCUMENTS
    ]
    accepted = [r for r in accepted if r.recovery is not None]
    assert accepted
    for report in accepted:
        assert report.signals["positional"] == 1.0
        assert report.signals["all_text"] == 1.0
        assert report.signals["density"] >= 0.6
        assert report.signals["distinctness"] == 1.0
        assert report.signals["contrast"] > 0.0
        assert report.recovery is not None
        assert report.recovery.signals == report.signals


def test_declined_near_miss_keeps_its_measurements() -> None:
    """The legal aid table's header tiers each fill a third of the columns.
    Density is what declines it, and the report says so with the number."""
    doc = next(d for d in DOCUMENTS if d["document_id"].startswith("57f2ef5417"))
    report = explain_header(doc["rows"], doc["generated_columns"])
    assert report.recovery is None
    # Three tiers: past what composition will read, and no single row
    # names the columns.
    assert report.reason == "tier_split"
    assert report.signals["all_text"] == 1.0


def test_reason_is_stable_vocabulary() -> None:
    """Reasons are counted by the rollout report, so they are a closed set
    rather than free text."""
    known = {
        "accepted",
        "accepted_composed",
        "no_rows",
        "no_generated_columns",
        "no_data_rows_in_window",
        "data_starts_at_row_0",
        "multi_tier",
        "tier_split",
        "no_generated_names",
        "positional",
        "density",
        "all_text",
        "distinctness",
        "contrast",
    }
    seen = {
        explain_header(doc["rows"], doc["generated_columns"]).reason
        for doc in DOCUMENTS
    }
    assert seen <= known
    assert len(seen) >= 4


def test_generated_header_ratio_matches_the_positional_key_shape() -> None:
    assert generated_header_ratio([]) == 0.0
    assert generated_header_ratio(["__col_1", "__col_2"]) == 1.0
    assert generated_header_ratio(["__col_1x"]) == 0.0
    assert generated_header_ratio(("__col_1", "Region")) == 0.5
    assert GENERATED_COL_RE.fullmatch("__col_12") is not None
    assert GENERATED_COL_RE.fullmatch("__col_") is None


# ── invariants that hold for any input ──

_BLANK = st.sampled_from([None, "", " ", "\n"])
_LABEL = st.sampled_from(
    [
        "Region",
        "Total",
        "Total Amount ($000)",
        "Under 25",
        "2019",
        "2020",
        "Ville\nCity",
        "n.a.",
        "__col_1",
    ]
)
_VALUE = st.sampled_from(["2,550", "-3.5", "$1,000", "2015-09-04", "0", "17"])
_ANY = st.one_of(_BLANK, _LABEL, _VALUE)

# Rows are drawn from the shapes real spreadsheets stack: blank spacers,
# sparse preamble banners, dense label rows, and data. Drawing cells
# uniformly instead would almost never assemble a table with a header in
# it, and the invariants below would pass without ever being tested.
_ROW_SHAPES = st.sampled_from(["blank", "sparse", "label", "data", "mixed"])


@st.composite
def _tables(draw: st.DrawFn) -> tuple[list[dict[str, object]], list[str]]:
    width = draw(st.integers(min_value=1, max_value=5))
    height = draw(st.integers(min_value=0, max_value=6))
    named = draw(st.integers(min_value=0, max_value=width))
    keys = [f"name_{i}" for i in range(named)] + [
        f"__col_{i}" for i in range(named, width)
    ]
    rows: list[dict[str, object]] = []
    for _ in range(height):
        shape = draw(_ROW_SHAPES)
        row: dict[str, object] = {}
        for position, key in enumerate(keys):
            if shape == "blank":
                row[key] = draw(_BLANK)
            elif shape == "sparse":
                row[key] = draw(_LABEL) if position == 0 else draw(_BLANK)
            elif shape == "label":
                row[key] = draw(_LABEL)
            elif shape == "data":
                row[key] = draw(_VALUE)
            else:
                row[key] = draw(_ANY)
        rows.append(row)
    generated = [k for k in keys if GENERATED_COL_RE.fullmatch(k)]
    return rows, generated


@settings(max_examples=400)
@given(_tables())
def test_names_are_always_drawn_from_one_row_and_never_synthesised(
    table: tuple[list[dict[str, object]], list[str]],
) -> None:
    rows, generated = table
    recovery = detect_header(rows, generated)
    if recovery is None:
        return
    header = rows[recovery.header_row_index]
    for column, name in recovery.names.items():
        # Only ever a generated column, only ever from the header row,
        # only ever that cell's own text.
        assert column in generated
        assert name == " ".join(str(header[column]).split())
        assert name
        assert GENERATED_COL_RE.fullmatch(name) is None
    assert len(set(recovery.names.values())) == len(recovery.names)
    assert recovery.preamble_rows == recovery.header_row_index
    assert 0 <= recovery.header_row_index < len(rows)


@settings(max_examples=400)
@given(_tables())
def test_detect_and_explain_never_disagree(
    table: tuple[list[dict[str, object]], list[str]],
) -> None:
    rows, generated = table
    assert detect_header(rows, generated) == explain_header(rows, generated).recovery


@settings(max_examples=200)
@given(_tables())
def test_detection_is_deterministic(
    table: tuple[list[dict[str, object]], list[str]],
) -> None:
    """Column names have to be reproducible: the same rows must always
    produce the same names, or the corpus's vocabulary drifts under it."""
    rows, generated = table
    assert detect_header(rows, generated) == detect_header(rows, generated)


def test_settings_defaults_match_the_detectors_own() -> None:
    """The tunables exist in two places — here, where the algorithm is,
    and in `Settings`, where the tool path can move them without a
    release. Config sits below core in the layering, so it cannot import
    these constants; this asserts they have not drifted apart instead."""
    from semantic_enrich.config.settings import Settings
    from semantic_enrich.core.header_recovery import (
        HEADER_MIN_DENSITY,
        HEADER_SCAN_ROWS,
    )

    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    assert settings.agent_header_scan_rows == HEADER_SCAN_ROWS
    assert settings.agent_header_min_density == HEADER_MIN_DENSITY
    # On, on the evidence of the offline recovery report rather than a
    # live run. See test_header_recovery_gate.py for the reasoning.
    assert settings.agent_header_recovery is True
