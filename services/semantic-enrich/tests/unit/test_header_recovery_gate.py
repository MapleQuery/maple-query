"""The go/no-go criteria, encoded as data so softening them shows up.

The temptation at the end of a piece of work like this is to pick a
recall target retroactively so it looks finished. These tests exist to
make that a visible diff rather than a quiet edit — particularly the
wrong-name bar, which is **zero and not a percentage**: one wrong name in
fifty implies ~2% of a 5k-document corpus carries a confidently wrong
column name, and nothing downstream would catch it.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.header_recovery import (
    HEADER_MIN_DENSITY,
    HEADER_SCAN_ROWS,
)
from semantic_enrich.core.header_recovery_report import GATE, build_report


def _report(**kwargs: Any) -> dict[str, Any]:
    return build_report(
        [],
        scan_rows=HEADER_SCAN_ROWS,
        min_density=HEADER_MIN_DENSITY,
        bytes_scanned=0,
        **kwargs,
    )


def test_the_wrong_name_bar_is_zero_and_not_a_percentage() -> None:
    assert GATE["wrong_names_max"] == 0
    assert isinstance(GATE["wrong_names_max"], int)


def test_the_recovery_rate_floor_is_a_measurement_threshold() -> None:
    """Recall has a floor for shipping, but it is the soft half of the
    gate — below it the decline reasons get read, they do not get
    ignored."""
    assert GATE["recovery_rate_min"] == 0.3


def test_a_human_reads_fifty_documents() -> None:
    assert GATE["review_sample_size"] == 50


def test_the_criteria_travel_inside_the_report() -> None:
    """In the file, not in someone's memory. Whoever revisits this needs
    the bar and the numbers in the same place."""
    report = _report()
    assert report["gate"] == GATE


def test_the_verdict_starts_pending_rather_than_passing() -> None:
    """Whether a recovered name is *wrong* is a reading task. The report
    records the bar and refuses to invent a pass for it."""
    result = _report()["gate_result"]
    assert result["wrong_names_observed"] is None
    assert result["wrong_names_verdict"] == "pending_human_review"
    assert result["decision"] == "no_go_until_reviewed"


def test_the_report_says_what_wrong_means() -> None:
    """Awkward is not wrong. `Total Amount ($000)` is an awkward
    identifier and a correct recovery, and a reviewer who does not know
    that will report false positives."""
    note = _report()["gate_result"]["note"]
    assert "not the header" in note
    assert "wrong positional key" in note
    assert "Awkward is not wrong" in note


def test_recovery_rate_is_scored_automatically() -> None:
    """The half a machine can decide, it decides."""
    result = _report()["gate_result"]
    assert result["recovery_rate_pass"] is False
    assert result["recovery_rate_observed"] == 0.0


def test_recovery_ships_off_so_the_report_is_what_turns_it_on() -> None:
    """Unlike every other gate in the codebase, this one is boolean
    rather than log/act. Computing names and not showing them changes
    nothing observable, so there is no useful in-turn shadow mode — the
    offline report is the shadow evidence, and it covers thousands of
    documents rather than whatever a live run happens to touch."""
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    assert settings.agent_header_recovery is False
