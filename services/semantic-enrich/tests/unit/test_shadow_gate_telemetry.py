"""The shadow gates' evidence actually reaches somewhere durable.

Two gates have been sitting in `log` collecting evidence for an act-flip.
Neither was reaching a place the evidence survives: the descriptive
rubric's verdict was assembled into the turn observer and then dropped
before the span was written, and the deterministic numeric gate was
skipped by the observer entirely — it makes no model call, so
`wrap_openai` never saw it either. That left Cloud Logging as the only
record, on a 30-day roll-off that expires faster than the evidence
accrues.

Telemetry fails silently by nature: nothing breaks, a number just stops
arriving, and the gap is invisible until someone goes looking months
later. So these assert the wiring rather than trusting it.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_tracing import TurnObserver


def _observe(*events: agent_events.AgentEvent) -> TurnObserver:
    observer = TurnObserver()
    for event in events:
        observer.observe(event)
    return observer


def _verification(**overrides: Any) -> agent_events.Verification:
    base: dict[str, Any] = {
        "fits": True,
        "action": "answer",
        "confidence": 0.9,
        "reason": "",
        "enforced": False,
    }
    base.update(overrides)
    return agent_events.Verification(**base)


def test_explore_verdict_reaches_the_span_metadata() -> None:
    """It was collected onto the observer and never written out."""
    observer = _observe(
        _verification(kind="explore", fits=False, action="caveat", reason="gap")
    )
    meta = observer.metadata()
    assert meta["verify_explore"] == {
        "fits": False,
        "action": "caveat",
        "confidence": 0.9,
        "reason": "gap",
        "enforced": False,
    }


def test_magnitude_findings_reach_the_span_metadata() -> None:
    """The gate with no other durable record at all."""
    observer = _observe(
        _verification(
            kind="magnitude",
            fits=False,
            action="caveat",
            confidence=1.0,
            reason="absurd_floor: $8 from 1400 rows",
        )
    )
    assert observer.metadata()["magnitude"] == [
        {
            "action": "caveat",
            "reason": "absurd_floor: $8 from 1400 rows",
            "enforced": False,
        }
    ]


def test_several_magnitude_findings_are_all_kept() -> None:
    """A turn can produce more than one, and each is evidence."""
    observer = _observe(
        _verification(kind="magnitude", fits=False, reason="a"),
        _verification(kind="magnitude", fits=False, reason="b"),
    )
    assert [f["reason"] for f in observer.metadata()["magnitude"]] == ["a", "b"]


def test_magnitude_never_seeds_the_fit_metric() -> None:
    """The fits-rate is about the LLM fit checker. Seeding it with a
    deterministic bounds verdict would make the two act-flip gates share
    one meaningless number — the exact pollution the magnitude PR called
    out when it introduced the `kind` discriminator."""
    observer = _observe(_verification(kind="magnitude", fits=False))
    assert observer.verify is None
    assert "verify" not in observer.metadata()


def test_explore_never_seeds_the_fit_metric() -> None:
    observer = _observe(_verification(kind="explore", fits=False))
    assert observer.verify is None


def test_a_fit_verdict_still_seeds_only_the_fit_metric() -> None:
    observer = _observe(_verification(kind="fit", fits=True))
    meta = observer.metadata()
    assert meta["verify"]["fits_first"] is True
    assert "verify_explore" not in meta
    assert "magnitude" not in meta


def test_a_clean_turn_carries_neither_key() -> None:
    """Absent, not null: an empty key on every turn would make the
    shadow-evidence count unreadable in the trace explorer."""
    start = agent_events.TurnStart(
        turn_id="t1", conversation_id="c1", cached=False
    )
    meta = _observe(start).metadata()
    assert "verify_explore" not in meta
    assert "magnitude" not in meta


def test_all_three_verdict_kinds_coexist_on_one_turn() -> None:
    """A numeric turn can produce all three, and they must not overwrite
    each other on the way out."""
    observer = _observe(
        _verification(kind="magnitude", fits=False, reason="unknown_units"),
        _verification(kind="fit", fits=True),
        _verification(kind="explore", fits=True),
    )
    meta = observer.metadata()
    assert meta["magnitude"][0]["reason"] == "unknown_units"
    assert meta["verify"]["fits_first"] is True
    assert meta["verify_explore"]["fits"] is True
