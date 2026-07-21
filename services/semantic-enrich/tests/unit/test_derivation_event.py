"""The additive DerivationEvent: registration, round-trip, replay."""
from __future__ import annotations

from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_events import DerivationEvent


def _event(**overrides: object) -> DerivationEvent:
    base: dict[str, object] = {
        "dataset_titles": ["Main Estimates 2024-25"],
        "source_packages": ["pkg-1"],
        "aggregation": "SUM",
        "value_columns": ["Amount"],
        "scope": "WHERE document_id IN ('')",
        "row_count": 1,
        "source_row_estimate": 1400,
        "result_value": 900.84e9,
        "result_label": "total",
        "unit_scale": "dollars",
        "unit_source": "column_description",
        "flags": ["cross_source_sum"],
    }
    base.update(overrides)
    return DerivationEvent(**base)  # type: ignore[arg-type]


def test_registered_in_all_touch_points() -> None:
    assert agent_events._EVENT_CLASSES["derivation"] is DerivationEvent
    assert "derivation" in agent_events.EventType.__args__  # type: ignore[attr-defined]
    assert _event().event_type == "derivation"


def test_sse_round_trip() -> None:
    original = _event()
    frame = original.to_sse_frame()
    assert "event: derivation" in frame
    restored = agent_events.from_sse_frame(frame)
    assert restored == original


def test_dict_to_frame_replay() -> None:
    # The cache-replay path reconstructs events from stored dicts.
    payload = _event().to_dict()
    from semantic_enrich.core.agent.phases import _dict_to_frame

    restored = _dict_to_frame(payload)
    assert isinstance(restored, DerivationEvent)
    assert restored.flags == ["cross_source_sum"]


def test_null_result_value_serializes() -> None:
    ev = _event(result_value=None, result_label=None)
    restored = agent_events.from_sse_frame(ev.to_sse_frame())
    assert restored == ev
