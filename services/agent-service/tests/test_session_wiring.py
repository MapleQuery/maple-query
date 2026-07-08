"""Session-span map wiring on AppState.

Span behaviour itself is unit-tested in semantic-enrich
(test_session_span_map.py); here we only assert the service wiring:
the map exists on AppState and is inert when tracing is unconfigured,
so `/chat` never pays for tracing that isn't on."""
from __future__ import annotations

from semantic_enrich.core.agent_tracing import SessionSpanMap

from agent_service.deps import AppState


def test_app_state_has_inert_session_map_by_default(app_state: AppState) -> None:
    assert isinstance(app_state.session_spans, SessionSpanMap)
    # Tracing is unconfigured in tests → no parent, no span creation.
    assert app_state.session_spans.get_or_create("conv-1") is None
