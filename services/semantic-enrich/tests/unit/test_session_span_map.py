"""`SessionSpanMap` — one session root per conversation, LRU + TTL."""
from __future__ import annotations

import threading
import time

from semantic_enrich.core.agent_tracing import SessionSpanMap
from tests.unit.conftest import FakeBraintrustModule


def _map(**overrides: object) -> SessionSpanMap:
    kwargs: dict = {"max_entries": 10, "ttl_seconds": 60, "enabled": True}
    kwargs.update(overrides)
    return SessionSpanMap(**kwargs)


def test_same_conversation_reuses_exported_parent(
    fake_braintrust: FakeBraintrustModule,
) -> None:
    session_map = _map()
    first = session_map.get_or_create("conv-1")
    second = session_map.get_or_create("conv-1")

    assert first is not None
    assert first == second
    assert len(fake_braintrust.spans) == 1
    span = fake_braintrust.spans[0]
    assert span.name == "session"
    assert span.kwargs["input"] == {"conversation_id": "conv-1"}
    # The session root closes immediately; children attach through the
    # export string afterwards.
    assert span.ended is True


def test_disabled_map_returns_none(
    fake_braintrust: FakeBraintrustModule,
) -> None:
    session_map = _map(enabled=False)
    assert session_map.get_or_create("conv-1") is None
    assert fake_braintrust.spans == []


def test_unconfigured_tracing_returns_none(tracing_disabled) -> None:
    assert _map().get_or_create("conv-1") is None


def test_ttl_eviction_creates_new_session_root(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    session_map = _map(ttl_seconds=30)
    first = session_map.get_or_create("conv-1")
    clock["now"] += 31
    second = session_map.get_or_create("conv-1")

    assert first != second
    assert len(fake_braintrust.spans) == 2


def test_lru_eviction_bounds_the_map(
    fake_braintrust: FakeBraintrustModule,
) -> None:
    session_map = _map(max_entries=1)
    session_map.get_or_create("conv-1")
    session_map.get_or_create("conv-2")  # evicts conv-1
    session_map.get_or_create("conv-1")  # re-created

    assert len(fake_braintrust.spans) == 3


def test_concurrent_requests_create_exactly_one_session_span(
    fake_braintrust: FakeBraintrustModule,
) -> None:
    session_map = _map()
    barrier = threading.Barrier(8)
    exports: list[str | None] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        export = session_map.get_or_create("conv-1")
        with lock:
            exports.append(export)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(fake_braintrust.spans) == 1
    assert len(set(exports)) == 1
    assert exports[0] is not None
