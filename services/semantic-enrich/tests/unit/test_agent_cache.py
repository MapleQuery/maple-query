"""LRU, TTL, and snapshot invalidation invariants for the response cache."""
from __future__ import annotations

import time

from semantic_enrich.core.agent_cache import (
    ResponseCache,
    cache_key,
    normalize_question,
)
from semantic_enrich.core.agent_events import Done, MessageDelta


def _events() -> list:
    return [
        MessageDelta(delta="hi"),
        Done(turn_id="t1", total_tool_calls=1, total_dollars=0.001, elapsed_ms=100),
    ]


def test_normalize_question_collapses_whitespace() -> None:
    assert normalize_question("  Hello   World  ") == "hello world"


def test_cache_key_stable_across_normalization() -> None:
    a = cache_key(question="Hello  World", prompt_hash="p", snapshot_hash="s")
    b = cache_key(question="hello world", prompt_hash="p", snapshot_hash="s")
    assert a == b


def test_cache_put_and_get() -> None:
    cache = ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60)
    key = "k1"
    assert cache.put(
        key, events=_events(), prompt_hash="p", snapshot_hash="s"
    )
    entry = cache.get(key)
    assert entry is not None
    assert entry.events[0]["type"] == "message_delta"


def test_lru_eviction_on_insert() -> None:
    cache = ResponseCache(max_entries=2, max_value_bytes=1_000_000, ttl_seconds=60)
    cache.put("k1", events=_events(), prompt_hash="p", snapshot_hash="s")
    cache.put("k2", events=_events(), prompt_hash="p", snapshot_hash="s")
    cache.put("k3", events=_events(), prompt_hash="p", snapshot_hash="s")
    # k1 evicted, k2/k3 remain.
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    assert cache.get("k3") is not None


def test_ttl_expiry(monkeypatch) -> None:
    cache = ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=1)
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cache.put("k1", events=_events(), prompt_hash="p", snapshot_hash="s")
    now[0] += 2
    assert cache.get("k1") is None


def test_value_size_cap_skips_cache() -> None:
    cache = ResponseCache(max_entries=10, max_value_bytes=10, ttl_seconds=60)
    ok = cache.put("k1", events=_events(), prompt_hash="p", snapshot_hash="s")
    assert ok is False
    assert cache.get("k1") is None


def test_snapshot_invalidation_drops_stale_entries() -> None:
    cache = ResponseCache(max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60)
    cache.put("k1", events=_events(), prompt_hash="p", snapshot_hash="s_old")
    cache.put("k2", events=_events(), prompt_hash="p", snapshot_hash="s_new")
    dropped = cache.invalidate_on_snapshot("s_new")
    assert dropped == 1
    assert cache.get("k1") is None
    assert cache.get("k2") is not None


def test_lru_promotes_on_get() -> None:
    cache = ResponseCache(max_entries=2, max_value_bytes=1_000_000, ttl_seconds=60)
    cache.put("k1", events=_events(), prompt_hash="p", snapshot_hash="s")
    cache.put("k2", events=_events(), prompt_hash="p", snapshot_hash="s")
    cache.get("k1")
    # k1 is now MRU; k2 evicts on next insert.
    cache.put("k3", events=_events(), prompt_hash="p", snapshot_hash="s")
    assert cache.get("k2") is None
    assert cache.get("k1") is not None
    assert cache.get("k3") is not None
