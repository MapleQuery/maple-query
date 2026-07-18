"""Replay cache v2: digest values, put reason enum, only-answered
eligibility, TTL/LRU, snapshot invalidation, and replay emission."""
from __future__ import annotations

import time
from typing import Any

from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.memory import (
    SANITY_CEILING_BYTES,
    ReplayCacheV2,
    build_digest,
    replay,
)


def _turn_events(
    *, message: str = "the answer.", rows: int = 500
) -> list[agent_events.AgentEvent]:
    return [
        agent_events.TurnStart(
            conversation_id="c1", turn_id="t-orig", cached=False
        ),
        agent_events.RetrievalStarted(query="q", k=5),
        agent_events.DatasetsRanked(
            candidates=[
                {
                    "package_id": "pkg-1",
                    "title": "T",
                    "summary": "long prose " * 50,
                    "similarity": 0.9,
                    "distance": 0.1,
                }
            ],
            top_similarity=0.9,
        ),
        agent_events.SqlGenerated(sql="SELECT 1", rationale="r"),
        agent_events.SqlExecuted(
            row_count=rows,
            bytes_billed=10,
            elapsed_ms=5,
            sample_rows=[{"n": i} for i in range(10)],
        ),
        agent_events.Rows(
            sql_call_id="s1",
            rows=[{"n": i} for i in range(rows)],
            is_last=True,
        ),
        agent_events.MessageDelta(delta=message[: len(message) // 2]),
        agent_events.MessageDelta(delta=message[len(message) // 2 :]),
        agent_events.TurnRecordEvent(
            record={"turn_id": "t-orig", "outcome": "answered"}
        ),
        agent_events.Done(
            turn_id="t-orig",
            total_tool_calls=2,
            total_dollars=0.01,
            elapsed_ms=100,
        ),
    ]


def _cache() -> ReplayCacheV2:
    return ReplayCacheV2(max_entries=3, ttl_seconds=60)


def test_digest_drops_rows_and_prose_and_collapses_deltas() -> None:
    digest = build_digest(_turn_events())
    kinds = [d["type"] for d in digest]
    assert "rows" not in kinds
    assert "retrieval_started" not in kinds
    assert kinds.count("message_delta") == 1
    ranked = next(d for d in digest if d["type"] == "datasets_ranked")
    assert ranked["candidates"] == [
        {"package_id": "pkg-1", "title": "T", "similarity": 0.9}
    ]
    executed = next(d for d in digest if d["type"] == "sql_executed")
    assert len(executed["sample_rows"]) == 3
    assert kinds[-1] == "done"


def test_worst_case_digest_stays_under_ceiling() -> None:
    events = _turn_events(message="x" * 20_000, rows=5_000)
    digest = build_digest(events)
    assert sum(len(str(d)) for d in digest) < SANITY_CEILING_BYTES


def test_put_reason_enum() -> None:
    cache = _cache()
    events = _turn_events()
    assert (
        cache.put("k1", events=events, outcome="answered", snapshot_hash="s")
        == "stored"
    )
    assert (
        cache.put("k2", events=events[:-1], outcome="answered", snapshot_hash="s")
        == "not_terminal"
    )
    for bad_outcome in ("no_data", "clarified", "deflected", "error",
                        "answered_with_caveat"):
        assert (
            cache.put(
                "k3", events=events, outcome=bad_outcome, snapshot_hash="s"
            )
            == "not_answered"
        )
    assert len(cache) == 1


def test_get_roundtrip_ttl_and_lru() -> None:
    cache = ReplayCacheV2(max_entries=2, ttl_seconds=60)
    events = _turn_events()
    cache.put("k1", events=events, outcome="answered", snapshot_hash="s")
    entry = cache.get("k1")
    assert entry is not None
    assert entry.snapshot_hash == "s"
    assert cache.get("missing") is None
    # LRU: adding two more evicts the oldest untouched key.
    cache.put("k2", events=events, outcome="answered", snapshot_hash="s")
    cache.get("k1")  # touch
    cache.put("k3", events=events, outcome="answered", snapshot_hash="s")
    assert cache.get("k2") is None
    assert cache.get("k1") is not None
    # TTL: an expired entry is a miss.
    expired = ReplayCacheV2(max_entries=2, ttl_seconds=0)
    expired.put("k", events=events, outcome="answered", snapshot_hash="s")
    time.sleep(0.01)
    assert expired.get("k") is None


def test_invalidate_on_snapshot_drops_stale_only() -> None:
    cache = _cache()
    events = _turn_events()
    cache.put("old", events=events, outcome="answered", snapshot_hash="s1")
    cache.put("new", events=events, outcome="answered", snapshot_hash="s2")
    assert cache.invalidate_on_snapshot("s2") == 1
    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_replay_emits_fresh_ids_and_record() -> None:
    cache = _cache()
    cache.put(
        "k", events=_turn_events(), outcome="answered", snapshot_hash="s"
    )
    entry = cache.get("k")
    assert entry is not None
    replayed = list(
        replay(
            entry,
            turn_id="t-fresh",
            conversation_id="c1",
            key="k" * 20,
            delay_ms=0,
        )
    )
    kinds = [e.event_type for e in replayed]
    assert kinds[0] == "turn_start"
    assert kinds[1] == "cache_hit"
    assert kinds[-1] == "done"
    start = replayed[0]
    assert isinstance(start, agent_events.TurnStart)
    assert start.cached is True
    assert start.turn_id == "t-fresh"
    done = replayed[-1]
    assert isinstance(done, agent_events.Done)
    assert done.turn_id == "t-fresh"
    record_events = [
        e
        for e in replayed
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(record_events) == 1
    assert record_events[0].record["turn_id"] == "t-fresh"
    final: dict[str, Any] = {}
    for e in replayed:
        if isinstance(e, agent_events.MessageDelta):
            final["message"] = final.get("message", "") + e.delta
    assert final["message"] == "the answer."
