"""In-memory LRU response cache for the agent loop.

Per-instance. Cold-start empty; not shared across Cloud Run instances.

Key = sha256(question_normalized || prompt_hash || snapshot_hash).
Value = the recorded event stream (typed events serialised to dicts).

Cache hits replay events one-by-one with a small delay so the FE
renders progressively rather than teleporting. History is
intentionally NOT part of the key — follow-up turns re-run.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from semantic_enrich.core import agent_events

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Lower-case, collapse whitespace, strip. The cache treats trivial
    formatting differences as the same question."""
    return _WHITESPACE_RE.sub(" ", question.strip().lower())


def cache_key(
    *,
    question: str,
    prompt_hash: str,
    snapshot_hash: str,
) -> str:
    payload = "||".join([normalize_question(question), prompt_hash, snapshot_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    """One recorded turn.

    Events are stored as dicts (JSON-serializable) rather than typed
    dataclasses so a future evolution of the event schema doesn't
    invalidate all in-flight entries."""

    events: list[dict[str, Any]]
    created_at: float
    size_bytes: int
    prompt_hash: str
    snapshot_hash: str


@dataclass
class ResponseCache:
    """Thread-safe LRU with TTL + value-size cap.

    Only synchronous callers — the loop runs on one worker thread per
    request in the CLI, and the future HTTP surface will front the
    cache with a per-request handler. The lock is a cheap defence in
    depth."""

    max_entries: int
    max_value_bytes: int
    ttl_seconds: int
    _entries: OrderedDict[str, CacheEntry] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> CacheEntry | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if now - entry.created_at > self.ttl_seconds:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return entry

    def put(
        self,
        key: str,
        *,
        events: list[agent_events.AgentEvent],
        prompt_hash: str,
        snapshot_hash: str,
    ) -> bool:
        """Insert a fresh entry. Returns False if the value exceeds
        `max_value_bytes` (skip-cache posture) so the caller knows the
        turn was not memoised."""
        payloads = [e.to_dict() for e in events]
        serialised = _rough_size(payloads)
        if serialised > self.max_value_bytes:
            return False
        entry = CacheEntry(
            events=payloads,
            created_at=time.monotonic(),
            size_bytes=serialised,
            prompt_hash=prompt_hash,
            snapshot_hash=snapshot_hash,
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return True

    def invalidate_on_snapshot(self, current_snapshot_hash: str) -> int:
        """Drop entries whose recorded `snapshot_hash` differs. Called
        when the snapshot-hash refresh detects a warehouse change.
        Returns the number of entries dropped."""
        dropped = 0
        with self._lock:
            for key in list(self._entries):
                if self._entries[key].snapshot_hash != current_snapshot_hash:
                    self._entries.pop(key, None)
                    dropped += 1
        return dropped

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._entries)


def _rough_size(payloads: list[dict[str, Any]]) -> int:
    # A cheap upper bound on the byte cost of the payload — accurate
    # enough to keep runaway turns out of the cache without doing a
    # full JSON round-trip on every put.
    total = 0
    for p in payloads:
        for k, v in p.items():
            total += len(str(k))
            total += _value_size(v)
    return total


def _value_size(value: Any) -> int:
    if value is None:
        return 4
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bool):
        return 5
    if isinstance(value, (int, float)):
        return 16
    if isinstance(value, dict):
        return sum(len(str(k)) + _value_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_value_size(v) for v in value)
    return len(str(value))
