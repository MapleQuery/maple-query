"""The v2 memory phase: replay cache, plan hints, compaction.

Four prior failures share one root — the loop had no structured
memory of what worked — and this module closes them by construction:

- **Replay cache v2** stores a *digest* of each answered turn (final
  message + condensed evidence events), two orders of magnitude under
  the value ceiling that silently killed v1 caching, and `put`
  returns a reason enum so a skip is never silent again.
- **`CachedSnapshotHash`** runs the two warehouse freshness queries
  at most once per refresh window (not twice per turn), canonicalizes
  the timestamps before hashing so key stability doesn't depend on
  BQ STRING-cast formatting, and fires `invalidate_on_snapshot` when
  the hash actually changes — the dead method gets its caller.
- **Plan hints** render prior *answered* turn records (client-echoed,
  validated) as a deterministic system hint when their lexical
  overlap with the new question clears a threshold. Caveated,
  no-data, and clarified plans never prime the model.
- **Compaction v2** keeps the last N turns verbatim and represents
  everything older only by its records — the LLM topic summary (and
  its cost, and its priming bias) is gone.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent import records as records_mod
from semantic_enrich.core.agent.phases import (
    RecallOutcome,
    SystemHint,
    TurnContext,
    _clarify_followup_hint,
    last_clarify_record,
)
from semantic_enrich.core.agent_cache import cache_key
from semantic_enrich.core.sql_normalize import _mask_string_literals
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent.memory")

PutReason = Literal["stored", "too_large", "not_terminal", "not_answered"]

# Digest values are ~10-50 KB; anything above this ceiling on a
# worst-case turn indicates a digest bug, not a big answer.
SANITY_CEILING_BYTES = 256 * 1024

_DIGEST_CANDIDATE_KEYS = ("package_id", "title", "similarity")
_DIGEST_SAMPLE_ROWS = 3


# ── replay cache v2 ──


@dataclass
class DigestEntry:
    events: list[dict[str, Any]]
    created_at: float
    size_bytes: int
    snapshot_hash: str


@dataclass
class ReplayCacheV2:
    """Thread-safe LRU+TTL cache of turn digests, keyed exactly as v1
    (`sha256(question_normalized || prompt_hash || snapshot_hash)`)."""

    max_entries: int
    ttl_seconds: int
    _entries: OrderedDict[str, DigestEntry] = field(
        default_factory=OrderedDict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> DigestEntry | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                _LOG.info("replay_cache_get", hit=False, key_prefix=key[:12])
                return None
            if now - entry.created_at > self.ttl_seconds:
                self._entries.pop(key, None)
                _LOG.info(
                    "replay_cache_get",
                    hit=False,
                    expired=True,
                    key_prefix=key[:12],
                )
                return None
            self._entries.move_to_end(key)
            _LOG.info("replay_cache_get", hit=True, key_prefix=key[:12])
            return entry

    def put(
        self,
        key: str,
        *,
        events: list[agent_events.AgentEvent],
        outcome: str,
        snapshot_hash: str,
    ) -> PutReason:
        """Digest and store one finished turn. Every non-`stored`
        outcome is returned (and logged by the caller) — the v1
        design's silent skips are what let 'cache never hits' go
        undiagnosed."""
        if not events or not isinstance(events[-1], agent_events.Done):
            return "not_terminal"
        if outcome != "answered":
            # Replaying a surrender would freeze a failure; deflections
            # are near-free to recompute.
            return "not_answered"
        digest = build_digest(events)
        size = sum(len(str(p)) for p in digest)
        if size > SANITY_CEILING_BYTES:
            return "too_large"
        entry = DigestEntry(
            events=digest,
            created_at=time.monotonic(),
            size_bytes=size,
            snapshot_hash=snapshot_hash,
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return "stored"

    def invalidate_on_snapshot(self, current_snapshot_hash: str) -> int:
        """Drop entries recorded under a different warehouse snapshot.
        Returns the number dropped."""
        dropped = 0
        with self._lock:
            for key in list(self._entries):
                if (
                    self._entries[key].snapshot_hash
                    != current_snapshot_hash
                ):
                    self._entries.pop(key, None)
                    dropped += 1
        if dropped:
            _LOG.info("replay_cache_invalidated", dropped=dropped)
        return dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def build_digest(
    events: list[agent_events.AgentEvent],
) -> list[dict[str, Any]]:
    """Condense a turn's event stream to what a replay needs: the
    evidence-rail summary and the answer. Full `rows` batches and
    per-token deltas are dropped; `message_delta`s collapse to one."""
    digest: list[dict[str, Any]] = []
    message = ""
    for event in events:
        payload = event.to_dict()
        kind = payload.get("type")
        if kind == "message_delta":
            message += str(payload.get("delta", ""))
        elif kind == "datasets_ranked":
            digest.append(
                {
                    **payload,
                    "candidates": [
                        {
                            k: c.get(k)
                            for k in _DIGEST_CANDIDATE_KEYS
                        }
                        for c in payload.get("candidates", [])
                    ],
                }
            )
        elif kind == "sql_executed":
            digest.append(
                {
                    **payload,
                    "sample_rows": payload.get("sample_rows", [])[
                        :_DIGEST_SAMPLE_ROWS
                    ],
                }
            )
        elif kind in ("sql_generated", "verification", "turn_record", "done"):
            digest.append(payload)
    if message:
        # One collapsed delta, placed before the terminal events.
        insert_at = max(0, len(digest) - 1)
        digest.insert(insert_at, {"type": "message_delta", "delta": message})
    return digest


def replay(
    entry: DigestEntry,
    *,
    turn_id: str,
    conversation_id: str,
    key: str,
    delay_ms: int,
) -> Iterator[agent_events.AgentEvent]:
    """Digest → fresh event stream: new turn ids throughout, a fresh
    `turn_record` (so client memory stays consistent), the configured
    inter-event delay for progressive rendering."""
    yield agent_events.TurnStart(
        conversation_id=conversation_id, turn_id=turn_id, cached=True
    )
    yield agent_events.CacheHit(cache_key_prefix=key[:12])
    delay = max(0, delay_ms) / 1000.0
    for payload in entry.events:
        kind = payload.get("type")
        if kind in ("done", "turn_record"):
            payload = dict(payload)
            if kind == "done":
                payload["turn_id"] = turn_id
            else:
                payload["record"] = {
                    **payload.get("record", {}),
                    "turn_id": turn_id,
                }
        frame = _payload_to_event(payload)
        if frame is None:
            continue
        if delay:
            time.sleep(delay)
        yield frame


def _payload_to_event(
    payload: dict[str, Any],
) -> agent_events.AgentEvent | None:
    kind = payload.get("type")
    if not isinstance(kind, str):
        return None
    cls = agent_events._EVENT_CLASSES.get(kind)
    if cls is None:
        return None
    body = {k: v for k, v in payload.items() if k != "type"}
    try:
        return cls(**body)  # type: ignore[return-value]
    except TypeError:
        return None


# ── cached snapshot hash ──


class CachedSnapshotHash:
    """Wrap a snapshot-hash provider so the freshness queries run at
    most once per refresh window, and snapshot *changes* trigger cache
    invalidation at the refresh boundary (no background thread — the
    check piggybacks on the first turn after the window lapses)."""

    def __init__(
        self,
        *,
        provider: Callable[[], str],
        refresh_seconds: int,
        on_change: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._refresh_seconds = refresh_seconds
        self._on_change = on_change
        self._clock = clock
        self._lock = threading.Lock()
        self._value: str | None = None
        self._fetched_at = 0.0

    def __call__(self) -> str:
        with self._lock:
            now = self._clock()
            if (
                self._value is not None
                and now - self._fetched_at < self._refresh_seconds
            ):
                return self._value
            fresh = self._provider()
            changed = self._value is not None and fresh != self._value
            self._value = fresh
            self._fetched_at = now
        if changed and self._on_change is not None:
            _LOG.info("snapshot_hash_changed", new_hash_prefix=fresh[:12])
            self._on_change(fresh)
        return fresh


def make_snapshot_hash_provider_v2(
    bq: BqClient, settings: Settings
) -> Callable[[], str]:
    """The v1 provider's two freshness queries, with the timestamps
    canonicalized before hashing so key stability no longer depends on
    BQ STRING-cast determinism. Wrap in `CachedSnapshotHash` — this
    raw provider still costs two queries per call."""
    project_id = settings.gcp_project_id
    if not project_id:
        return lambda: "no-snapshot"
    sql_ds = (
        f"SELECT CAST(MAX(generated_at) AS STRING) AS max_ts "
        f"FROM `{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_datasets_table}`"
    )
    sql_cols = (
        f"SELECT CAST(MAX(generated_at) AS STRING) AS max_ts "
        f"FROM `{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_columns_table}`"
    )

    def _provider() -> str:
        try:
            ds_rows = list(bq.query_rows(sql_ds))
            col_rows = list(bq.query_rows(sql_cols))
        except Exception:  # pragma: no cover - defensive
            return "unknown-snapshot"
        ds_ts = ds_rows[0].get("max_ts") if ds_rows else None
        col_ts = col_rows[0].get("max_ts") if col_rows else None
        raw = "||".join(
            (canonicalize_timestamp(ds_ts), canonicalize_timestamp(col_ts))
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return _provider


def canonicalize_timestamp(raw: Any) -> str:
    """Parse → UTC → isoformat, so two STRING-cast spellings of the
    same instant hash identically. Unparseable values pass through as
    strings (a stable wrong spelling still yields a stable key)."""
    if raw is None:
        return "none"
    text = str(raw).strip()
    try:
        parsed = datetime.fromisoformat(text.replace(" ", "T"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


# ── plan hints ──


def select_plan_hints(
    *,
    question: str,
    turn_records: list[dict[str, Any]],
    max_hints: int,
    min_overlap: float,
) -> list[dict[str, Any]]:
    """Top-scoring prior *answered* records by Jaccard overlap between
    the new question's gist tokens and each record's gist + columns +
    package titles. Strictly answered-only: caveated, no-data, and
    clarified plans must not prime the model."""
    question_tokens = set(records_mod.question_gist(question).split())
    if not question_tokens:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    seen_plans: set[tuple[str, ...]] = set()
    for record in turn_records:
        if record.get("outcome") != "answered":
            continue
        # One hint per distinct plan: replayed turns re-emit identical
        # records, which must not crowd out a second, different plan.
        plan_key = (
            str(record.get("question_gist", "")),
            *sorted(
                str(p.get("package_id", ""))
                for p in record.get("packages", [])
                if isinstance(p, dict)
            ),
        )
        if plan_key in seen_plans:
            continue
        seen_plans.add(plan_key)
        record_tokens = set(str(record.get("question_gist", "")).split())
        for column in record.get("columns_used", []):
            record_tokens.update(
                records_mod.question_gist(str(column).replace("_", " ")).split()
            )
        for package in record.get("packages", []):
            title = package.get("title") if isinstance(package, dict) else None
            if isinstance(title, str):
                record_tokens.update(records_mod.question_gist(title).split())
        if not record_tokens:
            continue
        union = question_tokens | record_tokens
        overlap = len(question_tokens & record_tokens) / len(union)
        if overlap >= min_overlap:
            scored.append((overlap, record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [record for _score, record in scored[:max_hints]]


def render_plan_hints(selected: list[dict[str, Any]]) -> str:
    lines = [
        "Prior resolved plans from this conversation. When one covers "
        "the new question, do NOT call search_datasets or "
        "search_columns — go straight to list_documents on its "
        "packages (that call re-validates the documents and columns; "
        "fall back to normal retrieval only if they changed):"
    ]
    for record in selected:
        packages = ", ".join(
            f'{p.get("package_id")} ("{p.get("title")}")'
            for p in record.get("packages", [])
        )
        columns = ", ".join(str(c) for c in record.get("columns_used", []))
        sql = record.get("sql")
        sql_shape = (
            _mask_string_literals(str(sql))[:200] if sql else "none"
        )
        doc_ids = ", ".join(str(d) for d in record.get("document_ids", []))
        lines.append(
            f'- "{record.get("question")}" → packages [{packages}], '
            f"columns [{columns}], SQL shape: {sql_shape}; "
            f"document_ids [{doc_ids}]; outcome: answered."
        )
    return "\n".join(lines)


# ── compaction v2 ──


def compact_v2(
    history: list[dict[str, Any]], *, keep_turns: int
) -> list[dict[str, Any]]:
    """Keep the last `keep_turns` user-led turns of user/assistant
    prose verbatim; drop everything older (records carry that load)
    and every tool transcript. No model call, ever."""
    prose = [
        m
        for m in history
        if m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ]
    # A turn starts at a user message: walk back keep_turns user turns.
    starts = [i for i, m in enumerate(prose) if m.get("role") == "user"]
    if not starts or keep_turns <= 0:
        return []
    cut = starts[-keep_turns] if len(starts) >= keep_turns else 0
    return prose[cut:]


# ── the phase ──


class SessionMemory:
    """`MemoryPhase` backed by turn records and the digest cache."""

    def __init__(self, *, cache: ReplayCacheV2) -> None:
        self._cache = cache

    def recall(self, ctx: TurnContext) -> RecallOutcome:
        deps = ctx.deps
        settings = deps.settings
        key = cache_key(
            question=ctx.request.question,
            prompt_hash=deps.prompt_hash,
            snapshot_hash=ctx.snapshot_hash,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return RecallOutcome(
                replay=replay(
                    cached,
                    turn_id=ctx.turn_id,
                    conversation_id=ctx.request.conversation_id,
                    key=key,
                    delay_ms=settings.agent_cache_replay_delay_ms,
                )
            )

        incoming = records_mod.sanitize_incoming(
            ctx.request.turn_records,
            max_records=settings.agent_turn_records_max,
        )
        events: list[agent_events.AgentEvent] = []
        hints: list[SystemHint] = []

        clarify = last_clarify_record(incoming)
        if clarify is not None:
            ctx.state.prior_clarify = True
            hints.append(SystemHint(text=_clarify_followup_hint(clarify)))

        selected = select_plan_hints(
            question=ctx.request.question,
            turn_records=incoming,
            max_hints=settings.agent_plan_hints_max,
            min_overlap=settings.agent_plan_hint_min_overlap,
        )
        if selected:
            hints.append(SystemHint(text=render_plan_hints(selected)))
            events.append(agent_events.PlanHint(records_used=selected))
            # Admit the plans' packages to the tool whitelist so the
            # model can go straight to list_documents without a
            # search_datasets round-trip — the ids come from validated
            # records of this conversation's own answered turns, and
            # list_documents still re-validates docs and columns.
            for record in selected:
                for package in record.get("packages", []):
                    pid = (
                        package.get("package_id")
                        if isinstance(package, dict)
                        else None
                    )
                    if isinstance(pid, str) and pid:
                        ctx.state.known_package_ids.add(pid)

        return RecallOutcome(
            events=events,
            hints=hints,
            history_messages=compact_v2(
                ctx.request.history,
                keep_turns=settings.agent_history_keep_turns_v2,
            ),
        )

    def commit(self, ctx: TurnContext) -> None:
        record = _final_record(ctx)
        if record is None:
            _LOG.info("replay_cache_put", reason="not_terminal")
            return
        reason = self._cache.put(
            cache_key(
                question=ctx.request.question,
                prompt_hash=ctx.deps.prompt_hash,
                snapshot_hash=ctx.snapshot_hash,
            ),
            events=ctx.events,
            outcome=str(record.get("outcome", "")),
            snapshot_hash=ctx.snapshot_hash,
        )
        _LOG.info(
            "replay_cache_put",
            reason=reason,
            outcome=record.get("outcome"),
        )


def _final_record(ctx: TurnContext) -> dict[str, Any] | None:
    """The turn's own record, from the recorded event stream. None when
    the turn never reached a terminal `done` (error paths)."""
    if not ctx.events or not isinstance(ctx.events[-1], agent_events.Done):
        return None
    for event in reversed(ctx.events):
        if isinstance(event, agent_events.TurnRecordEvent):
            return event.record
    return None
