"""The TurnRecord: the deterministic memory primitive of the v2 loop.

Constructed by the pipeline's finish step from the turn context and
trace — no LLM involved, so what survives a turn is exact, not a
paraphrase. Records ride to the client in the `turn_record` event and
come back on the next request as `ChatRequest.turn_records`; the
server stays stateless.

Everything client-supplied is validated on ingest: schema version,
field types, string-length caps. Invalid records are dropped with a
log, never a 400 — a corrupt localStorage entry must not brick a
conversation.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from semantic_enrich.providers.logging import get_logger

if TYPE_CHECKING:  # circular-import guard: phases imports stay type-only
    from semantic_enrich.core.agent.phases import (
        ResearchResult,
        TurnContext,
    )

_LOG = get_logger("semantic_enrich.agent.records")

RECORD_VERSION = 1

OUTCOMES = frozenset(
    {
        "answered",
        "answered_with_caveat",
        "no_data",
        "deflected",
        "clarified",
        "error",
    }
)

# Per-record serialized ceiling. Typical records are 1-2 KB; anything
# past this is malformed or hostile and gets dropped on ingest.
MAX_RECORD_BYTES = 16 * 1024
_MAX_STR = 2_000
_MAX_LIST = 50

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Minimal English stopword set for gist normalization — enough to make
# lexical overlap meaningful without a language-processing dependency.
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "how", "i", "in", "is", "it", "its", "many", "me",
        "much", "my", "of", "on", "or", "our", "should", "since",
        "tell", "that", "the", "their", "there", "these", "they",
        "this", "to", "us", "was", "we", "were", "what", "when",
        "where", "which", "who", "will", "with", "would", "you", "your",
    ]
)


def question_gist(question: str) -> str:
    """Casefold, strip punctuation, drop stopwords, collapse
    whitespace. The lexical form plan-hint overlap is scored on."""
    text = _PUNCT_RE.sub(" ", question.casefold())
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def build(
    ctx: TurnContext,
    *,
    message: str,
    result: ResearchResult | None,
    outcome: str,
) -> dict[str, Any]:
    """Assemble the record deterministically from the turn context.
    `result` is None on triage short-circuit paths."""
    sql_ok = [
        r
        for r in (result.sql_runs if result else [])
        if r.get("status") == "ok"
    ]
    last_ok = sql_ok[-1] if sql_ok else None
    final_sql = str(last_ok.get("sql", "")) if last_ok else ""
    return {
        "v": RECORD_VERSION,
        "turn_id": ctx.turn_id,
        "question": ctx.request.question,
        "question_gist": question_gist(ctx.request.question),
        "category": ctx.triage_category,
        "outcome": outcome if outcome in OUTCOMES else "error",
        "packages": _packages(ctx, result),
        "columns_used": list(result.columns_referenced) if result else [],
        "document_ids": sorted(ctx.state.known_document_ids)[:_MAX_LIST],
        "sql": final_sql[:_MAX_STR] or None,
        "row_count": last_ok.get("row_count") if last_ok else None,
        "searches_tried": list(ctx.trace.searches),
        "answer_digest": message[:300],
        "dollars": round(ctx.dollars_spent, 6),
        "snapshot_hash": ctx.snapshot_hash,
        "created_at": datetime.now(UTC).isoformat(),
        # Loop diagnostics — additive over the versioned schema; the
        # eval harness and parity report read them.
        "loop_impl": "v2",
        "tool_call_count": ctx.tool_call_count,
        "reformulations_used": ctx.reformulations_used,
        "verify_retries_used": ctx.verify_retries_used,
    }


def validate(record: Any) -> dict[str, Any] | None:
    """One client-supplied record → the record, or None if malformed.
    Unknown extra keys are tolerated; wrong types and oversizes are
    not."""
    if not isinstance(record, dict):
        return None
    if record.get("v") != RECORD_VERSION:
        return None
    if record.get("outcome") not in OUTCOMES:
        return None
    for key in ("question", "question_gist", "answer_digest"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) > _MAX_STR:
            return None
    sql = record.get("sql")
    if sql is not None and (
        not isinstance(sql, str) or len(sql) > _MAX_STR
    ):
        return None
    for key in ("packages", "columns_used", "document_ids", "searches_tried"):
        value = record.get(key)
        if not isinstance(value, list) or len(value) > _MAX_LIST:
            return None
    if not all(
        isinstance(p, dict) and isinstance(p.get("package_id"), str)
        for p in record["packages"]
    ):
        return None
    if not all(isinstance(c, str) for c in record["columns_used"]):
        return None
    try:
        if len(json.dumps(record, default=str)) > MAX_RECORD_BYTES:
            return None
    except (TypeError, ValueError):
        return None
    return record


def sanitize_incoming(
    turn_records: list[dict[str, Any]], *, max_records: int
) -> list[dict[str, Any]]:
    """Validate a client-supplied record list, dropping invalid
    entries with a log and keeping only the newest `max_records`."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for raw in turn_records[-max_records:]:
        record = validate(raw)
        if record is None:
            dropped += 1
            continue
        kept.append(record)
    if dropped:
        _LOG.info(
            "turn_records_dropped", dropped=dropped, kept=len(kept)
        )
    return kept


def _packages(
    ctx: TurnContext, result: ResearchResult | None
) -> list[dict[str, Any]]:
    """Cited packages with titles resolved from this turn's search
    results (the trace stores ids; titles live in the tool payloads)."""
    if result is None:
        return []
    titles: dict[str, str | None] = {}
    for payload in ctx.state.search_results.values():
        for candidate in payload.get("candidates", []):
            pid = str(candidate.get("package_id", ""))
            if pid:
                titles.setdefault(pid, candidate.get("title"))
    return [
        {"package_id": pid, "title": titles.get(pid)}
        for pid in result.packages_cited
    ]
