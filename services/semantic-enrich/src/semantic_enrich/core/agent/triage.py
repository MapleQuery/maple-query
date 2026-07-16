"""Query triage: classify each turn before research spends anything.

One cheap structured-output call routes the question to `in_scope`
(continue the pipeline), `off_scope` (templated deflection), `meta`
(describe_corpus-backed short answer), or `clarify` (one focused
question back). The posture is fail-open everywhere: a classifier
error, timeout, schema violation, or low-confidence verdict routes to
`in_scope`, so a misclassification can never block a legitimate
question — it can only cost the research loop it would have cost
anyway.

Deflection text is templated, never model prose: deterministic output
is testable and gives prompt injection no surface. The only
model-phrased handler output is the meta answer, whose inputs are our
own corpus statistics.

Modes (`settings.agent_triage_mode`): `off` skips the phase entirely
(the dispatch layer wires `PassthroughTriage`); `log` classifies and
emits events but always continues (shadow mode, `enforced: false`);
`act` short-circuits confidently-classified non-in_scope turns.
"""
from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

import jinja2

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.agent.phases import TriageOutcome, TurnContext
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent.triage")

_CATEGORIES = frozenset({"in_scope", "off_scope", "meta", "clarify"})

_OFF_SCOPE_REASONS = (
    "provincial",
    "news",
    "opinion",
    "non_canada",
    "personal",
    "jailbreak",
    "other",
)

# Strict Structured Outputs schema: every property required, nullables
# via anyOf, so the vendor guarantees the shape and any deviation is
# caught by the fail-open validation below.
CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category",
        "confidence",
        "reason",
        "off_scope_reason",
        "deflection_hint",
        "clarify_question",
    ],
    "properties": {
        "category": {"type": "string", "enum": sorted(_CATEGORIES)},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "off_scope_reason": {
            "anyOf": [
                {"type": "string", "enum": list(_OFF_SCOPE_REASONS)},
                {"type": "null"},
            ]
        },
        "deflection_hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "clarify_question": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
    },
}

_DEFLECTION_BASE = (
    "MapleQuery answers questions from Canadian **federal** open data "
    "(open.canada.ca)."
)

# Fixed clauses keyed on the classifier's sub-reason — never free text,
# so a hostile question cannot steer the deflection wording.
_REASON_CLAUSES: dict[str, str] = {
    "provincial": (
        "Provincial and municipal matters aren't in the federal corpus."
    ),
    "news": (
        "News and current events aren't covered — the corpus holds "
        "published datasets, not reporting."
    ),
    "opinion": (
        "Opinion and ranking questions can't be answered from the data."
    ),
    "non_canada": "Data about other countries isn't in the corpus.",
    "personal": "It holds no personal or private records.",
    "jailbreak": "That request falls outside what it can help with.",
    "other": "That question falls outside what the data can answer.",
}

_HINT_MAX_CHARS = 160

IDENTITY_LINE = (
    "MapleQuery is a research agent that answers questions from "
    "Canadian federal open data (open.canada.ca). It doesn't disclose "
    "or discuss its underlying model configuration."
)

# Meta questions whose answer is a corpus stat get the deterministic
# template; anything about the underlying model gets the identity line.
_IDENTITY_KEYWORDS = (
    "model",
    "gpt",
    "openai",
    "claude",
    "llm",
    "language model",
    "powered by",
    "built on",
    "which ai",
    "what ai",
)
_STAT_KEYWORDS = (
    "row",
    "record",
    "dataset",
    "package",
    "document",
    "how much data",
    "fresh",
    "updated",
    "up to date",
    "latest",
    "recent",
)

_PHRASE_STATS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


@dataclass(frozen=True)
class _Classification:
    category: str
    confidence: float
    reason: str
    off_scope_reason: str | None
    deflection_hint: str | None
    clarify_question: str | None


def load_triage_template(settings: Settings) -> jinja2.Template:
    """Load the classifier prompt template, crashing at startup when it
    is missing — same posture as the system prompt."""
    path = settings.agent_triage_prompt_path
    if not path.exists():
        raise RuntimeError(f"triage prompt template missing: {path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(path.name)


class QueryTriage:
    """`TriagePhase` implementation backed by the mini classifier."""

    def __init__(self, *, template: jinja2.Template) -> None:
        self._template = template

    @classmethod
    def from_settings(cls, settings: Settings) -> QueryTriage:
        return cls(template=load_triage_template(settings))

    def classify(self, ctx: TurnContext) -> TriageOutcome:
        settings = ctx.deps.settings
        mode = settings.agent_triage_mode
        started = time.monotonic()
        events: list[agent_events.AgentEvent] = []

        classification, fail_open_reason = self._run_classifier(
            ctx, events=events
        )

        category = "in_scope"
        confidence = 0.0
        enforced = False
        short_circuit: str | None = None

        if classification is not None:
            category = classification.category
            confidence = classification.confidence
            if mode == "act":
                if category == "in_scope":
                    enforced = True
                elif confidence < settings.agent_triage_min_confidence:
                    fail_open_reason = "low_confidence"
                else:
                    short_circuit = self._handle(
                        ctx, classification, events=events
                    )
                    if short_circuit is None:
                        # Handler declined (e.g. clarify with no
                        # question) — taxonomy rule says in_scope.
                        category = "in_scope"
                        fail_open_reason = "missing_clarify_question"
                    else:
                        enforced = True

        elapsed_ms = int((time.monotonic() - started) * 1000)
        events.append(
            agent_events.TriageResult(
                category=category,
                confidence=round(confidence, 3),
                elapsed_ms=elapsed_ms,
                enforced=enforced,
            )
        )
        _LOG.info(
            "triage_result",
            category=category,
            confidence=round(confidence, 3),
            mode=mode,
            enforced=enforced,
            fail_open_reason=fail_open_reason,
            elapsed_ms=elapsed_ms,
        )
        return TriageOutcome(
            category=category, events=events, short_circuit=short_circuit
        )

    # ── classifier call ──

    def _run_classifier(
        self,
        ctx: TurnContext,
        *,
        events: list[agent_events.AgentEvent],
    ) -> tuple[_Classification | None, str | None]:
        """One classifier call under a hard deadline. Returns
        `(classification, None)` or `(None, fail_open_reason)`."""
        settings = ctx.deps.settings
        prompt = self._template.render(
            question=ctx.request.question,
            context_hint=_context_hint(ctx.request.history),
        )
        timeout_s = settings.agent_triage_timeout_ms / 1000.0

        def call() -> Any:
            return ctx.deps.openai_client.generate_structured(
                prompt=prompt,
                schema=CLASSIFIER_SCHEMA,
                schema_name="triage",
                model=settings.agent_triage_model,
                temperature=0.0,
                max_tokens=150,
                timeout_s=timeout_s,
            )

        # The deadline is enforced here, not just at the vendor: retry
        # policies or slow connects must not stretch the turn beyond
        # the configured budget. The contextvars copy keeps the
        # Braintrust span scope attached inside the worker thread.
        call_ctx = contextvars.copy_context()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(call_ctx.run, call)
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return None, "classifier_timeout"
        except Exception:
            return None, "classifier_error"
        finally:
            pool.shutdown(wait=False)

        events.append(
            ctx.charge_model_call(
                tokens_in=result.tokens_in, tokens_out=result.tokens_out
            )
        )
        classification = _validate(result.parsed)
        if classification is None:
            return None, "invalid_output"
        return classification, None

    # ── category handlers ──

    def _handle(
        self,
        ctx: TurnContext,
        classification: _Classification,
        *,
        events: list[agent_events.AgentEvent],
    ) -> str | None:
        if classification.category == "off_scope":
            return off_scope_message(
                sub_reason=classification.off_scope_reason,
                deflection_hint=classification.deflection_hint,
            )
        if classification.category == "meta":
            return self._meta_answer(ctx, events=events)
        if classification.category == "clarify":
            question = (classification.clarify_question or "").strip()
            return question or None
        return None

    def _meta_answer(
        self,
        ctx: TurnContext,
        *,
        events: list[agent_events.AgentEvent],
    ) -> str:
        question = ctx.request.question.lower()
        if any(k in question for k in _IDENTITY_KEYWORDS):
            return IDENTITY_LINE

        stats = self._corpus_stats(ctx)
        if stats is None:
            # Warehouse unavailable: still a real answer, just numberless.
            return IDENTITY_LINE

        template_answer = _stats_sentence(stats)
        if any(k in question for k in _STAT_KEYWORDS):
            return template_answer
        # Meta question with no direct stat mapping: one mini call to
        # phrase our own stats — the model never sees anything but them.
        return self._phrase_stats(
            ctx, stats=stats, fallback=template_answer, events=events
        )

    def _corpus_stats(self, ctx: TurnContext) -> dict[str, Any] | None:
        deps = ctx.deps
        tool_ctx = agent_tools.ToolContext(
            bq=deps.bq,
            openai_client=deps.openai_client,
            settings=deps.settings,
            state=ctx.state,
            emit=lambda _event: None,
            trace_parent=deps.trace_parent,
        )
        try:
            stats = agent_tools.dispatch(
                ctx=tool_ctx, tool_name="describe_corpus", args={}
            )
        except Exception:
            return None
        ctx.charge_tool_calls(1)
        return stats

    def _phrase_stats(
        self,
        ctx: TurnContext,
        *,
        stats: dict[str, Any],
        fallback: str,
        events: list[agent_events.AgentEvent],
    ) -> str:
        settings = ctx.deps.settings
        prompt = (
            "You are MapleQuery, a research agent over Canadian federal "
            "open data (open.canada.ca).\n"
            "Answer the user's question about the system using ONLY the "
            "statistics below. Do not discuss models or configuration.\n"
            f"Statistics: {json.dumps(stats, default=str)}\n"
            f"Question: {ctx.request.question}\n"
            "Answer in one or two sentences."
        )
        try:
            result = ctx.deps.openai_client.generate_structured(
                prompt=prompt,
                schema=_PHRASE_STATS_SCHEMA,
                schema_name="triage_meta_answer",
                model=settings.agent_triage_model,
                temperature=0.0,
                max_tokens=150,
                timeout_s=settings.agent_triage_timeout_ms / 1000.0,
            )
        except Exception:
            return fallback
        events.append(
            ctx.charge_model_call(
                tokens_in=result.tokens_in, tokens_out=result.tokens_out
            )
        )
        answer = str(result.parsed.get("answer", "")).strip()
        return answer or fallback


# ── pure helpers ──


def off_scope_message(
    *, sub_reason: str | None, deflection_hint: str | None
) -> str:
    """Deterministic deflection. Jailbreak-flavoured inputs get the
    bare template: no suggestion clause, no echo of anything."""
    reason = sub_reason if sub_reason in _REASON_CLAUSES else "other"
    parts = [_DEFLECTION_BASE, _REASON_CLAUSES[reason]]
    hint = (deflection_hint or "").strip()
    if reason != "jailbreak" and _valid_hint(hint):
        parts.append(f"You could ask instead: {hint}")
    return " ".join(parts)


def _valid_hint(hint: str | None) -> bool:
    if not hint or not hint.strip():
        return False
    text = hint.strip()
    if len(text) > _HINT_MAX_CHARS:
        return False
    lowered = text.lower()
    return not ("http" in lowered or "www." in lowered)


def _stats_sentence(stats: dict[str, Any]) -> str:
    rows = int(stats.get("rows_total") or 0)
    packages = int(stats.get("packages") or 0)
    documents = int(stats.get("documents_loaded") or 0)
    latest = stats.get("latest_load_at")
    sentence = (
        f"The corpus holds {rows:,} rows across {documents:,} loaded "
        f"documents from {packages:,} datasets on open.canada.ca."
    )
    if latest:
        sentence += f" The most recent warehouse load was {latest}."
    return sentence


def _context_hint(history: list[dict[str, Any]]) -> str | None:
    """One line of conversation context: the most recent user message,
    so follow-ups like "what about 2021?" don't misroute as clarify."""
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:200]
    return None


def _validate(parsed: dict[str, Any]) -> _Classification | None:
    category = parsed.get("category")
    if not isinstance(category, str) or category not in _CATEGORIES:
        return None
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(
        confidence, bool
    ):
        return None
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None
    reason = parsed.get("reason")
    if not isinstance(reason, str):
        return None

    def _opt_str(key: str) -> str | None:
        value = parsed.get(key)
        return value if isinstance(value, str) else None

    return _Classification(
        category=category,
        confidence=confidence,
        reason=reason,
        off_scope_reason=_opt_str("off_scope_reason"),
        deflection_hint=_opt_str("deflection_hint"),
        clarify_question=_opt_str("clarify_question"),
    )
