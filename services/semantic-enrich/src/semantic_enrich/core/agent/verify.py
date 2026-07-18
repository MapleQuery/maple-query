"""Answer-fit verification: a cheap semantic check on the composed
answer before it ships.

The observed failure this closes: the loop completes the whole happy
path — retrieval, guard-passing SQL, real rows — then ships an answer
that does not address the question (or a surrender no search history
justifies). The checker judges *fit*, not factual correctness: does
this answer address the shape of what was asked. It sees evidence
assembled deterministically from the turn trace, never vibes.

Four dispositions: ship it (`answer`), prepend a template-composed
caveat (`caveat`), re-enter research once with a gap hint (`retry`),
or ask the user (`clarify` — only when no real data would be withheld
behind the question). The posture is fail-open everywhere: checker
error, timeout, or schema-invalid output ships the answer unchanged.

Modes (`settings.agent_verify_mode`): `off` skips the phase entirely
(the dispatch layer wires `AlwaysFitsVerifier`); `log` checks and
emits `verification` events but never alters the answer (shadow mode,
the data source for the act-mode precision gate); `act` enforces the
dispositions.
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
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import (
    ResearchResult,
    SystemHint,
    TurnContext,
    Verdict,
)
from semantic_enrich.core.sql_normalize import _mask_string_literals
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent.verify")

_ACTIONS = frozenset({"answer", "caveat", "retry", "clarify"})

# Strict Structured Outputs schema — every property required, nullables
# via anyOf, so any deviation is caught by the fail-open validation.
CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fits", "confidence", "gap", "action", "retry_hint"],
    "properties": {
        "fits": {"type": "boolean"},
        "confidence": {"type": "number"},
        "gap": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "retry_hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}


@dataclass(frozen=True)
class _CheckResult:
    fits: bool
    confidence: float
    gap: str | None
    action: str
    retry_hint: str | None


def load_verify_template(settings: Settings) -> jinja2.Template:
    """Load the checker prompt template, crashing at startup when it is
    missing — same posture as the system prompt."""
    path = settings.agent_verify_prompt_path
    if not path.exists():
        raise RuntimeError(f"verify prompt template missing: {path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(path.name)


# ── evidence assembly (pure) ──


def assemble_inputs(
    ctx: TurnContext, result: ResearchResult
) -> dict[str, Any]:
    """The checker's evidence, assembled deterministically from the
    turn trace. SQL ships shape-only (string literals blanked) so no
    row content or document ids leak into the checker prompt."""
    sql_ok = [r for r in result.sql_runs if r.get("status") == "ok"]
    last_ok = sql_ok[-1] if sql_ok else None
    return {
        "question": ctx.request.question,
        "candidate_answer": result.candidate_answer,
        "answer_kind": "answer" if sql_ok else "no_data",
        "datasets_used": _datasets_used(ctx, result),
        "columns_referenced": list(result.columns_referenced),
        "sql_shapes": [
            _mask_string_literals(str(r.get("sql", ""))) for r in sql_ok
        ],
        "result_summary": {
            "row_count": last_ok.get("row_count") if last_ok else None,
            "null_ratio_warning": (
                last_ok.get("null_ratio_warning") if last_ok else None
            ),
        },
        "searches_tried": [
            {
                "query": s.get("query"),
                "top_similarity": s.get("top_similarity"),
            }
            for s in ctx.trace.searches
        ],
        "question_asks_for": None,
    }


def _datasets_used(
    ctx: TurnContext, result: ResearchResult
) -> list[dict[str, str | None]]:
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


def compose_caveat(*, gap: str, answer: str) -> str:
    """Template-composed, never model-rewritten: the answer text the
    user sees is the answer text the research model wrote, with the
    declared gap prepended."""
    gap_text = gap.strip().rstrip(".")
    return f"**Partial answer:** this does not cover {gap_text}.\n\n{answer}"


def compose_clarify(*, gap: str) -> str:
    gap_text = gap.strip().rstrip(".")
    return (
        "I couldn't confidently find data for this as asked. Could you "
        f"narrow it down — specifically: {gap_text}? A program name, "
        "department, or timeframe helps me search better."
    )


def compose_retry_hint(*, gap: str, retry_hint: str | None) -> str:
    text = f"Your previous answer missed: {gap.strip()}."
    if retry_hint and retry_hint.strip():
        text += f" Look for: {retry_hint.strip()}."
    return text


# ── the phase ──


class AnswerFitVerifier:
    """`VerifyPhase` implementation backed by the mini fit checker."""

    def __init__(self, *, template: jinja2.Template) -> None:
        self._template = template

    @classmethod
    def from_settings(cls, settings: Settings) -> AnswerFitVerifier:
        return cls(template=load_verify_template(settings))

    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict:
        settings = ctx.deps.settings
        mode = settings.agent_verify_mode
        if mode == "off":  # dispatch wires the stub; guard anyway
            return Verdict(action="accept")

        started = time.monotonic()
        events: list[agent_events.AgentEvent] = []
        inputs = assemble_inputs(ctx, result)
        check, fail_open_reason = self._run_checker(
            ctx, inputs=inputs, events=events
        )

        if check is None:
            events.append(
                agent_events.Verification(
                    fits=True,
                    action="answer",
                    confidence=0.0,
                    reason=fail_open_reason or "",
                    enforced=False,
                )
            )
            self._log(
                mode=mode,
                action="answer",
                fits=True,
                enforced=False,
                fail_open_reason=fail_open_reason,
                started=started,
            )
            return Verdict(action="accept", events=events)

        action = "answer" if check.fits else check.action
        gap = (check.gap or "").strip()
        demotions: list[str] = []

        if mode == "act" and action != "answer":
            if not gap:
                # Every non-answer disposition composes from the gap;
                # a checker that names none has nothing to enforce.
                demotions.append("empty_gap")
                action = "answer"
            elif check.confidence < settings.agent_verify_min_confidence:
                demotions.append("low_confidence")
                action = "caveat"
        if mode == "act" and action == "retry" and (
            final or not ctx.retries_remaining()
        ):
            demotions.append("retry_unavailable")
            action = "caveat"
        if mode == "act" and action == "clarify":
            if inputs["answer_kind"] != "no_data":
                # Never withhold real data behind a question.
                demotions.append("has_real_data")
                action = "caveat"
            elif ctx.state.prior_clarify:
                demotions.append("consecutive_clarify")
                action = "caveat"

        enforced = mode == "act"
        events.append(
            agent_events.Verification(
                fits=check.fits,
                action=action if enforced else check.action,
                confidence=round(check.confidence, 3),
                reason=gap,
                enforced=enforced,
            )
        )
        self._log(
            mode=mode,
            action=action,
            fits=check.fits,
            enforced=enforced,
            fail_open_reason=None,
            started=started,
            demotions=demotions,
        )
        if not enforced or action == "answer":
            return Verdict(action="accept", events=events)
        if action == "caveat":
            return Verdict(
                action="accept",
                events=events,
                composed_message=compose_caveat(
                    gap=gap, answer=result.candidate_answer
                ),
                outcome_override="answered_with_caveat",
            )
        if action == "retry":
            return Verdict(
                action="retry",
                events=events,
                hints=[
                    SystemHint(
                        text=compose_retry_hint(
                            gap=gap, retry_hint=check.retry_hint
                        )
                    )
                ],
            )
        return Verdict(
            action="accept",
            events=events,
            composed_message=compose_clarify(gap=gap),
            outcome_override="clarify",
        )

    # ── checker call ──

    def _run_checker(
        self,
        ctx: TurnContext,
        *,
        inputs: dict[str, Any],
        events: list[agent_events.AgentEvent],
    ) -> tuple[_CheckResult | None, str | None]:
        """One checker call under a hard deadline. Returns
        `(check, None)` or `(None, fail_open_reason)`."""
        settings = ctx.deps.settings
        prompt = self._template.render(
            evidence=json.dumps(inputs, indent=1, default=str)
        )
        timeout_s = settings.agent_verify_timeout_ms / 1000.0

        def call() -> Any:
            return ctx.deps.openai_client.generate_structured(
                prompt=prompt,
                schema=CHECK_SCHEMA,
                schema_name="verify",
                model=settings.agent_verify_model,
                temperature=0.0,
                max_tokens=300,
                timeout_s=timeout_s,
            )

        # Deadline enforced here, not just at the vendor — same posture
        # as triage. The contextvars copy keeps the tracing span scope
        # attached inside the worker thread.
        call_ctx = contextvars.copy_context()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(call_ctx.run, call)
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return None, "checker_timeout"
        except Exception:
            return None, "checker_error"
        finally:
            pool.shutdown(wait=False)

        events.append(
            ctx.charge_model_call(
                tokens_in=result.tokens_in, tokens_out=result.tokens_out
            )
        )
        check = _validate(result.parsed)
        if check is None:
            return None, "invalid_output"
        return check, None

    def _log(
        self,
        *,
        mode: str,
        action: str,
        fits: bool,
        enforced: bool,
        fail_open_reason: str | None,
        started: float,
        demotions: list[str] | None = None,
    ) -> None:
        _LOG.info(
            "verification",
            mode=mode,
            action=action,
            fits=fits,
            enforced=enforced,
            fail_open_reason=fail_open_reason,
            demotions=demotions or [],
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )


def _validate(parsed: dict[str, Any]) -> _CheckResult | None:
    fits = parsed.get("fits")
    if not isinstance(fits, bool):
        return None
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(
        confidence, bool
    ):
        return None
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None
    action = parsed.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        return None

    def _opt_str(key: str) -> str | None:
        value = parsed.get(key)
        return value if isinstance(value, str) else None

    return _CheckResult(
        fits=fits,
        confidence=confidence,
        gap=_opt_str("gap"),
        action=action,
        retry_hint=_opt_str("retry_hint"),
    )
