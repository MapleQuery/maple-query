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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

import jinja2

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.evidence import titles_by_package
from semantic_enrich.core.agent.grounding import extract_numbers
from semantic_enrich.core.agent.magnitude import (
    MagnitudeVerdict,
    evaluate_magnitude,
)
from semantic_enrich.core.agent.phases import (
    ResearchResult,
    SystemHint,
    TurnContext,
    Verdict,
    is_descriptive,
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


# The explore rubric's dispositions are restricted *structurally*, not
# by runtime demotion. `clarify` and `retry` are absent from the enum
# rather than demoted away — and that is the safety property that makes
# the rubric strictly better than the bypass it replaces. The fit
# checker guards `clarify` with a demotion, and that demotion failing to
# fire is precisely the bug this milestone exists to close; an absent
# enum value depends on nothing firing correctly. Even a badly
# miscalibrated explore checker cannot replace a summary with a question
# or spend a second research leg.
_EXPLORE_ACTIONS = frozenset({"answer", "caveat"})

EXPLORE_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fits", "confidence", "gap", "action"],
    "properties": {
        "fits": {"type": "boolean"},
        "confidence": {"type": "number"},
        "gap": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "action": {"type": "string", "enum": sorted(_EXPLORE_ACTIONS)},
    },
}


@dataclass(frozen=True)
class _CheckResult:
    fits: bool
    confidence: float
    gap: str | None
    action: str
    retry_hint: str | None


def load_verify_explore_template(
    settings: Settings,
) -> jinja2.Template | None:
    """The descriptive rubric's template. Unlike the fit prompt this
    returns None when missing rather than raising: the explore path
    degrades to the bypass it replaced, which is a safe posture, while a
    missing fit prompt has no safe fallback."""
    path = settings.agent_verify_explore_prompt_path
    if not path.exists():
        return None
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(path.name)


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
    titles = titles_by_package(ctx)
    return [
        {"package_id": pid, "title": titles.get(pid)}
        for pid in result.packages_cited
    ]


def assemble_explore_inputs(
    ctx: TurnContext, result: ResearchResult
) -> dict[str, Any]:
    """Evidence for a descriptive turn.

    Deliberately a different shape from `assemble_inputs`: the fit
    checker's block is built for numeric answers (`sql_shapes`,
    `row_count`, `null_ratio_warning`) and feeding descriptions through
    it would degrade both. Two small prompts beat one prompt with a mode
    flag.
    """
    titles = titles_by_package(ctx)
    inputs: dict[str, Any] = {
        "question": ctx.request.question,
        "candidate_answer": result.candidate_answer,
        "packages_described": list(result.packages_cited),
        "columns_surfaced": len(ctx.state.doc_columns),
        "documents_listed": len(ctx.state.known_document_ids),
        "tools_used": sorted(
            {str(call.get("tool")) for call in ctx.trace.tool_calls}
        ),
        # Deterministic, not a model judgement: reuses the same monetary
        # extractor the numeric-trust work uses, so "did this answer
        # state a figure" never becomes a matter of opinion.
        "claims_a_total": bool(
            extract_numbers(_prose_only(result.candidate_answer)).monetary
        ),
    }
    # Present only on a genuinely scoped turn. A *typed* exploratory
    # question names no datasets, so an empty list here is not "the
    # answer went out of scope" — it is "there was no scope". Sending
    # `[]` invited the rubric to read the absence as a defect, which is
    # exactly what it did on the first shadow run. Omitting the key
    # makes the prompt's wrong-dataset condition unanswerable rather
    # than falsely answerable.
    if ctx.scope_package_ids:
        inputs["scope_packages"] = [
            {"package_id": pid, "title": titles.get(pid)}
            for pid in ctx.scope_package_ids
        ]
    return inputs


# Dataset citations render as `[Title](/datasets/<id>)`, and federal
# dataset titles routinely carry figures — "Dashboard for infrastructure
# projects worth $20 million and over". A dollar sign inside a citation
# label is part of a *name*, not a claim the answer is making.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")


def _prose_only(text: str) -> str:
    """The answer with dataset citations removed.

    Scoped deliberately to `claims_a_total`. The grounding report reads
    the raw answer, and narrowing *that* would change calibrated numeric
    behaviour for a benefit this rubric does not need — a title figure
    matches no derivation either way.
    """
    return _MD_LINK_RE.sub(" ", text or "")


def compose_explore_caveat(*, gap: str, answer: str) -> str:
    gap_text = gap.strip().rstrip(".")
    return (
        f"**Note:** this description does not cover {gap_text}."
        f"\n\n{answer}"
    )


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


# ── magnitude gate composition (deterministic) ──


def _magnitude_action(
    mag: MagnitudeVerdict, *, retry_available: bool
) -> str:
    """The disposition a magnitude finding maps to, independent of mode
    (the event records this as the would-be action even in shadow)."""
    finding = mag.finding
    if finding is None:
        return "answer"
    # Only the "re-examine the column" class (floor/ceiling) spends a
    # retry; a cross-source sum is possibly legitimate, so it caveats.
    if finding.retry_eligible and finding.severity == "hard" and retry_available:
        return "retry"
    return "caveat"


def _magnitude_hint(mag: MagnitudeVerdict) -> str:
    finding = mag.finding
    if finding is not None and finding.hint:
        return finding.hint
    return "your computed total looks implausible; re-examine the column and scope"


def _compose_magnitude(
    mag: MagnitudeVerdict, fit: Verdict, result: ResearchResult
) -> Verdict:
    """Prepend the magnitude caveat to the fit disposition. Caveats
    compose (both prepend); a fit retry/clarify wins (the numeric caveat
    is moot on a discarded or replaced answer)."""
    finding = mag.finding
    assert finding is not None
    if fit.action == "retry" or fit.outcome_override == "clarified":
        return fit
    base = (
        fit.composed_message
        if fit.composed_message is not None
        else result.candidate_answer
    )
    real_answer = any(r.get("status") == "ok" for r in result.sql_runs)
    return Verdict(
        action="accept",
        events=fit.events,
        composed_message=finding.caveat + base,
        outcome_override=fit.outcome_override
        or ("answered_with_caveat" if real_answer else None),
    )


# ── the phase ──


class AnswerFitVerifier:
    """`VerifyPhase` implementation backed by the mini fit checker."""

    def __init__(
        self,
        *,
        template: jinja2.Template,
        explore_template: jinja2.Template | None = None,
    ) -> None:
        self._template = template
        self._explore_template = explore_template

    @classmethod
    def from_settings(cls, settings: Settings) -> AnswerFitVerifier:
        return cls(
            template=load_verify_template(settings),
            explore_template=load_verify_explore_template(settings),
        )

    def check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool = False,
    ) -> Verdict:
        settings = ctx.deps.settings
        if settings.agent_verify_mode == "off":  # dispatch wires the stub
            return Verdict(action="accept")
        if is_descriptive(ctx, result):
            # A description is judged on description. The numeric gates
            # below (magnitude, fit) are calibrated against answers that
            # ran SQL and would misfire here.
            return self._explore_check(ctx, result)

        events: list[agent_events.AgentEvent] = []
        # Deterministic magnitude/units gate runs first, before the
        # fit-checker's model call. It consumes the captured derivation
        # (7.1) + grounding (7.2); no new model call, no confidence to
        # demote — a "1,000 rows summed to $8" is arithmetic, not a
        # judgement call.
        mag = evaluate_magnitude(result.derivations, result.grounding, settings)
        mag_enforce = settings.agent_magnitude_mode == "act"
        retry_available = not final and ctx.retries_remaining()
        mag_action = _magnitude_action(mag, retry_available=retry_available)
        if mag.finding is not None:
            self._emit_magnitude(
                mag, mag_action, events=events, enforced=mag_enforce
            )
        # A hard finding with a retry available discards this candidate
        # before the fit-checker call is even spent.
        if mag_enforce and mag_action == "retry":
            return Verdict(
                action="retry",
                events=events,
                hints=[SystemHint(text=_magnitude_hint(mag))],
            )

        fit = self._fit_check(ctx, result, final, events=events)
        if mag_enforce and mag.finding is not None:
            return _compose_magnitude(mag, fit, result)
        return fit

    def _explore_check(
        self, ctx: TurnContext, result: ResearchResult
    ) -> Verdict:
        """The descriptive rubric. Fail-open like every other gate here,
        and structurally incapable of `clarify` or `retry`."""
        settings = ctx.deps.settings
        mode = settings.agent_verify_explore_mode
        started = time.monotonic()
        events: list[agent_events.AgentEvent] = []
        if mode == "off" or self._explore_template is None:
            return Verdict(action="accept")

        inputs = assemble_explore_inputs(ctx, result)
        check, fail_open_reason = self._run_checker(
            ctx,
            inputs=inputs,
            events=events,
            template=self._explore_template,
            schema=EXPLORE_CHECK_SCHEMA,
            schema_name="verify_explore",
            actions=_EXPLORE_ACTIONS,
        )
        if check is None:
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
        enforced = mode == "act"
        if enforced and action == "caveat" and not gap:
            # Every corrective disposition composes from the gap; a
            # checker naming none has nothing to enforce.
            action = "answer"
        events.append(
            agent_events.Verification(
                fits=check.fits,
                action=action,
                confidence=round(check.confidence, 3),
                reason=gap,
                enforced=enforced,
                kind="explore",
            )
        )
        self._log(
            kind="explore",
            mode=mode,
            action=action,
            fits=check.fits,
            enforced=enforced,
            fail_open_reason=None,
            started=started,
        )
        if not enforced or action == "answer":
            return Verdict(action="accept", events=events)
        return Verdict(
            action="accept",
            events=events,
            composed_message=compose_explore_caveat(
                gap=gap, answer=result.candidate_answer
            ),
            # No outcome override: a caveated exploration is still an
            # exploration, and `_outcome` already tags it `explored`.
        )

    def _fit_check(
        self,
        ctx: TurnContext,
        result: ResearchResult,
        final: bool,
        *,
        events: list[agent_events.AgentEvent],
    ) -> Verdict:
        settings = ctx.deps.settings
        mode = settings.agent_verify_mode
        started = time.monotonic()
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
        if mode == "act" and action == "retry" and (
            inputs["answer_kind"] != "no_data"
        ):
            # Retry is the reformulation policy's second enforcement
            # point — a surrender fix. An answer grounded in real rows
            # that misses a dimension ships cheaper and just as
            # honestly under a caveat; a second research leg to
            # *improve* real data rarely pays its full-turn cost.
            demotions.append("answer_already_grounded")
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
            kind="fit",
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
                # A caveat on a real answer is a caveated answer; a
                # caveat prepended to a no-data claim does not upgrade
                # it — the record (and the replay/plan-hint gates
                # behind it) must keep calling a surrender a surrender.
                outcome_override=(
                    "answered_with_caveat"
                    if inputs["answer_kind"] == "answer"
                    else None
                ),
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
            outcome_override="clarified",
        )

    # ── checker call ──

    def _run_checker(
        self,
        ctx: TurnContext,
        *,
        inputs: dict[str, Any],
        events: list[agent_events.AgentEvent],
        template: jinja2.Template | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "verify",
        actions: frozenset[str] = _ACTIONS,
    ) -> tuple[_CheckResult | None, str | None]:
        """One checker call under a hard deadline. Returns
        `(check, None)` or `(None, fail_open_reason)`.

        `actions` bounds what the response may name. An out-of-enum
        action fails validation and therefore fails open to `answer` —
        which is what keeps the explore path's restriction real even if
        the vendor ignores the schema."""
        settings = ctx.deps.settings
        prompt = (template or self._template).render(
            evidence=json.dumps(inputs, indent=1, default=str)
        )
        timeout_s = settings.agent_verify_timeout_ms / 1000.0

        def call() -> Any:
            return ctx.deps.openai_client.generate_structured(
                prompt=prompt,
                schema=schema or CHECK_SCHEMA,
                schema_name=schema_name,
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
        check = _validate(result.parsed, actions=actions)
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
        kind: str = "fit",
        demotions: list[str] | None = None,
    ) -> None:
        _LOG.info(
            "verification",
            # Without this the fit checker and the descriptive rubric are
            # indistinguishable in the logs, and the only way to tell
            # them apart was `mode=log` — which works solely because
            # explore happens to be the one gate in shadow. Demote verify
            # for any reason and that proxy silently merges the two.
            kind=kind,
            mode=mode,
            action=action,
            fits=fits,
            enforced=enforced,
            fail_open_reason=fail_open_reason,
            demotions=demotions or [],
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _emit_magnitude(
        self,
        mag: MagnitudeVerdict,
        action: str,
        *,
        events: list[agent_events.AgentEvent],
        enforced: bool,
    ) -> None:
        """Surface the deterministic numeric verdict on the existing
        verification event (fits=False, confidence=1.0 — no model). In
        `log` mode `enforced=False` and the answer is untouched: this is
        the shadow data for the act-flip precision gate."""
        finding = mag.finding
        assert finding is not None
        events.append(
            agent_events.Verification(
                fits=False,
                action=action,
                confidence=1.0,
                reason=f"{finding.tag}: {finding.detail}",
                enforced=enforced,
                kind="magnitude",
            )
        )
        _LOG.info(
            "magnitude",
            tag=finding.tag,
            severity=finding.severity,
            action=action,
            enforced=enforced,
        )


def _validate(
    parsed: dict[str, Any], *, actions: frozenset[str] = _ACTIONS
) -> _CheckResult | None:
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
    if not isinstance(action, str) or action not in actions:
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
