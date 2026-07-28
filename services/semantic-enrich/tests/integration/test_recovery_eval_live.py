"""The guided-recovery lock — live tier (gated, ~$1).

Covers only what fakes structurally cannot: whether triage classifies
the boundary questions correctly, and whether the research model
produces a genuine description on a scoped turn rather than a refusal.
Everything assertable on shape lives in the free tier, which is where
the milestone's actual regression protection sits.

Also writes `eval/reports/guided-recovery-eval.json` carrying the
act-flip criteria for `agent_verify_explore_mode`, so the go/no-go is a
lookup rather than a re-derivation months from now.

    WHENRICH_RUN_LIVE_EVALS=1 uv run pytest -m live \\
        tests/integration/test_recovery_eval_live.py
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent import recovery_eval
from semantic_enrich.core.agent.recovery_eval import CASES, RecoveryCase
from semantic_enrich.core.agent_dispatch import (
    build_loop_handle,
    resolve_run_turn,
)
from semantic_enrich.core.agent_request import ChatRequest

pytestmark = pytest.mark.live

# Act-flip gate for `agent_verify_explore_mode`, recorded in the report
# so the decision is a lookup. The false-positive bar is zero rather
# than a percentage on purpose: a rubric that caveats faithful
# descriptions is strictly worse than the bypass it replaces, and the
# bypass is free to keep indefinitely.
ACT_FLIP_CRITERIA: dict[str, Any] = {
    "min_shadow_explore_turns": 30,
    "max_false_positive_caveats": 0,
    "min_true_positive_caveats": 1,
    "max_p95_added_latency_ms": 400,
    "no_go_behaviour": (
        "stay in log; the scoped-turn bypass remains enforced and is "
        "safe indefinitely — nothing in the milestone depends on the flip"
    ),
}


def _live_settings() -> Settings:
    settings = Settings()
    if settings.openai_api_key is None:
        pytest.skip("WHENRICH_OPENAI_API_KEY not configured")
    if not settings.gcp_project_id:
        pytest.skip("WHENRICH_GCP_PROJECT_ID not configured")
    return settings


def _observe(events: list[agent_events.AgentEvent]) -> recovery_eval.Observation:
    message = "".join(
        e.delta for e in events if isinstance(e, agent_events.MessageDelta)
    )
    record = next(
        (
            e.record
            for e in events
            if isinstance(e, agent_events.TurnRecordEvent)
        ),
        {},
    )
    return recovery_eval.Observation(
        outcome=str(record.get("outcome", "unknown")),
        message=message,
        suggestion_count=sum(
            len(e.items)
            for e in events
            if isinstance(e, agent_events.Suggestions)
        ),
        derivation_count=sum(
            1
            for e in events
            if isinstance(e, agent_events.DerivationEvent)
        ),
        event_types=tuple(e.event_type for e in events),
    )


def _run_live(case: RecoveryCase) -> recovery_eval.Observation:
    settings = _live_settings()
    handle = build_loop_handle(
        settings=settings,
        bq=_real_bq(settings),
        openai_client=_real_openai(settings),
        loop_impl="v2",
    )
    request = ChatRequest(
        conversation_id=f"recovery-eval-{case.id}",
        history=[],
        question=case.question,
        scope_package_ids=case.scope_package_ids,
        turn_records=[
            {"v": 1, "outcome": o} for o in case.prior_outcomes
        ],
    )
    events = list(
        resolve_run_turn("v2")(request=request, deps=handle.deps)
    )
    return _observe(events)


def _real_bq(settings: Settings) -> Any:
    from semantic_enrich.clients.bq import RealBqClient

    return RealBqClient.for_project(settings.gcp_project_id)


def _real_openai(settings: Settings) -> Any:
    from semantic_enrich.clients.openai import RealOpenAIClient

    assert settings.openai_api_key is not None
    return RealOpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        embedding_model=settings.openai_embedding_model,
        request_timeout_s=settings.openai_request_timeout_s,
        max_retries=settings.openai_max_retries,
    )


@pytest.mark.skipif(
    not os.environ.get("WHENRICH_RUN_LIVE_EVALS"),
    reason="live vendor eval; set WHENRICH_RUN_LIVE_EVALS=1 to run",
)
def test_live_tier_and_report(tmp_path: Path) -> None:
    """Runs every case live, grades it, and writes the report.

    Graded leniently on *outcome* for the retrieval-dependent cases:
    a question that fails here for vocabulary reasons is a **correct**
    outcome for this milestone as long as it fails informatively, and
    conflating that with a guided-recovery regression would make this
    fixture fail for another milestone's reasons.
    """
    settings = _live_settings()
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []

    for case in CASES:
        observed = _run_live(case)
        result = recovery_eval.grade(case, observed)
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "passed": result.passed,
                "failures": list(result.failures),
                "observed": {
                    "outcome": observed.outcome,
                    "suggestions": observed.suggestion_count,
                    "derivations": observed.derivation_count,
                    "footer": recovery_eval.FOOTER_MARKER
                    in observed.message,
                },
                "message": observed.message[:600],
            }
        )
        hard_failures.extend(_invariant_failures(case, observed))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "loop_impl": "v2",
        "verify_explore_mode": settings.agent_verify_explore_mode,
        "act_flip_criteria": ACT_FLIP_CRITERIA,
        "cases": rows,
        "summary": {
            "total": len(rows),
            "passed": sum(1 for r in rows if r["passed"]),
        },
    }
    out = settings.eval_reports_dir / "guided-recovery-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )

    assert not hard_failures, "; ".join(hard_failures)


def _invariant_failures(
    case: RecoveryCase, observed: recovery_eval.Observation
) -> list[str]:
    """The assertions that hold regardless of what retrieval did.

    The distinction matters, and the first run of this tier got it
    wrong. `expect_derivation` is *retrieval-dependent* — it says "this
    question should be answerable", and when the corpus cannot answer
    it there is no derivation and no regression: the loop surrendered
    honestly, which is a correct outcome for this milestone. Asserting
    it absolutely made the lock fail for the corpus's reasons rather
    than for guided recovery's.

    The real M6 invariant is *conditional*: an answer must be traced.
    Not "this question must produce an answer."
    """
    failures: list[str] = []

    if case.forbid_clarify_replacement and observed.message.startswith(
        recovery_eval.CLARIFY_MARKER
    ):
        failures.append(f"{case.id}: a substantive answer was replaced")

    if observed.outcome == "answered" and observed.derivation_count == 0:
        failures.append(f"{case.id}: answered with no derivation (M6)")

    # The false-invitation guard, also conditional: a turn that never
    # cleared the retrieval floor must offer nothing.
    if (
        observed.outcome in ("clarified", "deflected")
        and observed.suggestion_count > 0
    ):
        failures.append(
            f"{case.id}: offered {observed.suggestion_count} suggestions "
            "on a turn with nothing worth exploring"
        )

    if observed.suggestion_count > 3:
        failures.append(
            f"{case.id}: {observed.suggestion_count} suggestions exceeds "
            "the cap"
        )
    return failures


def test_act_flip_criteria_are_recorded_not_folklore() -> None:
    """Free — no vendor call. Pins that the gate exists and that its
    false-positive bar is zero, so nobody softens it to a percentage
    without the diff being visible."""
    assert ACT_FLIP_CRITERIA["max_false_positive_caveats"] == 0
    assert ACT_FLIP_CRITERIA["min_true_positive_caveats"] >= 1
    assert ACT_FLIP_CRITERIA["min_shadow_explore_turns"] >= 30
    assert "stay in log" in ACT_FLIP_CRITERIA["no_go_behaviour"]
