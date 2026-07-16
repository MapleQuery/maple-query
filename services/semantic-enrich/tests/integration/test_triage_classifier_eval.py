"""Live classifier eval against the trace-fixture labels.

Hits the real vendor (network + cost ≈ a cent), so it is marked
`live`, gated on an explicit env opt-in, and excluded from CI. Run
manually:

    WHENRICH_RUN_LIVE_EVALS=1 uv run pytest \
        tests/integration/test_triage_classifier_eval.py -m live -s

Gates:
- Precision (blocking): zero questions labeled `in_scope` may be
  classified as anything else at the confidence threshold — a false
  deflection is strictly worse than the wasted research it replaces.
- Recall (non-blocking, reported): deflectable turns actually
  classified off_scope/meta at the threshold.

The confusion table is written to
`eval/reports/triage-classifier-eval.json` for the record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from semantic_enrich.clients.openai import RealOpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.triage import (
    CLASSIFIER_SCHEMA,
    load_triage_template,
)
from semantic_enrich.core.agent_eval import load_agent_question_set

pytestmark = pytest.mark.live

_TRACE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "questions-agent-traces.yaml"
)


@pytest.mark.skipif(
    not os.environ.get("WHENRICH_RUN_LIVE_EVALS"),
    reason="live vendor eval; set WHENRICH_RUN_LIVE_EVALS=1 to run",
)
def test_classifier_precision_and_recall_on_the_fixture() -> None:
    settings = Settings()
    api_key = settings.openai_api_key
    if api_key is None:
        pytest.skip("WHENRICH_OPENAI_API_KEY not configured")

    client = RealOpenAIClient.for_settings(
        api_key=api_key.get_secret_value(),
        embedding_model=settings.openai_embedding_model,
        request_timeout_s=settings.openai_request_timeout_s,
        max_retries=settings.openai_max_retries,
    )
    template = load_triage_template(settings)
    questions = load_agent_question_set(_TRACE_FIXTURE)
    threshold = settings.agent_triage_min_confidence

    rows: list[dict[str, Any]] = []
    false_deflections: list[dict[str, Any]] = []
    deflectable = 0
    deflected = 0
    for q in questions:
        result = client.generate_structured(
            prompt=template.render(question=q.question, context_hint=None),
            schema=CLASSIFIER_SCHEMA,
            schema_name="triage",
            model=settings.agent_triage_model,
            temperature=0.0,
            max_tokens=150,
        )
        category = str(result.parsed.get("category"))
        confidence = float(result.parsed.get("confidence") or 0.0)
        # Mirror the runtime confidence gate: a low-confidence verdict
        # fails open, so it neither deflects nor counts against
        # precision.
        acted = category if confidence >= threshold else "in_scope"
        row = {
            "id": q.id,
            "question": q.question,
            "expected": q.expected_triage,
            "classified": category,
            "confidence": round(confidence, 3),
            "acted": acted,
            "reason": result.parsed.get("reason"),
        }
        rows.append(row)
        if q.expected_triage == "in_scope" and acted != "in_scope":
            false_deflections.append(row)
        if q.expected_triage in ("off_scope", "meta"):
            deflectable += 1
            if acted in ("off_scope", "meta"):
                deflected += 1

    report = {
        "model": settings.agent_triage_model,
        "confidence_threshold": threshold,
        "questions": len(rows),
        "false_deflections": false_deflections,
        "deflectable": deflectable,
        "deflected": deflected,
        "rows": rows,
    }
    report_path = (
        settings.eval_reports_dir / "triage-classifier-eval.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\ntriage eval: {len(rows)} questions, "
        f"{len(false_deflections)} false deflections, "
        f"recall {deflected}/{deflectable} → {report_path}"
    )

    # Blocking precision gate.
    assert false_deflections == [], (
        "in_scope questions would be deflected at the threshold: "
        f"{false_deflections}"
    )
    # Recall is a target, not a gate — reported above and in the JSON.
