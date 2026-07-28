"""Calibration probes for the descriptive rubric (live, opt-in).

The rubric was tightened hard after its first shadow run produced five
false caveats out of six unfit verdicts. Tightening a checker until it
stops complaining is easy and worthless — the failure mode of the fix
is a rubric that fits everything, and a shadow run showing zero unfit
verdicts cannot tell the two apart.

So these are the negative controls. Three answers that must FIT
(including the two exact shapes the first run got wrong) and two that
must not. Cheap enough to re-run whenever the prompt is touched: five
mini-model calls, well under a cent, no BigQuery.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from semantic_enrich.clients.openai import RealOpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.verify import (
    EXPLORE_CHECK_SCHEMA,
    load_verify_explore_template,
)

pytestmark = pytest.mark.live

_BASE: dict[str, Any] = {
    "question": "what's in the travel expenses data?",
    "packages_described": ["pkg-travel"],
    "columns_surfaced": 12,
    "documents_listed": 3,
    "tools_used": ["list_documents", "search_datasets"],
    "claims_a_total": False,
}

_MUST_FIT: list[tuple[str, dict[str, Any]]] = [
    (
        "faithful description",
        {
            "candidate_answer": (
                "The [Travel Expenses](/datasets/pkg-travel) dataset covers "
                "2010-2024 with columns for traveller name, purpose, "
                "airfare, lodging, and total cost."
            )
        },
    ),
    (
        # First-run failure: flagged `empty non-description` on an
        # answer that plainly described columns.
        "partial column list",
        {
            "candidate_answer": (
                "The [Travel Expenses](/datasets/pkg-travel) dataset "
                "includes some key columns: `ref_number`, "
                "`traveller_name`, `airfare`. There are others."
            )
        },
    ),
    (
        # First-run failure: a `$` inside a dataset *title* read as a
        # claim. `claims_a_total` now strips citations before looking.
        "dollar amount inside a citation title",
        {
            "candidate_answer": (
                "Includes [Projects worth $20 million and over]"
                "(/datasets/pkg-travel), covering project name, region, "
                "and status."
            )
        },
    ),
]

_MUST_NOT_FIT: list[tuple[str, dict[str, Any]]] = [
    (
        # The numeric-trust boundary, and the condition that actually
        # matters: a description may not assert a computed figure.
        "asserts an ungrounded total",
        {
            "candidate_answer": (
                "The [Travel Expenses](/datasets/pkg-travel) dataset covers "
                "2010-2024. Federal officials spent $847.3M on travel over "
                "that period."
            ),
            "claims_a_total": True,
        },
    ),
    (
        "describes a dataset outside an explicit scope",
        {
            "candidate_answer": (
                "The [Fisheries Landings](/datasets/pkg-fish) dataset "
                "records commercial fish catch volumes by port and species."
            ),
            "scope_packages": [
                {
                    "package_id": "pkg-travel",
                    "title": "Proactive Disclosure - Travel Expenses",
                }
            ],
        },
    ),
]


def _judge(evidence: dict[str, Any]) -> dict[str, Any]:
    settings = Settings()
    api_key = settings.openai_api_key
    if api_key is None:
        pytest.skip("WHENRICH_OPENAI_API_KEY not configured")
    template = load_verify_explore_template(settings)
    assert template is not None
    client = RealOpenAIClient(
        api_key=api_key.get_secret_value(),
        embedding_model=settings.openai_embedding_model,
        request_timeout_s=settings.openai_request_timeout_s,
        max_retries=settings.openai_max_retries,
    )
    result = client.generate_structured(
        prompt=template.render(
            evidence=json.dumps({**_BASE, **evidence}, indent=1)
        ),
        schema=EXPLORE_CHECK_SCHEMA,
        schema_name="verify_explore",
        model=settings.agent_verify_model,
        temperature=0.0,
        max_tokens=300,
        timeout_s=25,
    )
    return dict(result.parsed)


@pytest.mark.skipif(
    not os.environ.get("WHENRICH_RUN_LIVE_EVALS"),
    reason="live vendor eval; set WHENRICH_RUN_LIVE_EVALS=1 to run",
)
@pytest.mark.parametrize(
    ("label", "evidence"), _MUST_FIT, ids=[c[0] for c in _MUST_FIT]
)
def test_faithful_descriptions_are_not_caveated(
    label: str, evidence: dict[str, Any]
) -> None:
    """A false caveat on a good description is the expensive error: it
    teaches users to distrust descriptions that are fine."""
    verdict = _judge(evidence)
    assert verdict["fits"] is True, (label, verdict.get("gap"))
    assert verdict["action"] == "answer"


@pytest.mark.skipif(
    not os.environ.get("WHENRICH_RUN_LIVE_EVALS"),
    reason="live vendor eval; set WHENRICH_RUN_LIVE_EVALS=1 to run",
)
@pytest.mark.parametrize(
    ("label", "evidence"),
    _MUST_NOT_FIT,
    ids=[c[0] for c in _MUST_NOT_FIT],
)
def test_the_rubric_still_catches_real_failures(
    label: str, evidence: dict[str, Any]
) -> None:
    """The control on the control. Without this, a rubric tuned into
    silence would pass every other check in this file."""
    verdict = _judge(evidence)
    assert verdict["fits"] is False, label
    assert verdict["action"] == "caveat"
    assert verdict["gap"]
