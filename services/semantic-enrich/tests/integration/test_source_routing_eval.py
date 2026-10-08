"""Source-routing regression set (live, opt-in).

Two lists, kept together on purpose. The first held up before routing
existed and must keep landing on the source that answered it; the
second answered badly or not at all, and routing is what fixed it.
Healthcare is pinned because the first cut of routing sent it to
'payments' (itemized health grants) and broke a good answer.

The end-to-end version of this check (full turns, real answers) costs
about $0.08 a question and is run by hand; this one checks only the
triage call: fifteen mini-model calls, well under a cent.
"""

from __future__ import annotations

import os

import pytest

from semantic_enrich.clients.openai import RealOpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.triage import (
    CLASSIFIER_SCHEMA,
    load_triage_template,
)

pytestmark = pytest.mark.live

# Already answered well before routing: must not move.
STAY_GREAT: list[tuple[str, str]] = [
    ("What percentage of Canadian exports go to the United States?", "statcan"),
    ("How much revenue does Canada collect from tariffs?", "statcan"),
    ("Has Canada's GDP per capita actually grown over the last decade?", "statcan"),
    ("How much money does the federal government spend on healthcare?", "statcan"),
    ("Which provinces are building the most housing per capita?", "statcan"),
    ("how much has canadian federal government donated to ukraine in the last year", "payments"),
    ("How much has the average Canadian household's purchasing power changed since 2015?", "statcan"),
    ("How has immigration changed Canada's population growth over the last 10 years?", "statcan"),
]

# Answered badly before routing (wrong source, refused, or wasted calls).
GET_BETTER: list[tuple[str, str]] = [
    ("How much did federal departments spend on travel in 2024?", "payments"),
    ("What is the unemployment rate in Alberta right now?", "statcan"),
    ("Which companies received the largest federal contracts in 2025?", "payments"),
    ("How much has rent increased in Canada since 2020?", "statcan"),
    ("How much did Canada give to the United Nations in grants and contributions in 2024-25?", "payments"),
    ("Who owns Canada's government debt?", "statcan"),
    ("How much has grocery inflation contributed to the increase in household expenses?", "statcan"),
]


@pytest.mark.skipif(
    not os.environ.get("WHENRICH_RUN_LIVE_EVALS"),
    reason="live vendor eval; set WHENRICH_RUN_LIVE_EVALS=1 to run",
)
@pytest.mark.parametrize(("question", "expected"), STAY_GREAT + GET_BETTER, ids=lambda v: str(v)[:40])
def test_question_routes_to_its_source(question: str, expected: str) -> None:
    settings = Settings()
    api_key = settings.openai_api_key
    if api_key is None:
        pytest.skip("WHENRICH_OPENAI_API_KEY not configured")
    client = RealOpenAIClient(
        api_key=api_key.get_secret_value(),
        embedding_model=settings.openai_embedding_model,
        request_timeout_s=settings.openai_request_timeout_s,
        max_retries=settings.openai_max_retries,
    )
    result = client.generate_structured(
        prompt=load_triage_template(settings).render(question=question, context_hint=None),
        schema=CLASSIFIER_SCHEMA,
        schema_name="triage",
        model=settings.agent_triage_model,
        temperature=0.0,
        max_tokens=300,
        timeout_s=25,
    )
    parsed = result.parsed
    assert parsed["category"] == "in_scope", (question, parsed.get("reason"))
    assert parsed["source"] == expected, (question, parsed.get("reason"))
    assert parsed["source_confidence"] >= settings.agent_route_min_confidence
