"""The evidence footer on every path that ships a surrender.

The example that motivated this feature carries a `**Partial answer:**`
header, which makes `verify.compose_caveat` look like the obvious hook.
It is not: the caveat composer covers one of the five paths that ship a
surrender. A verify `fits=True` on a no-data answer, verify in shadow
mode, and a budget-forced answer all ship the research model's raw
text — and those are exactly the paths carrying the "check with the
relevant departments" ending the footer exists to follow.

So the footer attaches at the pipeline's composition funnel, and this
file drives one scripted turn down each path to prove all four get the
same footer — plus the two that must not get one.
"""
from __future__ import annotations

import math
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import (
    PipelineOutcome,
    run_turn_collected,
)
from semantic_enrich.core.agent.verify import AnswerFitVerifier
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import BoundedQueryResult, FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

TITLE = "Supplementary Estimates B, 2025-26"
COLUMNS = ["Department", "Vote", "Transportation and communications"]

SURRENDER = (
    "The search did not return any columns specific to air travel "
    "expenditures. Given that, I recommend checking with the relevant "
    "government departments for more detailed disclosures."
)

EXPECTED_FOOTER = (
    "\n\n**What I searched:** *Supplementary Estimates B, 2025-26* "
    "(3 columns)"
    "\n\n**Search terms tried:** \"air travel expenditures\""
)


def _settings(**overrides: Any) -> Settings:
    overrides.setdefault("agent_verify_mode", "act")
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
        **overrides,
    )


def _deps(
    *, settings: Settings, bq: FakeBqClient, openai: FakeOpenAIClient
) -> PipelineDeps:
    return PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-v2-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
        verifier=AnswerFitVerifier.from_settings(settings),
    )


def _unit_vec(_text: str) -> list[float]:
    return [1.0 / math.sqrt(1536)] * 1536


def _bq() -> FakeBqClient:
    """A turn that searched, found a strong candidate, and opened it —
    the state a surrender holds while telling the user nothing."""
    bq = FakeBqClient()
    for _ in range(3):
        bq.register_query(
            "VECTOR_SEARCH",
            [
                {
                    "package_id": "pkg-1",
                    "title": TITLE,
                    "summary": "estimates",
                    "grain": None,
                    "measures": [],
                    "dimensions": [],
                    # Comfortably above the similarity floor: retrieval
                    # was sound, the question still failed.
                    "distance": 0.1,
                }
            ],
        )
    bq.register_query(
        "FROM `proj.raw.documents`",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": TITLE,
                "source_url": "http://x",
                "row_count": 4120,
            }
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING",
        BoundedQueryResult(
            rows=[
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": (
                        '{"Department": "TBS", "Vote": "1", '
                        '"Transportation and communications": "100"}'
                    ),
                }
            ],
            total_bytes_billed=64,
            slot_ms=1,
            elapsed_ms=1,
            timed_out=False,
            error=None,
        ),
    )
    return bq


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def _search_then_list(*tail: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _call("c1", "search_datasets", {"query": "air travel expenditures"}),
        _call("c2", "list_documents", {"package_ids": ["pkg-1"]}),
        *tail,
    ]


def _run(
    *, settings: Settings, script: list[dict[str, Any]], checks: Any = None
) -> PipelineOutcome:
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=script,
        structured_responses=checks,
    )
    return run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="how much did the government spend on air travel?",
        ),
        deps=_deps(settings=settings, bq=_bq(), openai=openai),
    )


def _record(outcome: PipelineOutcome) -> dict[str, Any]:
    events = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(events) == 1
    return events[0].record


def _check(action: str, *, fits: bool = False) -> dict[str, Any]:
    return {
        "fits": fits,
        "confidence": 0.95,
        "gap": "columns specific to air travel expenditures",
        "action": action,
        "retry_hint": None,
    }


# ── the four paths that ship a surrender ──


def test_path_a_verify_caveat() -> None:
    outcome = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("caveat")],
    )
    assert outcome.final_message.startswith("**Partial answer:**")
    assert outcome.final_message.endswith(EXPECTED_FOOTER)
    assert _record(outcome)["outcome"] == "no_data"


def test_path_b_verify_fits_a_no_data_answer() -> None:
    """The checker agrees the surrender is honest, so the candidate
    ships unrewritten — with nothing to tell the user what was searched
    until now."""
    outcome = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("answer", fits=True)],
    )
    assert outcome.final_message == SURRENDER + EXPECTED_FOOTER
    assert _record(outcome)["outcome"] == "no_data"


def test_path_c_verify_shadow_mode() -> None:
    """Shadow mode never alters the answer, so a `clarify` verdict here
    is recorded and ignored — the raw surrender ships."""
    outcome = _run(
        settings=_settings(agent_verify_mode="log"),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("clarify")],
    )
    assert outcome.final_message == SURRENDER + EXPECTED_FOOTER


def test_path_d_budget_forced_answer() -> None:
    """Verify is skipped entirely on a forced answer, so this path never
    reaches any composer in the verify phase."""
    outcome = _run(
        settings=_settings(agent_max_tool_calls=2),
        script=_search_then_list(
            _call("c3", "search_datasets", {"query": "travel costs"}),
            {"content": SURRENDER},
        ),
        checks=None,
    )
    types = [e.event_type for e in outcome.events]
    assert "budget_exceeded" in types
    # The refused search never ran, so it is not evidence.
    assert outcome.final_message == SURRENDER + EXPECTED_FOOTER


def test_all_four_surrender_paths_render_the_same_footer() -> None:
    """The point of hooking the composition funnel rather than the
    caveat composer: one footer, four paths, byte-identical."""
    caveated = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("caveat")],
    )
    fits = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("answer", fits=True)],
    )
    shadow = _run(
        settings=_settings(agent_verify_mode="log"),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("clarify")],
    )
    forced = _run(
        settings=_settings(agent_max_tool_calls=2),
        script=_search_then_list(
            _call("c3", "search_datasets", {"query": "travel costs"}),
            {"content": SURRENDER},
        ),
    )
    footers = {
        o.final_message[o.final_message.index("\n\n**What I searched:**") :]
        for o in (caveated, fits, shadow, forced)
    }
    assert footers == {EXPECTED_FOOTER}


# ── the paths that must stay bare ──


def test_path_e_clarify_carries_no_footer() -> None:
    """A clarify only fires when retrieval was weak. Listing datasets
    under the question would invite the user to pick something the loop
    already scored as irrelevant."""
    outcome = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("clarify")],
    )
    assert outcome.final_message.startswith("I couldn't confidently find")
    assert "**What I searched:**" not in outcome.final_message
    assert _record(outcome)["outcome"] == "clarified"


def test_answered_turn_carries_no_footer() -> None:
    sql = (
        "SELECT SUM(CAST(JSON_VALUE(r.row, "
        "'$.Transportation and communications') AS FLOAT64)) AS total "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 4_200_000.0}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=_search_then_list(
            _call("c3", "run_sql", {"sql": sql, "rationale": "sum"}),
            {"content": "Transportation and communications was $4.2M."},
        ),
        structured_responses=[_check("answer", fits=True)],
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="how much did the government spend on air travel?",
        ),
        deps=_deps(settings=_settings(), bq=bq, openai=openai),
    )
    assert _record(outcome)["outcome"] == "answered"
    assert "**What I searched:**" not in outcome.final_message


# ── kill switch ──


def test_kill_switch_reproduces_the_pre_prd_build() -> None:
    off = _run(
        settings=_settings(agent_evidence_footer=False),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("answer", fits=True)],
    )
    on = _run(
        settings=_settings(),
        script=_search_then_list({"content": SURRENDER}),
        checks=[_check("answer", fits=True)],
    )
    assert off.final_message == SURRENDER
    assert on.final_message == SURRENDER + EXPECTED_FOOTER
    # Event streams differ only in the message payload the footer rides
    # on; every other frame is untouched.
    assert [e.event_type for e in off.events] == [
        e.event_type for e in on.events
    ]
