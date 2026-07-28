"""Scoped turns through the pipeline: the bypass, and its limit.

The bypass keys on `scoped AND no successful SQL`, never on `scoped`
alone. That distinction is the whole PRD: a numeric follow-up ("now sum
these columns") is also a scoped turn, so keying on scope would strip
the derivation/grounding/magnitude protection from exactly the answers
guided recovery exists to produce.

The headline case is the regression that motivates it — a descriptive
scoped turn whose fit-checker *would* have said `clarify` ships its
summary unchanged, because it never reaches the checker at all.
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

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
TITLE = "Supplementary Estimates B, 2025-26"

SUMMARY = (
    "This dataset covers departmental spending authorities for 2025-26. "
    "It has 3 columns — Department, Vote, and Transportation and "
    "communications — across 4,120 rows, with amounts running from "
    "$1,000 to $2.5M."
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
    bq = FakeBqClient()
    bq.register_query(
        "FROM `proj.raw.documents`",
        [
            {
                "document_id": "doc-1",
                "package_id": PKG,
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


def _call(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def _clarify_check() -> dict[str, Any]:
    """The verdict that would replace a summary with a question."""
    return {
        "fits": False,
        "confidence": 0.95,
        "gap": "which program you mean",
        "action": "clarify",
        "retry_hint": None,
    }


def _run(
    *,
    settings: Settings,
    script: list[dict[str, Any]],
    scope: tuple[str, ...] = (PKG,),
    checks: Any = None,
    bq: FakeBqClient | None = None,
) -> tuple[PipelineOutcome, FakeOpenAIClient]:
    openai = FakeOpenAIClient(
        vector_factory=_unit_vec,
        chat_script=script,
        structured_responses=checks,
    )
    outcome = run_turn_collected(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="summarize this dataset",
            scope_package_ids=scope,
        ),
        deps=_deps(settings=settings, bq=bq or _bq(), openai=openai),
    )
    return outcome, openai


def _record(outcome: PipelineOutcome) -> dict[str, Any]:
    events = [
        e
        for e in outcome.events
        if isinstance(e, agent_events.TurnRecordEvent)
    ]
    assert len(events) == 1
    return events[0].record


def _descriptive_script() -> list[dict[str, Any]]:
    return [
        _call("c1", "list_documents", {"package_ids": [PKG]}),
        {"content": SUMMARY},
    ]


# ── the scope reaches the model, and the tools accept it ──


def test_scoped_turn_lists_documents_without_searching_first() -> None:
    """§3.1, and the assertion is written for a *silent* failure.

    `known_package_ids` is empty until `search_datasets` runs, and
    `list_documents` rejects unknown ids. Without the whitelist
    admission the model obeys the hint, takes a tool error, and falls
    back to a broad search — the turn still completes and still ships an
    answer, so only the absence of `tool_error` catches it.
    """
    outcome, openai = _run(
        settings=_settings(), script=_descriptive_script()
    )
    kinds = [e.event_type for e in outcome.events]
    assert "tool_error" not in kinds
    assert "documents_listed" in kinds
    # No search was needed, and none was made.
    assert "retrieval_started" not in kinds
    # The scope reached the model as a system hint naming the package.
    hints = [
        m
        for m in openai.chat_calls[0]["messages"]
        if m.get("role") == "system"
        and "exploring these datasets specifically" in str(m.get("content"))
    ]
    assert len(hints) == 1
    assert PKG in str(hints[0]["content"])


# ── the §4 table ──


def test_descriptive_turn_never_reaches_the_numeric_fit_checker() -> None:
    """The fit prompt is calibrated against answers that ran SQL; a
    description must not be judged by it. With the explore rubric in its
    default shadow mode a checker *does* run — the descriptive one."""
    outcome, openai = _run(
        settings=_settings(), script=_descriptive_script()
    )
    assert [c["schema_name"] for c in openai.structured_calls] == [
        "verify_explore"
    ]
    # Grounding is skipped regardless of the rubric: a summary quoting a
    # monetary range would ground as `ungrounded` and paint a false "no
    # computation behind this number" panel on a turn that claimed no
    # total.
    assert not any(
        isinstance(e, agent_events.DerivationEvent) for e in outcome.events
    )
    assert outcome.final_message == SUMMARY
    assert _record(outcome)["outcome"] == "explored"


def test_full_bypass_when_the_explore_rubric_is_off() -> None:
    """`off` is the interim posture the rubric replaced: no checker call
    at all on a descriptive turn."""
    outcome, openai = _run(
        settings=_settings(agent_verify_explore_mode="off"),
        script=_descriptive_script(),
        checks=[_clarify_check()],
    )
    assert openai.structured_calls == []
    assert not any(
        isinstance(e, agent_events.Verification) for e in outcome.events
    )
    assert outcome.final_message == SUMMARY
    assert _record(outcome)["outcome"] == "explored"


def test_the_clarify_regression_that_motivates_the_prd() -> None:
    """A checker that *would* have replaced the summary with a question
    cannot. This is the failure the milestone exists to close: the user
    clicks "summarize this dataset" and is asked a question back.

    Two independent guards hold here, and the test exercises both. The
    descriptive turn never reaches the fit checker; and the checker it
    *does* reach has no `clarify` in its action enum, so a response
    naming one fails validation and fails open to `answer`. The second
    guard is what makes this safe even against a miscalibrated rubric.
    """
    outcome, openai = _run(
        settings=_settings(agent_verify_explore_mode="act"),
        script=_descriptive_script(),
        checks=[_clarify_check()],
    )
    assert outcome.final_message == SUMMARY
    assert "couldn't confidently find" not in outcome.final_message
    # The verdict was consumed by the explore checker and rejected —
    # not merely skipped. Even in `act` mode it changed nothing.
    assert openai.structured_responses == []
    assert [c["schema_name"] for c in openai.structured_calls] == [
        "verify_explore"
    ]
    assert _record(outcome)["outcome"] == "explored"


def test_scoped_turn_whose_sql_errored_is_still_explored() -> None:
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[],
        total_bytes_billed=0,
        slot_ms=0,
        elapsed_ms=1,
        timed_out=False,
        error="syntax error",
    )
    sql = (
        "SELECT SUM(CAST(JSON_VALUE(r.row, '$.Vote') AS FLOAT64)) AS total "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    outcome, _ = _run(
        settings=_settings(),
        bq=bq,
        script=[
            _call("c1", "list_documents", {"package_ids": [PKG]}),
            _call("c2", "run_sql", {"sql": sql, "rationale": "sum"}),
            {"content": SUMMARY},
        ],
    )
    assert _record(outcome)["outcome"] == "explored"


def test_numeric_scoped_follow_up_keeps_the_full_path() -> None:
    """The non-regression that decides the PRD. A scoped turn that ran
    SQL is a numeric answer, not an exploration: verify runs, grounding
    runs, the derivation event ships, and the record says `answered`."""
    bq = _bq()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 4_200_000.0}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=5,
        timed_out=False,
        error=None,
    )
    sql = (
        "SELECT SUM(CAST(JSON_VALUE(r.row, "
        "'$.Transportation and communications') AS FLOAT64)) AS total "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    outcome, openai = _run(
        settings=_settings(),
        bq=bq,
        script=[
            _call("c1", "list_documents", {"package_ids": [PKG]}),
            _call("c2", "run_sql", {"sql": sql, "rationale": "sum"}),
            {"content": "Transportation and communications was $4.2M."},
        ],
        checks=[
            {
                "fits": True,
                "confidence": 0.95,
                "gap": None,
                "action": "answer",
                "retry_hint": None,
            }
        ],
    )
    assert len(openai.structured_calls) == 1  # verify ran
    assert any(
        isinstance(e, agent_events.DerivationEvent) for e in outcome.events
    )
    assert _record(outcome)["outcome"] == "answered"


# ── degradation ──


def test_malformed_scope_degrades_to_an_ordinary_turn() -> None:
    outcome, openai = _run(
        settings=_settings(),
        scope=("nope", "!!"),
        script=[
            _call("c1", "search_datasets", {"query": "estimates"}),
            {"content": "I could not find a match."},
        ],
        checks=[
            {
                "fits": True,
                "confidence": 0.9,
                "gap": None,
                "action": "answer",
                "retry_hint": None,
            }
        ],
    )
    # No hint, no bypass: verify ran and the turn is a plain surrender.
    assert len(openai.structured_calls) == 1
    assert not any(
        "exploring these datasets" in str(m.get("content"))
        for m in openai.chat_calls[0]["messages"]
    )
    assert _record(outcome)["outcome"] == "no_data"


def test_kill_switch_reproduces_the_unscoped_build() -> None:
    off, openai_off = _run(
        settings=_settings(agent_scoped_turns=False),
        script=_descriptive_script(),
        checks=[_clarify_check()],
    )
    # With scoping off the hint is absent, so `list_documents` on an
    # unsearched package is rejected — which is the pre-PRD behaviour,
    # and exactly why §3.1's admission is not optional.
    assert "tool_error" in [e.event_type for e in off.events]
    assert openai_off.structured_calls != []
    assert _record(off)["outcome"] != "explored"
