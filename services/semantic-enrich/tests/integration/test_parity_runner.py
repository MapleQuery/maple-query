"""The parity harness itself: fixture lint, suite runners with fakes,
and gate arithmetic. Live parity runs are manual; this keeps the
runner CI-covered so a broken harness can't block the cutover."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_parity
from semantic_enrich.core.agent.memory import ReplayCacheV2, SessionMemory
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn
from semantic_enrich.core.agent_cache import ResponseCache
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

SCENARIOS_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "scenarios-multiturn.yaml"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "questions.yaml"
)


def test_committed_scenarios_lint() -> None:
    scenarios = agent_parity.load_scenarios(SCENARIOS_PATH)
    assert len(scenarios) == 5
    ids = {s.id for s in scenarios}
    assert "replay-identical" in ids
    assert "plan-reuse" in ids


def test_committed_fixture_loads_as_pairs() -> None:
    pairs = agent_parity.load_fixture_questions(FIXTURE_PATH)
    assert len(pairs) == 20
    assert all(q for _id, q in pairs)


def test_scenario_lint_rejects_bad_shapes(tmp_path: Path) -> None:
    bad = tmp_path / "scenarios.yaml"
    bad.write_text("- id: x\n  turns: ['only one']\n")
    with pytest.raises(agent_parity.ScenarioSetError):
        agent_parity.load_scenarios(bad)
    bad.write_text("[]")
    with pytest.raises(agent_parity.ScenarioSetError):
        agent_parity.load_scenarios(bad)


def _run_turn_fn() -> Any:
    """A v2 pipeline over fakes: every question gets the fake's canned
    'no scripted response' final message (no tools)."""
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=Settings(
            gcp_project_id="proj",
            openai_api_key="sk-test",  # type: ignore[arg-type]
            agent_cache_replay_delay_ms=0,
        ),
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
        memory=SessionMemory(
            cache=ReplayCacheV2(max_entries=100, ttl_seconds=600)
        ),
    )
    return lambda request: run_turn(request=request, deps=deps)


def test_suites_run_dry_and_report_shape_holds() -> None:
    run_turn_fn = _run_turn_fn()
    fixture_rows = agent_parity.run_fixture_suite(
        [("q1", "housing spend?"), ("q2", "airfare spend?")],
        run_turn=run_turn_fn,
        run_tag="test",
    )
    assert [r["id"] for r in fixture_rows] == ["q1", "q2"]
    assert all(r["uniform_outcome"] == "unanswered" for r in fixture_rows)

    scenario_rows = agent_parity.run_scenario_suite(
        agent_parity.load_scenarios(SCENARIOS_PATH),
        run_turn=run_turn_fn,
        run_tag="test",
    )
    assert len(scenario_rows) == 5
    replay = next(
        s for s in scenario_rows if s["id"] == "replay-identical"
    )
    # Canned answers carry no SQL → outcome no_data → never cached, so
    # the dry harness reports the expectation unmet rather than lying.
    assert replay["expect_met"] == {"cached_turns": False}


def _row(
    *,
    outcome: str = "answered",
    record_outcome: str | None = "answered",
    dollars: float = 0.05,
    elapsed_ms: int = 5000,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "question": "q",
        "uniform_outcome": outcome,
        "record_outcome": record_outcome,
        "final_message": "m",
        "caveated": False,
        "cached": False,
        "tool_calls": 3,
        "retrievals": 1,
        "reformulations": 0,
        "plan_hint": False,
        "dollars": dollars,
        "elapsed_ms": elapsed_ms,
        "verify": {"fits_first": True, "action_first": "answer"},
        "record": None,
        **extra,
    }


def _mk_run(
    *,
    fixture_answered: int,
    traces: list[dict[str, Any]],
    replay_cached: bool,
) -> dict[str, Any]:
    fixture = [
        _row()
        if i < fixture_answered
        else _row(outcome="unanswered", record_outcome="no_data")
        for i in range(10)
    ]
    scenarios = [
        {
            "id": "replay-identical",
            "expect": {"cached_turns": [2, 3]},
            "turns": [
                {"cached": False},
                {"cached": replay_cached},
                {"cached": replay_cached},
            ],
            "expect_met": {"cached_turns": replay_cached},
        }
    ]
    return {"fixture": fixture, "traces": traces, "scenarios": scenarios}


def test_gates_pass_on_a_clean_v2() -> None:
    traces_v2 = [
        _row(expected_triage="in_scope", must_caveat=False),
        _row(expected_triage="in_scope", must_caveat=False),
        _row(
            expected_triage="in_scope",
            must_caveat=True,
            record_outcome="answered_with_caveat",
            caveated=True,
        ),
        _row(
            expected_triage="off_scope",
            outcome="unanswered",
            record_outcome="deflected",
            retrievals=0,
        ),
    ]
    traces_v1 = [
        _row(expected_triage="in_scope", record_outcome=None),
        _row(
            expected_triage="in_scope",
            outcome="unanswered",
            record_outcome=None,
        ),
        _row(expected_triage="in_scope", record_outcome=None),
        _row(expected_triage="off_scope", record_outcome=None),
    ]
    v1_runs = [
        _mk_run(fixture_answered=6, traces=traces_v1, replay_cached=False)
    ]
    v2_runs = [
        _mk_run(fixture_answered=7, traces=traces_v2, replay_cached=True)
    ]
    gates = agent_parity.compute_gates(
        v1_runs=v1_runs, v2_runs=v2_runs, v2_prompt_tokens=900
    )
    assert all(g["pass"] for g in gates.values()), gates


def test_gates_fail_on_regressions() -> None:
    traces = [
        _row(
            expected_triage="in_scope",
            must_caveat=False,
            outcome="unanswered",
            record_outcome="no_data",
        )
    ]
    v1_runs = [
        _mk_run(fixture_answered=8, traces=traces, replay_cached=False)
    ]
    v2_runs = [
        _mk_run(
            fixture_answered=5,
            traces=traces,
            replay_cached=False,
        )
    ]
    gates = agent_parity.compute_gates(
        v1_runs=v1_runs, v2_runs=v2_runs, v2_prompt_tokens=2000
    )
    assert gates["G1_answered_rate"]["pass"] is False
    assert gates["G2_in_scope_surrender_rate"]["pass"] is False
    assert gates["G5_replay_hit_rate"]["pass"] is False
    assert gates["G6_cost"]["pass"] is False
