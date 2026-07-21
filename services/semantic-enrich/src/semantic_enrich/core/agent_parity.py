"""The M5 parity evaluation: three suites, both loop impls, gates.

Suite 1 runs the 20-question retrieval fixture through the live loop
(the regression floor), suite 2 the labeled trace fixture (the
improvement measure), suite 3 scripted multi-turn scenarios with real
history + turn-record carryover (replay, plan reuse, clarify flow,
mid-conversation deflection, stability). The runner is impl-agnostic:
the caller hands it a `run_turn` callable per (impl, run) so v1 and
v2 are measured from the same build against the same warehouse.

Outcomes are derived uniformly from the event stream (v1 emits no
turn records): an `error` event is an error, a successful
`sql_executed` is an answer, anything else is unanswered. v2's richer
record-based outcome is captured alongside for the per-question diff.

Gate arithmetic lives in `compute_gates`; the report is the artifact
the cutover is reviewed on.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semantic_enrich.core import agent_events
from semantic_enrich.core.agent_eval import load_agent_question_set
from semantic_enrich.core.agent_request import ChatRequest
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.agent_parity")

RunTurnFn = Callable[[ChatRequest], Iterator[agent_events.AgentEvent]]

SUITES = ("fixture", "traces", "scenarios")

PROMPT_TOKEN_CAP = 1_300
LATENCY_ALLOWANCE_MS = 1_500


# ── fixtures ──


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    turns: tuple[str, ...]
    expect: dict[str, Any]


class ScenarioSetError(RuntimeError):
    """Scenario fixture load or schema failure. Terminal for the run."""


def load_scenarios(path: Path) -> list[Scenario]:
    if not path.exists():
        raise ScenarioSetError(f"scenario fixture missing: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ScenarioSetError("scenario fixture must be a non-empty list")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ScenarioSetError(f"scenarios[{i}] must be a mapping")
        sid = entry.get("id")
        turns = entry.get("turns")
        if not isinstance(sid, str) or not sid.strip():
            raise ScenarioSetError(f"scenarios[{i}].id must be non-empty")
        if sid in seen:
            raise ScenarioSetError(f"duplicate scenario id {sid!r}")
        seen.add(sid)
        if (
            not isinstance(turns, list)
            or not (2 <= len(turns) <= 5)
            or not all(isinstance(t, str) and t.strip() for t in turns)
        ):
            raise ScenarioSetError(
                f"scenarios[{i}].turns must be 2-5 non-empty strings"
            )
        expect = entry.get("expect") or {}
        if not isinstance(expect, dict):
            raise ScenarioSetError(f"scenarios[{i}].expect must be a mapping")
        scenarios.append(
            Scenario(
                id=sid,
                description=str(entry.get("description", "")),
                turns=tuple(turns),
                expect=expect,
            )
        )
    return scenarios


def load_fixture_questions(path: Path) -> list[tuple[str, str]]:
    """The 4.6 retrieval fixture, reduced to (id, question) pairs for
    the agent-mode regression floor."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ScenarioSetError("fixture must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for i, entry in enumerate(raw):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("question"), str)
        ):
            raise ScenarioSetError(f"fixture[{i}] missing id/question")
        pairs.append((entry["id"], entry["question"]))
    return pairs


# ── one turn, observed ──


def observe_turn(
    *,
    run_turn: RunTurnFn,
    request: ChatRequest,
) -> dict[str, Any]:
    """Drive one turn and fold its event stream into the parity row."""
    started = time.monotonic()
    final_message = ""
    cached = False
    errored = False
    sql_ok = 0
    retrievals = 0
    tool_calls = 0
    dollars = 0.0
    reformulations = 0
    plan_hint = False
    record: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None
    for event in run_turn(request):
        if isinstance(event, agent_events.TurnStart):
            cached = cached or event.cached
        elif isinstance(event, agent_events.MessageDelta):
            final_message += event.delta
        elif isinstance(event, agent_events.SqlExecuted):
            sql_ok += 1
        elif isinstance(event, agent_events.RetrievalStarted):
            retrievals += 1
        elif isinstance(event, agent_events.Reformulation):
            reformulations += 1
        elif isinstance(event, agent_events.PlanHint):
            plan_hint = True
        elif isinstance(event, agent_events.TurnRecordEvent):
            record = event.record
        elif isinstance(event, agent_events.Verification) and (
            event.kind != "magnitude"
        ):
            # Magnitude-gate verdicts (kind="magnitude") are not fit
            # checks and must not seed the parity fits comparison.
            if verify is None:
                verify = {
                    "fits_first": event.fits,
                    "action_first": event.action,
                }
        elif isinstance(event, agent_events.Done):
            tool_calls = event.total_tool_calls
            dollars = event.total_dollars
        elif isinstance(event, agent_events.ErrorEvent):
            errored = True
    if errored:
        uniform = "error"
    elif sql_ok > 0:
        uniform = "answered"
    else:
        uniform = "unanswered"
    return {
        "question": request.question,
        "uniform_outcome": uniform,
        "record_outcome": record.get("outcome") if record else None,
        "final_message": final_message[:300],
        "caveated": final_message.startswith("**Partial answer:**"),
        "cached": cached,
        "tool_calls": tool_calls,
        "retrievals": retrievals,
        "reformulations": reformulations,
        "plan_hint": plan_hint,
        "dollars": dollars,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "verify": verify,
        "record": record,
    }


# ── suites ──


def run_fixture_suite(
    pairs: list[tuple[str, str]], *, run_turn: RunTurnFn, run_tag: str
) -> list[dict[str, Any]]:
    rows = []
    for qid, question in pairs:
        row = observe_turn(
            run_turn=run_turn,
            request=ChatRequest(
                conversation_id=f"parity-{run_tag}-{qid}",
                history=[],
                question=question,
            ),
        )
        row["id"] = qid
        rows.append(row)
        _LOG.info(
            "parity_turn",
            suite="fixture",
            id=qid,
            outcome=row["uniform_outcome"],
        )
    return rows


def run_traces_suite(
    path: Path, *, run_turn: RunTurnFn, run_tag: str
) -> list[dict[str, Any]]:
    rows = []
    for q in load_agent_question_set(path):
        row = observe_turn(
            run_turn=run_turn,
            request=ChatRequest(
                conversation_id=f"parity-{run_tag}-{q.id}",
                history=[],
                question=q.question,
            ),
        )
        row["id"] = q.id
        row["expected_triage"] = q.expected_triage
        row["expected_outcome"] = q.expected_outcome
        row["must_caveat"] = q.must_caveat
        rows.append(row)
        _LOG.info(
            "parity_turn",
            suite="traces",
            id=q.id,
            outcome=row["uniform_outcome"],
        )
    return rows


def run_scenario_suite(
    scenarios: list[Scenario], *, run_turn: RunTurnFn, run_tag: str
) -> list[dict[str, Any]]:
    """Each scenario is one conversation: history and turn records
    carry across turns exactly as a client would echo them."""
    results = []
    for scenario in scenarios:
        history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        turn_rows: list[dict[str, Any]] = []
        for turn_no, question in enumerate(scenario.turns, start=1):
            row = observe_turn(
                run_turn=run_turn,
                request=ChatRequest(
                    conversation_id=f"parity-{run_tag}-{scenario.id}",
                    history=list(history),
                    question=question,
                    turn_records=list(records),
                ),
            )
            row["turn"] = turn_no
            turn_rows.append(row)
            history.append({"role": "user", "content": question})
            history.append(
                {"role": "assistant", "content": row["final_message"]}
            )
            if row["record"] is not None:
                records.append(row["record"])
        results.append(
            {
                "id": scenario.id,
                "expect": scenario.expect,
                "turns": [
                    {k: v for k, v in r.items() if k != "record"}
                    for r in turn_rows
                ],
                "expect_met": _scenario_expectations_met(
                    scenario, turn_rows
                ),
            }
        )
        _LOG.info(
            "parity_scenario",
            id=scenario.id,
            expect_met=results[-1]["expect_met"],
        )
    return results


def _scenario_expectations_met(
    scenario: Scenario, rows: list[dict[str, Any]]
) -> dict[str, bool]:
    met: dict[str, bool] = {}
    cached_expected = scenario.expect.get("cached_turns")
    if isinstance(cached_expected, list):
        met["cached_turns"] = all(
            rows[t - 1]["cached"] for t in cached_expected
        )
    hint_expected = scenario.expect.get("plan_hint_turns")
    if isinstance(hint_expected, list):
        met["plan_hint_turns"] = all(
            rows[t - 1]["plan_hint"] for t in hint_expected
        )
    deflect_expected = scenario.expect.get("deflected_turns")
    if isinstance(deflect_expected, list):
        met["deflected_turns"] = all(
            rows[t - 1]["record_outcome"] == "deflected"
            and rows[t - 1]["retrievals"] == 0
            for t in deflect_expected
        )
    return met


# ── gates ──


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _is_answered(row: dict[str, Any]) -> bool:
    """Cross-impl 'answered': SQL-backed evidence (the only signal v1
    exposes), or a v2 record outcome of answered/answered_with_caveat.
    Caveated answers count — the pipeline's caveat mechanism is
    required behaviour (G4), and a gate that demanded caveats while
    another punished them would be self-contradictory."""
    if row["uniform_outcome"] == "answered":
        return True
    return row.get("record_outcome") in ("answered", "answered_with_caveat")


def _answered_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if _is_answered(r)) / len(rows)


def _per_run_metric(
    runs: list[dict[str, Any]], suite: str, fn: Callable[[list[dict[str, Any]]], float]
) -> float:
    return _median([fn(run[suite]) for run in runs])


def compute_gates(
    *,
    v1_runs: list[dict[str, Any]],
    v2_runs: list[dict[str, Any]],
    v2_prompt_tokens: int,
) -> dict[str, Any]:
    """G1-G7 from the parent PRD, medians across runs."""
    g1_v1 = _per_run_metric(v1_runs, "fixture", _answered_rate)
    g1_v2 = _per_run_metric(v2_runs, "fixture", _answered_rate)

    def in_scope_surrender_rate(rows: list[dict[str, Any]]) -> float:
        in_scope = [
            r for r in rows if r.get("expected_triage") == "in_scope"
        ]
        if not in_scope:
            return 0.0
        surrendered = sum(
            1
            for r in in_scope
            if (r.get("record_outcome") or r["uniform_outcome"])
            in ("no_data", "unanswered")
        )
        return surrendered / len(in_scope)

    g2_v1 = _per_run_metric(v1_runs, "traces", in_scope_surrender_rate)
    g2_v2 = _per_run_metric(v2_runs, "traces", in_scope_surrender_rate)

    def deflection(rows: list[dict[str, Any]]) -> tuple[float, int]:
        labeled = [
            r for r in rows if r.get("expected_triage") != "in_scope"
        ]
        deflected = sum(
            1
            for r in labeled
            if r.get("record_outcome") == "deflected"
            and r["retrievals"] == 0
        )
        false_deflections = sum(
            1
            for r in rows
            if r.get("expected_triage") == "in_scope"
            and r.get("record_outcome") == "deflected"
        )
        rate = deflected / len(labeled) if labeled else 1.0
        return rate, false_deflections

    deflect_rates = [deflection(run["traces"]) for run in v2_runs]
    g3_rate = _median([d[0] for d in deflect_rates])
    g3_false = max(d[1] for d in deflect_rates) if deflect_rates else 0

    def wrong_fit(rows: list[dict[str, Any]]) -> tuple[int, float]:
        uncaveated = 0
        for r in rows:
            if not r.get("must_caveat"):
                continue
            verify = r.get("verify") or {}
            handled = (
                r["caveated"]
                or r.get("record_outcome")
                in ("answered_with_caveat", "clarified", "no_data")
                or verify.get("action_first") in ("caveat", "retry")
            )
            if not handled:
                uncaveated += 1
        clean = [
            r
            for r in rows
            if not r.get("must_caveat")
            and (r.get("record_outcome") or "") == "answered"
            and r.get("verify") is not None
        ]
        if clean:
            precision = sum(
                1 for r in clean if (r["verify"] or {}).get("fits_first")
            ) / len(clean)
        else:
            precision = 1.0
        return uncaveated, precision

    wrong_fits = [wrong_fit(run["traces"]) for run in v2_runs]
    g4_uncaveated = max(w[0] for w in wrong_fits) if wrong_fits else 0
    g4_precision = _median([w[1] for w in wrong_fits])

    def replay_hit(run: dict[str, Any]) -> float:
        for scenario in run["scenarios"]:
            if scenario["id"] == "replay-identical":
                expected = scenario["expect"].get("cached_turns", [])
                if not expected:
                    return 0.0
                hits = sum(
                    1
                    for t in expected
                    if scenario["turns"][t - 1]["cached"]
                )
                return hits / len(expected)
        return 0.0

    g5 = _median([replay_hit(run) for run in v2_runs])

    def cost_per_answered(runs: list[dict[str, Any]]) -> float:
        # Median answered-turn cost (the protocol reports medians): a
        # mean would charge v2 for the expensive tail of hard
        # questions it answers and v1 never does.
        costs = []
        for run in runs:
            answered = [
                r["dollars"]
                for suite in ("fixture", "traces")
                for r in run[suite]
                if _is_answered(r)
            ]
            if answered:
                costs.append(_median(answered))
        return _median(costs)

    g6_v1 = cost_per_answered(v1_runs)
    g6_v2 = cost_per_answered(v2_runs)

    def p50_latency(runs: list[dict[str, Any]]) -> float:
        latencies = [
            float(r["elapsed_ms"])
            for run in runs
            for suite in ("fixture", "traces")
            for r in run[suite]
        ]
        return _median(latencies)

    g7_v1 = p50_latency(v1_runs)
    g7_v2 = p50_latency(v2_runs)

    return {
        "G1_answered_rate": {
            "v1": round(g1_v1, 3),
            "v2": round(g1_v2, 3),
            "pass": g1_v2 >= g1_v1,
        },
        "G2_in_scope_surrender_rate": {
            "v1": round(g2_v1, 3),
            "v2": round(g2_v2, 3),
            "pass": g2_v2 < 0.15,
        },
        "G3_deflection": {
            "labeled_deflected_rate": round(g3_rate, 3),
            "in_scope_false_deflections": g3_false,
            "pass": g3_rate >= 5 / 6 and g3_false == 0,
        },
        "G4_wrong_fit": {
            "uncaveated_wrong_fit": g4_uncaveated,
            "verify_first_check_precision": round(g4_precision, 3),
            "pass": g4_uncaveated == 0 and g4_precision >= 0.95,
        },
        "G5_replay_hit_rate": {"v2": round(g5, 3), "pass": g5 == 1.0},
        "G6_cost": {
            "v1_per_answered": round(g6_v1, 4),
            "v2_per_answered": round(g6_v2, 4),
            "prompt_tokens": v2_prompt_tokens,
            "pass": g6_v2 <= g6_v1
            and v2_prompt_tokens <= PROMPT_TOKEN_CAP,
        },
        "G7_latency_p50_ms": {
            "v1": g7_v1,
            "v2": g7_v2,
            "pass": g7_v2 <= g7_v1 + LATENCY_ALLOWANCE_MS,
        },
    }
