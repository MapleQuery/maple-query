"""person_record: flag-gated, fixed parameterised queries over curated,
per-return totals, and no change at all to the agent when the flag is off."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools


class FakeBq:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, list[Any]]] = []

    def query_rows(self, sql: str, *, params: Any = ()) -> Any:
        if self.fail:
            raise RuntimeError("Not found: Table proj:curated.people")
        self.calls.append((sql, list(params)))
        if "`proj.curated.people`" in sql:
            return iter(
                [
                    {
                        "person_id": "op:x",
                        "name": "X Y",
                        "current_mp": True,
                        "first_term_start": date(2015, 11, 3),
                        "last_term_end": None,
                        "openparliament_url": "https://openparliament.ca/politicians/x/",
                    }
                ]
            )
        if "person_terms" in sql:
            return iter(
                [
                    {
                        "start_date": date(2015, 11, 3),
                        "end_date": None,
                        "party": "Liberal",
                        "riding_name": "R",
                        "province": "ON",
                    }
                ]
            )
        if "SUM(contribution_count)" in sql:
            return iter(
                [
                    {
                        "financial_report": "Return",
                        "contributor_type": "Individuals",
                        "contributions": 10,
                        "monetary": Decimal("1234.50"),
                        "non_monetary": Decimal("0"),
                    }
                ]
            )
        if "COUNT(*)" in sql:
            return iter([{"n": 0}])
        if "person_expenses" in sql:
            return iter([{"fiscal_year": "2024-25", "quarter": 1, "travel": Decimal("100"),
                          "median_travel": Decimal("90")}])
        return iter([])


def _ctx(bq: FakeBq, *, enabled: bool = True) -> tuple[agent_tools.ToolContext, list[Any]]:
    events: list[Any] = []
    settings = Settings(gcp_project_id="proj", agent_people_enabled=enabled)
    state = agent_tools.LoopState(conversation_id="c", turn_id="t", question="q")
    return agent_tools.ToolContext(
        bq=bq,
        openai_client=None,
        settings=settings,  # type: ignore[arg-type]
        state=state,
        emit=events.append,
    ), events


def test_flag_off_changes_nothing() -> None:
    off = Settings(agent_people_enabled=False)
    for route in (None, "parliament", "statcan"):
        names = [t["function"]["name"] for t in agent_tools.routed_tool_schemas(route, off)]
        assert names == [t["function"]["name"] for t in agent_tools.routed_tool_schemas(route)]
        assert "person_record" not in names


def test_flag_on_adds_it_to_parliament_and_open_routes_only() -> None:
    on = Settings(agent_people_enabled=True)
    assert "person_record" in [
        t["function"]["name"] for t in agent_tools.routed_tool_schemas("parliament", on)
    ]
    assert "person_record" in [t["function"]["name"] for t in agent_tools.routed_tool_schemas(None, on)]
    assert "person_record" not in [
        t["function"]["name"] for t in agent_tools.routed_tool_schemas("statcan", on)
    ]


def test_queries_are_parameterised_and_per_return() -> None:
    bq = FakeBq()
    ctx, events = _ctx(bq)
    out = agent_tools.run_person_record(ctx=ctx, args={"politician": "x"})
    # Money stays an exact decimal string, never a float.
    assert out["contributions_by_return"][0]["monetary"] == "1234.50"
    assert all("@pid" in sql and "op:x" not in sql for sql, _ in bq.calls)
    assert isinstance(events[0], agent_events.SourceData) and events[0].source == "curated"
    assert "never summed" in out["notes"]


def test_missing_tables_are_a_named_error() -> None:
    ctx, _ = _ctx(FakeBq(fail=True))
    assert agent_tools.run_person_record(ctx=ctx, args={"politician": "x"})["reason"] == "curated_unavailable"


def test_refuses_when_disabled_or_given_a_name() -> None:
    ctx, _ = _ctx(FakeBq(), enabled=False)
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_person_record(ctx=ctx, args={"politician": "x"})
    ctx, _ = _ctx(FakeBq())
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_person_record(ctx=ctx, args={"politician": "Pierre Poilievre"})


def test_expenses_come_with_the_peer_median() -> None:
    ctx, _ = _ctx(FakeBq())
    out = agent_tools.run_person_record(ctx=ctx, args={"politician": "x"})
    [q] = out["office_expenses_by_quarter"]
    assert q["travel"] == "100" and q["median_travel"] == "90"
