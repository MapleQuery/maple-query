"""Parliament tools over a fake openparliament.ca: MP lookup, bill
parsing, ballots joined to votes, speech filtering, and the tool
contract (events, refs, named errors)."""

from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.clients.parliament import ParliamentError
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools, parliament_tools

MPS = [
    {
        "name": "Pierre Poilievre",
        "url": "/politicians/pierre-poilievre/",
        "current_party": {"short_name": {"en": "Conservative"}},
        "current_riding": {"province": "AB", "name": {"en": "Battle River—Crowfoot"}},
    },
    {
        "name": "Ziad Aboultaif",
        "url": "/politicians/ziad-aboultaif/",
        "current_party": {"short_name": {"en": "Conservative"}},
        "current_riding": {"province": "AB", "name": {"en": "Edmonton Manning"}},
    },
]
FORMER = [*MPS, {"name": "Justin Trudeau", "url": "/politicians/justin-trudeau/"}]
VOTE = {
    "session": "45-1",
    "number": 8,
    "date": "2025-06-12",
    "description": {"en": "2nd reading of Bill C-4"},
    "result": "Passed",
    "yea_total": 335,
    "nay_total": 0,
    "bill_url": "/bills/45-1/C-4/",
    "url": "/votes/45-1/8/",
}


class FakeParliament:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.fail:
            raise ParliamentError("openparliament.ca 503")
        if path == "/votes/":
            return {"objects": [VOTE]}
        if path == "/votes/45-1/8/":
            return {
                **VOTE,
                "party_votes": [
                    {"vote": "Yes", "disagreement": 0.0, "party": {"short_name": {"en": "Liberal"}}},
                    {"vote": "Yes", "disagreement": 0.05, "party": {"short_name": {"en": "Conservative"}}},
                ],
            }
        if path == "/bills/45-1/C-4/":
            return {
                "session": "45-1",
                "number": "C-4",
                "name": {"en": "An Act respecting certain affordability measures"},
                "law": True,
                "sponsor_politician_url": "/politicians/francois-philippe-champagne/",
                "vote_urls": ["/votes/45-1/8/"],
            }
        raise AssertionError(path)

    def list_all(
        self, path: str, params: dict[str, Any] | None = None, *, max_rows: int = 2000
    ) -> list[dict[str, Any]]:
        params = params or {}
        if path == "/votes/" and params.get("bill"):
            return [VOTE]
        if path == "/votes/ballots/":
            if params.get("politician") == "/politicians/pierre-poilievre/":
                return []  # not an MP when C-4 was voted
            return [{"vote_url": "/votes/45-1/8/", "ballot": "Yes"}]
        if path == "/speeches/":
            return [
                {
                    "time": "2026-10-07 14:40",
                    "content": {"en": "<p>Grocery prices are up.</p>"},
                    "url": "/s/1/",
                },
                {
                    "time": "2026-10-06 14:40",
                    "content": {"en": "<p>Canadians cannot afford housing.</p>"},
                    "h2": {"en": "Housing"},
                    "url": "/s/2/",
                },
            ]
        raise AssertionError(path)

    def list_cached(
        self, path: str, params: dict[str, Any] | None = None, *, ttl_s: float = 0
    ) -> list[dict[str, Any]]:
        if path == "/politicians/":
            return FORMER if (params or {}).get("include") == "all" else MPS
        if path == "/bills/":
            return [
                {
                    "name": {"en": "An Act respecting certain affordability measures"},
                    "url": "/bills/45-1/C-4/",
                }
            ]
        if path == "/votes/":
            return [VOTE]
        raise AssertionError(path)


def test_find_politician_by_name_riding_and_former() -> None:
    c = FakeParliament()
    assert parliament_tools.find_politician(c, "poilievre")[0]["politician"] == "pierre-poilievre"
    assert parliament_tools.find_politician(c, "Edmonton Manning")[0]["politician"] == "ziad-aboultaif"
    former = parliament_tools.find_politician(c, "Justin Trudeau")[0]
    assert former["politician"] == "justin-trudeau" and former["current_mp"] is False


def test_bill_numbers_parse_with_and_without_session() -> None:
    assert parliament_tools.parse_bill("C-4", "45-1") == ("45-1", "C-4")
    assert parliament_tools.parse_bill("44-1/c 21", "45-1") == ("44-1", "C-21")
    with pytest.raises(parliament_tools.ParliamentArgsError):
        parliament_tools.parse_bill("housing", "45-1")


def test_a_ballot_that_does_not_exist_reads_not_recorded() -> None:
    rows = parliament_tools.parliament_votes(FakeParliament(), politician="pierre-poilievre", bill="C-4")
    assert rows[0]["their_ballot"] == "not recorded"
    rows = parliament_tools.parliament_votes(FakeParliament(), politician="ziad-aboultaif", bill="C-4")
    assert rows[0]["their_ballot"] == "Yes"


def test_bill_votes_carry_the_party_breakdown() -> None:
    rows = parliament_tools.parliament_votes(FakeParliament(), bill="C-4")
    assert rows[0]["by_party"] == "Liberal: Yes; Conservative: Yes (5% split)"


def test_speeches_filter_by_subject_and_strip_markup() -> None:
    rows, scanned = parliament_tools.politician_speeches(
        FakeParliament(), politician="pierre-poilievre", query="housing"
    )
    assert scanned == 2
    assert [r["said"] for r in rows] == ["Canadians cannot afford housing."]
    assert rows[0]["topic"] == "Housing"


def _ctx(client: FakeParliament) -> tuple[agent_tools.ToolContext, list[Any]]:
    events: list[Any] = []
    state = agent_tools.LoopState(conversation_id="c", turn_id="t", question="q")
    ctx = agent_tools.ToolContext(
        bq=None,  # type: ignore[arg-type]
        openai_client=None,  # type: ignore[arg-type]
        settings=Settings(),
        state=state,
        emit=events.append,
        parliament=client,  # type: ignore[arg-type]
    )
    return ctx, events


def test_tool_records_refs_and_emits_source_data() -> None:
    ctx, events = _ctx(FakeParliament())
    out = agent_tools.run_find_bills(ctx=ctx, args={"query": "affordability"})
    assert out["rows"][0]["became_law"] is True
    assert "https://openparliament.ca/bills/45-1/C-4/" in ctx.state.parliament_refs
    assert isinstance(events[0], agent_events.SourceData) and events[0].source == "parliament"


def test_a_name_instead_of_a_slug_is_a_fixable_error() -> None:
    ctx, _ = _ctx(FakeParliament())
    with pytest.raises(agent_tools.InvalidToolArgsError, match="slug"):
        agent_tools.run_parliament_votes(ctx=ctx, args={"politician": "Pierre Poilievre"})


def test_outage_is_a_named_error() -> None:
    ctx, _ = _ctx(FakeParliament(fail=True))
    out = agent_tools.run_parliament_votes(ctx=ctx, args={"bill": "C-4"})
    assert out["reason"] == "parliament_unavailable"


def test_parliament_route_exposes_only_its_tools() -> None:
    names = {t["function"]["name"] for t in agent_tools.routed_tool_schemas("parliament")}
    assert names == {"find_politician", "parliament_votes", "find_bills", "politician_speeches", "calculate"}
