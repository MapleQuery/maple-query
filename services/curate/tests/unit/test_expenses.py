from __future__ import annotations

from datetime import date

import pytest

from curate.clients.ourcommons import QuarterRef, QuarterReport, parse_quarter_page, parse_quarters
from curate.core.contributions import NameIndex
from curate.core.expenses import ExpensesShapeError, ExpenseStats, fiscal_quarter, fiscal_year, link
from curate.types import PersonName, PersonTerm

LISTING = """
<a href="/ProactiveDisclosure/en/members/2023/1?summaryId=fdc0b05b-ae0b-4942-a454-0a9c08d2ea47">Q1</a>
<a href="/ProactiveDisclosure/en/members/2022/4?summaryId=7497f0ec-5fb8-46e4-ba84-b6bc30fd75f6">Q4</a>
<a href="/ProactiveDisclosure/en/members/2023/1?summaryId=fdc0b05b-ae0b-4942-a454-0a9c08d2ea47">dup</a>
"""
QUARTER_PAGE = """
<title>Members &#x2013; Summary of Expenditures</title>
<p>April 1, 2022 to June 30, 2022</p>
<a class="csv-btn" href="/ProactiveDisclosure/en/members/4c6263e7-61ad-4673-ada2-13585a65e15e/csv">CSV</a>
"""
CSV = (
    "Name,Constituency,Caucus,Salaries,Travel,Hospitality,Contracts\n"
    "Vacant,Mississauga—Lakeshore,Liberal,4262.5,0,0,3809.95\n"
    '"Arnold,  Mel",Kamloops,Conservative,77758.59,43670.6,249.77,20653.16\n'
    '"Carney, Right Hon. Mark",Nepean,Liberal,59297.49,0,274.83,36062.44\n'
    '"Anderson,  David",Somewhere,Conservative,1,2,3,4\n'
    '"Nobody,  Known",Elsewhere,Liberal,1,1,1,1\n'
)


def _report() -> QuarterReport:
    ref = QuarterRef("/x", 2023, 1)
    return QuarterReport(ref, date(2022, 4, 1), date(2022, 6, 30), "https://www.ourcommons.ca/csv", CSV)


def _term(pid: str, start: str, end: str | None, party: str) -> PersonTerm:
    return PersonTerm(
        f"t{pid}",
        pid,
        "MP",
        date.fromisoformat(start),
        date.fromisoformat(end) if end else None,
        party,
        None,
        None,
        None,
    )


INDEX = NameIndex.build(
    [
        PersonName("op:mel-arnold", "mel arnold", "", ""),
        PersonName("op:mark-carney", "mark carney", "", ""),
        PersonName("op:lib", "david anderson", "", ""),
        PersonName("op:con", "david anderson", "", ""),
    ],
    [
        _term("op:mel-arnold", "2015-11-03", None, "Conservative"),
        _term("op:mark-carney", "2025-04-28", None, "Liberal"),  # not an MP in 2022
        _term("op:lib", "2021-09-20", None, "Liberal"),
        _term("op:con", "2019-10-21", None, "Conservative"),
    ],
)


def test_listing_and_quarter_page_parse() -> None:
    refs = parse_quarters(LISTING)
    assert [(r.report_year, r.quarter) for r in refs] == [(2022, 4), (2023, 1)]
    start, end, csv_path = parse_quarter_page(QUARTER_PAGE)
    assert (start, end) == (date(2022, 4, 1), date(2022, 6, 30))
    assert csv_path.endswith("/csv")


def test_fiscal_labels() -> None:
    assert fiscal_year(date(2022, 4, 1)) == "2022-23" and fiscal_quarter(date(2022, 4, 1)) == 1
    assert fiscal_year(date(2023, 1, 1)) == "2022-23" and fiscal_quarter(date(2023, 1, 1)) == 4


def test_link_rules_on_one_quarter() -> None:
    stats = ExpenseStats()
    rows = {r.member_name: r for r in link(_report(), INDEX, stats)}
    assert rows["Arnold,  Mel"].person_id == "op:mel-arnold"
    assert rows["Arnold,  Mel"].travel == "43670.6"
    # Carney's term starts in 2025: a 2022 row is not his.
    assert "Carney, Right Hon. Mark" not in rows
    # Two David Andersons in office; the caucus settles it.
    assert rows["Anderson,  David"].person_id == "op:con"
    assert rows["Anderson,  David"].match_method == "exact_name_term_party"
    assert stats.skipped["vacant"] == 1 and stats.unmatched == 2


def test_missing_column_fails_loudly() -> None:
    bad = QuarterReport(_report().ref, date(2022, 4, 1), date(2022, 6, 30), "u", "Name,Travel\nX,1\n")
    with pytest.raises(ExpensesShapeError):
        list(link(bad, INDEX, ExpenseStats()))


def test_wind_down_quarters_after_a_term_still_link() -> None:
    idx = NameIndex.build(
        [PersonName("op:gone", "scott simms", "", "")],
        [_term("op:gone", "2015-10-19", "2021-09-20", "Liberal")],
    )
    csv = (
        'Name,Constituency,Caucus,Salaries,Travel,Hospitality,Contracts\n"Simms,  Scott",X,Liberal,0,0,0,10\n'
    )
    for start, end, linked in [
        (date(2022, 1, 1), date(2022, 3, 31), True),  # 3 months after: wind-down
        (date(2023, 1, 1), date(2023, 3, 31), False),  # 15 months after: not his
    ]:
        rep = QuarterReport(_report().ref, start, end, "u", csv)
        assert bool(list(link(rep, idx, ExpenseStats()))) is linked
