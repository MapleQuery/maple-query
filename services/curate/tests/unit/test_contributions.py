from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from curate.core.contributions import LinkStats, NameIndex, SourceShapeError, link
from curate.types import PersonName, PersonTerm

HEADER = {
    "Political Entity": "Candidates",
    "Recipient ID": "1",
    "Recipient": "Smith, John",
    "Recipient last name": "Smith",
    "Recipient first name": "John",
    "Political Party of Recipient": "Liberal Party of Canada",
    "Electoral District": "Somewhere",
    "Electoral event": "43rd general election",
    "Fiscal/Election date": "2019-10-21",
    "Financial Report": "Candidate's Electoral Campaign Return",
    "Contributor type": "Individuals",
    "Contributor name": "Doe, Jane",
    "Contributor Province": "ON",
    "Contributor Postal code": "K1A 0A6",
    "Contribution Received date": "2019-09-01",
    "Monetary amount": "   200.00",
    "Non-Monetary amount": "   .00",
}


def row(**over: str) -> dict[str, str]:
    return {**HEADER, **over}


def term(pid: str, start: str, end: str | None) -> PersonTerm:
    return PersonTerm(
        f"t-{pid}-{start}",
        pid,
        "MP",
        date.fromisoformat(start),
        date.fromisoformat(end) if end else None,
        None,
        None,
        None,
        None,
    )


INDEX = NameIndex.build(
    [
        PersonName("op:john-smith", "john smith", "John Smith", "openparliament_name"),
        PersonName("op:john-smith-2", "john smith", "John Smith", "openparliament_name"),
        PersonName("op:ann-lee", "ann lee", "Ann Lee", "openparliament_name"),
    ],
    [
        term("op:john-smith", "2019-12-05", "2021-08-15"),  # won 2019
        term("op:john-smith-2", "1997-06-02", "2000-10-22"),  # a different John Smith
        term("op:ann-lee", "2015-11-03", None),
    ],
)


def test_name_plus_term_window_links_the_right_namesake() -> None:
    [g] = list(link([row()], INDEX, source_object="gs://x"))
    assert g.status == "linked" and g.person_id == "op:john-smith"
    assert g.match_method == "exact_name_term" and g.confidence == 1.0


def test_namesakes_both_in_window_are_ambiguous_not_guessed() -> None:
    idx = NameIndex.build(
        [PersonName("a", "john smith", "", ""), PersonName("b", "john smith", "", "")],
        [term("a", "2019-12-05", None), term("b", "2015-11-03", None)],
    )
    [g] = list(link([row()], idx, source_object="gs://x"))
    assert g.status == "ambiguous" and g.person_id is None and g.candidates == ["a", "b"]


def test_a_namesake_outside_every_term_is_not_linked() -> None:
    stats = LinkStats()
    out = list(
        link([row(**{"Fiscal/Election date": "2008-10-14"})], INDEX, source_object="gs://x", stats=stats)
    )
    assert out == [] and stats.unmatched_rows == 1


def test_party_and_association_rows_are_ignored() -> None:
    stats = LinkStats()
    assert (
        list(link([row(**{"Political Entity": "Registered parties"})], INDEX, source_object="x", stats=stats))
        == []
    )
    assert stats.person_rows == 0


def test_individual_donors_are_aggregated_and_never_named() -> None:
    rows = [row(), row(**{"Contributor name": "Roe, Richard", "Monetary amount": "50.50"})]
    [g] = list(link(rows, INDEX, source_object="gs://x"))
    assert g.contributor_name is None
    assert g.contribution_count == 2
    assert Decimal(g.monetary_total) == Decimal("250.50")
    assert g.contributor_province == "ON"


def test_organizations_keep_their_name() -> None:
    [g] = list(
        link(
            [row(**{"Contributor type": "Businesses", "Contributor name": "ACME INC."})],
            INDEX,
            source_object="x",
        )
    )
    assert g.contributor_name == "ACME INC."


def test_a_renamed_column_fails_loudly() -> None:
    bad = {k: v for k, v in row().items() if k != "Recipient last name"}
    with pytest.raises(SourceShapeError):
        list(link([bad], INDEX, source_object="x"))


def test_party_on_the_return_settles_namesakes() -> None:
    def t(pid: str, party: str) -> PersonTerm:
        return PersonTerm(f"t-{pid}", pid, "MP", date(2004, 6, 28), date(2008, 9, 7), party, None, None, None)

    idx = NameIndex.build(
        [PersonName("lib", "david anderson", "", ""), PersonName("con", "david anderson", "", "")],
        [t("lib", "Liberal"), t("con", "Conservative")],
    )
    r = row(
        **{
            "Recipient last name": "Anderson",
            "Recipient first name": "David",
            "Fiscal/Election date": "2006-01-23",
            "Political Party of Recipient": "Conservative Party of Canada",
        }
    )
    [g] = list(link([r], idx, source_object="x"))
    assert g.status == "linked" and g.person_id == "con"
    assert g.match_method == "exact_name_term_party"
    # A party neither sat for leaves it ambiguous.
    [g2] = list(
        link([{**r, "Political Party of Recipient": "Green Party of Canada"}], idx, source_object="x")
    )
    assert g2.status == "ambiguous"
