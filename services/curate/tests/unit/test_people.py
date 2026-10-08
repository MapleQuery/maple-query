from __future__ import annotations

from datetime import date

from curate.core.people import build, person_id_for

POLITICIANS = [
    {"name": "Pierre Poilievre", "url": "/politicians/pierre-poilievre/"},
    {"name": "Fabian Manning", "url": "/politicians/438/"},
    {"name": "Never Elected", "url": "/politicians/never-elected/"},
]
MEMBERSHIPS = [
    {
        "url": "/politicians/memberships/1/",
        "politician_url": "/politicians/pierre-poilievre/",
        "start_date": "2004-06-28",
        "end_date": "2025-04-28",
        "party": {"short_name": {"en": "Conservative"}},
        "riding": {"name": {"en": "Carleton"}, "province": "ON"},
        "label": {"en": "Conservative MP for Carleton"},
    },
    {
        "url": "/politicians/memberships/2/",
        "politician_url": "/politicians/pierre-poilievre/",
        "start_date": "2025-08-18",
        "end_date": None,
        "party": {"short_name": {"en": "Conservative"}},
        "riding": {"name": {"en": "Battle River—Crowfoot"}, "province": "AB"},
        "label": {"en": "Conservative MP for Battle River—Crowfoot"},
    },
    {
        "url": "/politicians/memberships/3/",
        "politician_url": "/politicians/438/",
        "start_date": "2006-04-03",
        "end_date": "2008-09-07",
        "party": {"short_name": {"en": "Conservative"}},
        "riding": {"name": {"en": "Avalon"}, "province": "NL"},
    },
]
DETAILS = {
    "/politicians/pierre-poilievre/": {
        "given_name": "Pierre",
        "family_name": "Poilievre",
        "other_info": {"alternate_name": ["the honourable pierre poilievre", "Pierre Poilievre, M.P."]},
        "memberships": MEMBERSHIPS[:2],
    },
    "/politicians/438/": {"memberships": MEMBERSHIPS[2:]},
}


def test_person_ids_follow_the_url_slug_or_number() -> None:
    assert person_id_for("/politicians/pierre-poilievre/") == "op:pierre-poilievre"
    assert person_id_for("/politicians/438/") == "op:438"


def test_people_terms_and_current_status() -> None:
    t = build(POLITICIANS, DETAILS)
    by = {p.person_id: p for p in t.people}
    pp = by["op:pierre-poilievre"]
    assert pp.current_mp is True
    assert pp.first_term_start == date(2004, 6, 28)
    assert pp.last_term_end is None
    fm = by["op:438"]
    assert fm.current_mp is False and fm.last_term_end == date(2008, 9, 7)
    ne = by["op:never-elected"]
    assert ne.current_mp is False and ne.first_term_start is None
    assert {x.term_id for x in t.terms} == {f"/politicians/memberships/{i}/" for i in (1, 2, 3)}


def test_name_variants_collapse_to_one_normal_form() -> None:
    t = build(POLITICIANS, DETAILS)
    pp_names = {(n.name_norm, n.origin) for n in t.names if n.person_id == "op:pierre-poilievre"}
    assert ("pierre poilievre", "openparliament_name") in pp_names
    assert ("pierre poilievre", "alternate_name") in pp_names
    assert all(norm == "pierre poilievre" for norm, _ in pp_names)


def test_terms_come_from_each_persons_detail_only() -> None:
    # The paged memberships list drops terms; a person with no detail has
    # no terms rather than a partial set read from that list.
    t = build(POLITICIANS, {})
    assert t.terms == []
    assert all(not p.current_mp for p in t.people)
