"""People, name variants and Commons terms from openparliament.ca.

Pure: takes the API's objects, returns table rows. Fetching lives in
clients/, orchestration in runner.py.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from curate.core.names import normalize, variants
from curate.types import Person, PersonName, PersonTerm

SITE = "https://openparliament.ca"


def person_id_for(url: str) -> str:
    """'/politicians/pierre-poilievre/' -> 'op:pierre-poilievre'."""
    segment = (url or "").rstrip("/").rsplit("/", 1)[-1]
    if not segment:
        raise ValueError(f"not a politician url: {url!r}")
    return f"op:{segment}"


def _date(value: Any) -> date | None:
    return date.fromisoformat(str(value)[:10]) if value else None


@dataclass(frozen=True)
class PeopleTables:
    people: list[Person]
    names: list[PersonName]
    terms: list[PersonTerm]


def build(
    politicians: Iterable[dict[str, Any]],
    memberships: Iterable[dict[str, Any]],
    details: dict[str, dict[str, Any]],
) -> PeopleTables:
    """`details` maps a politician URL to its detail object (given and
    family name, alternate names); people without one still get a row,
    with names derived from the display name only."""
    terms_by_person: dict[str, list[PersonTerm]] = defaultdict(list)
    for m in memberships:
        pid = person_id_for(str(m.get("politician_url")))
        start = _date(m.get("start_date"))
        if start is None:
            continue
        riding = m.get("riding") or {}
        terms_by_person[pid].append(
            PersonTerm(
                term_id=str(m.get("url")),
                person_id=pid,
                role="MP",
                start_date=start,
                end_date=_date(m.get("end_date")),
                party=((m.get("party") or {}).get("short_name") or {}).get("en"),
                riding_name=(riding.get("name") or {}).get("en"),
                province=riding.get("province"),
                label=(m.get("label") or {}).get("en"),
            )
        )

    people: list[Person] = []
    names: dict[tuple[str, str, str], PersonName] = {}
    for p in politicians:
        url = str(p.get("url"))
        pid = person_id_for(url)
        detail = details.get(url) or {}
        given = detail.get("given_name") or None
        family = detail.get("family_name") or None
        terms = sorted(terms_by_person.get(pid, []), key=lambda t: t.start_date)
        ends = [t.end_date for t in terms]
        people.append(
            Person(
                person_id=pid,
                name=str(p.get("name")),
                given_name=given,
                family_name=family,
                openparliament_url=f"{SITE}{url}",
                current_mp=any(e is None for e in ends) if terms else False,
                first_term_start=terms[0].start_date if terms else None,
                last_term_end=(
                    None if not terms or any(e is None for e in ends) else max(e for e in ends if e)
                ),
            )
        )

        def add(raw: str, origin: str, person_id: str = pid) -> None:
            norm = normalize(raw)
            if norm:
                names.setdefault(
                    (person_id, norm, origin),
                    PersonName(person_id=person_id, name_norm=norm, name_raw=raw, origin=origin),
                )

        add(str(p.get("name")), "openparliament_name")
        if given and family:
            for v in variants(str(p.get("name")), given, family) - {normalize(str(p.get("name")))}:
                add(v, "given_family")
        other = detail.get("other_info") or {}
        for alt in other.get("alternate_name") or []:
            add(str(alt), "alternate_name")

    all_terms = [t for ts in terms_by_person.values() for t in ts]
    return PeopleTables(people=people, names=list(names.values()), terms=all_terms)
