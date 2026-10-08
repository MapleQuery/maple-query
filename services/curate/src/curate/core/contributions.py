"""Link Elections Canada contributions to people.

Reads the contributions CSV row by row (it is 3.8 GB; nothing is held
but the aggregates), keeps rows whose recipient is a person (candidates
and leadership or nomination contestants), and links a recipient to a
person only when BOTH hold:

- their normalised name equals one of the person's name variants, and
- the return's date falls within `TERM_WINDOW` of one of that person's
  Commons terms (a candidate files around the election that starts or
  ends a term).

One surviving person -> `linked`. Several -> `ambiguous`, with the
candidates recorded and nobody chosen. None -> not linked and not
stored, only counted: most candidates never sit in the Commons, and a
namesake who lost in 2004 is not the MP of the same name elected in
2015. Nothing here guesses, and nothing waits on a person to confirm.

Rows are aggregated per (recipient, return, contributor type, contributor
province, and organization name for non-individual contributors).
Individual donors are never named or located beyond province.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from curate.core.names import normalize
from curate.types import PersonContribution, PersonName, PersonTerm

PERSON_ENTITIES = frozenset({"Candidates", "Leadership contestants", "Nomination contestants"})
TERM_WINDOW = timedelta(days=365)

# Column names as Elections Canada publishes them (od_cntrbtn_de_e.csv).
COL_ENTITY = "Political Entity"
COL_RECIPIENT_ID = "Recipient ID"
COL_RECIPIENT = "Recipient"
COL_LAST = "Recipient last name"
COL_FIRST = "Recipient first name"
COL_PARTY = "Political Party of Recipient"
COL_DISTRICT = "Electoral District"
COL_EVENT = "Electoral event"
COL_REPORT_DATE = "Fiscal/Election date"
COL_REPORT = "Financial Report"
COL_CONTRIB_TYPE = "Contributor type"
COL_CONTRIB_NAME = "Contributor name"
COL_PROVINCE = "Contributor Province"
COL_RECEIVED = "Contribution Received date"
COL_MONETARY = "Monetary amount"
COL_NON_MONETARY = "Non-Monetary amount"
REQUIRED_COLUMNS = (
    COL_ENTITY,
    COL_RECIPIENT_ID,
    COL_LAST,
    COL_FIRST,
    COL_REPORT_DATE,
    COL_CONTRIB_TYPE,
    COL_MONETARY,
    COL_NON_MONETARY,
)


class SourceShapeError(ValueError):
    """The file does not have the columns this linker reads. Fails the
    run: a renamed column would otherwise link nothing, silently."""


@dataclass
class NameIndex:
    by_name: dict[str, set[str]]
    terms: dict[str, list[PersonTerm]]

    @classmethod
    def build(cls, names: Iterable[PersonName], terms: Iterable[PersonTerm]) -> NameIndex:
        by_name: dict[str, set[str]] = defaultdict(set)
        for n in names:
            by_name[n.name_norm].add(n.person_id)
        by_person: dict[str, list[PersonTerm]] = defaultdict(list)
        for t in terms:
            by_person[t.person_id].append(t)
        return cls(by_name=dict(by_name), terms=dict(by_person))

    def window_terms(self, person_id: str, when: date) -> list[PersonTerm]:
        out = []
        for t in self.terms.get(person_id, []):
            end = t.end_date or date.max - TERM_WINDOW
            if t.start_date - TERM_WINDOW <= when <= end + TERM_WINDOW:
                out.append(t)
        return out

    def in_window(self, person_id: str, when: date) -> bool:
        return bool(self.window_terms(person_id, when))


@dataclass
class LinkStats:
    rows_read: int = 0
    person_rows: int = 0
    linked_rows: int = 0
    ambiguous_rows: int = 0
    unmatched_rows: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _Resolution:
    status: str  # linked | ambiguous | unmatched
    person_id: str | None
    candidates: tuple[str, ...]
    method: str = "exact_name_term"


def _party_words(party: str | None) -> set[str]:
    stop = {"party", "of", "canada", "du", "parti", "the", "de"}
    return {w for w in normalize(party or "").split() if w not in stop}


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat((value or "").strip()[:10]) if (value or "").strip() else None
    except ValueError:
        return None


def _amount(value: str | None) -> Decimal:
    text = (value or "").strip()
    return Decimal(text) if text else Decimal("0")


def resolve(index: NameIndex, name_norm: str, when: date | None, party: str | None = None) -> _Resolution:
    people = index.by_name.get(name_norm, set())
    if not people or when is None:
        return _Resolution("unmatched", None, ())
    fitting = sorted(p for p in people if index.in_window(p, when))
    if len(fitting) == 1:
        return _Resolution("linked", fitting[0], ())
    if len(fitting) > 1:
        # Namesakes in the window (two MPs called David Anderson): the
        # return names the recipient's party; if exactly one of them sat
        # for that party in the matching term, it is them. Deterministic,
        # and the riding is still never consulted.
        filed = _party_words(party)
        if filed:
            same_party = [
                p
                for p in fitting
                if any((tw := _party_words(t.party)) and tw <= filed for t in index.window_terms(p, when))
            ]
            if len(same_party) == 1:
                return _Resolution("linked", same_party[0], (), "exact_name_term_party")
        return _Resolution("ambiguous", None, tuple(fitting))
    return _Resolution("unmatched", None, ())


def link(
    rows: Iterable[dict[str, str]],
    index: NameIndex,
    *,
    source_object: str,
    stats: LinkStats | None = None,
) -> Iterator[PersonContribution]:
    stats = stats if stats is not None else LinkStats()
    groups: dict[str, PersonContribution] = {}
    cache: dict[tuple[str, str], _Resolution] = {}
    checked_shape = False
    for row in rows:
        if not checked_shape:
            missing = [c for c in REQUIRED_COLUMNS if c not in row]
            if missing:
                raise SourceShapeError(f"contributions file is missing columns {missing}")
            checked_shape = True
        stats.rows_read += 1
        entity = (row.get(COL_ENTITY) or "").strip()
        if entity not in PERSON_ENTITIES:
            continue
        stats.person_rows += 1
        recipient_id = (row.get(COL_RECIPIENT_ID) or "").strip()
        report_date = _parse_date(row.get(COL_REPORT_DATE))
        key = (recipient_id, str(report_date))
        res = cache.get(key)
        if res is None:
            name = normalize(f"{row.get(COL_FIRST) or ''} {row.get(COL_LAST) or ''}") or normalize(
                row.get(COL_RECIPIENT) or ""
            )
            res = resolve(index, name, report_date, (row.get(COL_PARTY) or "").strip() or None)
            cache[key] = res
        if res.status == "unmatched":
            stats.unmatched_rows += 1
            continue
        try:
            monetary = _amount(row.get(COL_MONETARY))
            non_monetary = _amount(row.get(COL_NON_MONETARY))
        except InvalidOperation:
            stats.skipped["bad_amount"] += 1
            continue

        contributor_type = (row.get(COL_CONTRIB_TYPE) or "").strip() or "Unknown"
        individual = contributor_type.casefold().startswith("individual")
        contributor_name = None if individual else ((row.get(COL_CONTRIB_NAME) or "").strip() or None)
        province = (row.get(COL_PROVINCE) or "").strip() or None
        report = (row.get(COL_REPORT) or "").strip() or None
        event = (row.get(COL_EVENT) or "").strip() or None
        agg = "\x1f".join(
            [
                recipient_id,
                str(report_date),
                report or "",
                event or "",
                contributor_type,
                contributor_name or "",
                province or "",
            ]
        )
        ckey = hashlib.sha256(agg.encode()).hexdigest()
        g = groups.get(ckey)
        if g is None:
            g = PersonContribution(
                contribution_key=ckey,
                person_id=res.person_id,
                status=res.status,
                candidates=list(res.candidates),
                match_method=res.method,
                confidence=1.0 if res.status == "linked" else 0.0,
                political_entity=entity,
                recipient_id=recipient_id,
                recipient_name=(row.get(COL_RECIPIENT) or "").strip(),
                recipient_party=(row.get(COL_PARTY) or "").strip() or None,
                electoral_district=(row.get(COL_DISTRICT) or "").strip() or None,
                electoral_event=event,
                report_date=report_date,
                financial_report=report,
                contributor_type=contributor_type,
                contributor_name=contributor_name,
                contributor_province=province,
                source_object=source_object,
            )
            groups[ckey] = g
        g.contribution_count += 1
        g.monetary_total = str(Decimal(g.monetary_total) + monetary)
        g.non_monetary_total = str(Decimal(g.non_monetary_total) + non_monetary)
        received = _parse_date(row.get(COL_RECEIVED))
        if received:
            g.first_received = min(filter(None, [g.first_received, received]))
            g.last_received = max(filter(None, [g.last_received, received]))
        if res.status == "linked":
            stats.linked_rows += 1
        else:
            stats.ambiguous_rows += 1
    yield from groups.values()
