"""Link House of Commons members' quarterly expenditures to people.

Each quarter's CSV lists one row per seat: `Name` ('Arnold,  Mel',
'Carney, Right Hon. Mark'), `Constituency`, `Caucus`, and four totals.
A row links when the normalised name matches a person with a Commons
term overlapping the quarter (or ended within a year before it); namesakes are settled by caucus, else the
row is `ambiguous`. 'Vacant' seats and names with no fitting person are
counted, not stored. The constituency is stored as published, as
context; it never decides a link.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from curate.clients.ourcommons import QuarterReport
from curate.core.contributions import TERM_WINDOW, NameIndex, _party_words
from curate.core.names import normalize
from curate.types import PersonExpense

COLUMNS = ("Name", "Constituency", "Caucus", "Salaries", "Travel", "Hospitality", "Contracts")


class ExpensesShapeError(ValueError):
    """The CSV no longer has the columns this linker reads."""


@dataclass
class ExpenseStats:
    rows_read: int = 0
    linked: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    # Names that matched nobody, so a run report can show them (a
    # spelling the normaliser misses is visible, not silent).
    unmatched_names: Counter[str] = field(default_factory=Counter)


def fiscal_year(period_start: date) -> str:
    start = period_start.year if period_start.month >= 4 else period_start.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def fiscal_quarter(period_start: date) -> int:
    return ((period_start.month - 4) % 12) // 3 + 1


def _overlaps(index: NameIndex, pid: str, start: date, end: date) -> list[str]:
    """Parties of the person's terms that cover the quarter, or ended
    within `TERM_WINDOW` before it: the House keeps reporting a former
    member's office wind-down costs for several quarters after they
    leave (Jagmeet Singh, Scott Simms and others, four quarters each)."""
    parties = []
    for t in index.terms.get(pid, []):
        if t.start_date <= end and (t.end_date is None or t.end_date + TERM_WINDOW >= start):
            parties.append(t.party or "")
    return parties


def link(report: QuarterReport, index: NameIndex, stats: ExpenseStats) -> Iterator[PersonExpense]:
    reader = csv.DictReader(io.StringIO(report.csv_text))
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ExpensesShapeError(f"{report.csv_url} is missing columns {missing}")
    start, end = report.period_start, report.period_end
    fy, q = fiscal_year(start), fiscal_quarter(start)
    for row in reader:
        stats.rows_read += 1
        name = (row.get("Name") or "").strip()
        if not name or name.casefold() == "vacant":
            stats.skipped["vacant"] += 1
            continue
        people = index.by_name.get(normalize(name), set())
        fitting = {p: _overlaps(index, p, start, end) for p in sorted(people)}
        fitting = {p: parties for p, parties in fitting.items() if parties}
        status, person_id, candidates, method = "unmatched", None, [], "exact_name_term"
        if len(fitting) == 1:
            status, person_id = "linked", next(iter(fitting))
        elif len(fitting) > 1:
            caucus = _party_words(row.get("Caucus"))
            same = [
                p
                for p, parties in fitting.items()
                if caucus and any((pw := _party_words(x)) and pw <= caucus for x in parties)
            ]
            if len(same) == 1:
                status, person_id, method = "linked", same[0], "exact_name_term_party"
            else:
                status, candidates = "ambiguous", sorted(fitting)
        if status == "unmatched":
            stats.unmatched += 1
            stats.unmatched_names[name] += 1
            continue
        try:
            amounts = [
                str(Decimal((row.get(c) or "0").strip() or "0"))
                for c in ("Salaries", "Travel", "Hospitality", "Contracts")
            ]
        except InvalidOperation:
            stats.skipped["bad_amount"] += 1
            continue
        constituency = (row.get("Constituency") or "").strip() or None
        key = hashlib.sha256(f"{fy}\x1f{q}\x1f{name}\x1f{constituency or ''}".encode()).hexdigest()
        if status == "linked":
            stats.linked += 1
        else:
            stats.ambiguous += 1
        yield PersonExpense(
            expense_key=key,
            person_id=person_id,
            status=status,
            candidates=candidates,
            match_method=method,
            member_name=name,
            constituency=constituency,
            caucus=(row.get("Caucus") or "").strip() or None,
            fiscal_year=fy,
            quarter=q,
            period_start=start,
            period_end=end,
            salaries=amounts[0],
            travel=amounts[1],
            hospitality=amounts[2],
            contracts=amounts[3],
            source_url=report.csv_url,
        )
