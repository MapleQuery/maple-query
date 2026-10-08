"""Parliament tools: who an MP is, how they voted, what bills exist,
what they said.

Read live from openparliament.ca (see clients/parliament.py). The record
is the House of Commons' own: recorded divisions, Hansard, bill status.
Answers built on it report what the record shows — votes cast, words
spoken — and never characterise motive; that boundary lives in the
system prompt, these functions only return the record.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from semantic_enrich.clients.parliament import SITE_BASE, ParliamentClient

MAX_DETAIL_FETCHES = 5
MAX_SPEECHES_SCANNED = 300

_TAG_RE = re.compile(r"<[^>]+>")
_BILL_RE = re.compile(r"^(?:(\d{2,3}-\d)\s*/\s*)?([CS])-?\s*(\d+)$", re.IGNORECASE)
_STOP = frozenset(
    ["the", "a", "an", "of", "and", "to", "in", "on", "for", "mp", "minister", "hon", "honourable", "member"]
)


class ParliamentArgsError(ValueError):
    """An argument the model can fix (unknown MP slug, malformed bill)."""


def norm(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).casefold()


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(text)) if t not in _STOP}


def slug_of(url: str | None) -> str:
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def site_url(path: str | None) -> str:
    return f"{SITE_BASE}{path}" if path else SITE_BASE


def current_session(client: ParliamentClient) -> str:
    body = client.get("/votes/", {"limit": 1})
    objects = body.get("objects") or []
    if not objects:
        raise ParliamentArgsError("could not determine the current parliamentary session")
    return str(objects[0].get("session"))


# ── politicians ──


def _politician_row(p: dict[str, Any], current: bool) -> dict[str, Any]:
    riding = p.get("current_riding") or {}
    party = p.get("current_party") or {}
    return {
        "politician": slug_of(p.get("url")),
        "name": p.get("name"),
        "party": (party.get("short_name") or {}).get("en"),
        "riding": (riding.get("name") or {}).get("en"),
        "province": riding.get("province"),
        "current_mp": current,
        "url": site_url(p.get("url")),
    }


def find_politician(
    client: ParliamentClient, query: str, *, include_former: bool = False, k: int = 5
) -> list[dict[str, Any]]:
    current = client.list_cached("/politicians/", ttl_s=6 * 3600)
    q = _tokens(query)
    if not q:
        raise ParliamentArgsError("give a name or a riding to look up")

    def score(p: dict[str, Any]) -> float:
        name = _tokens(str(p.get("name") or ""))
        riding = _tokens(str(((p.get("current_riding") or {}).get("name") or {}).get("en") or ""))
        s = 0.0
        if q and q <= name:
            s += 3.0
        s += len(q & name)
        if q and q <= riding:
            s += 2.0
        s += 0.5 * len(q & riding)
        return s

    best = sorted(((score(p), p, True) for p in current if score(p) > 0), key=lambda r: -r[0])
    # Former MPs only when asked, or when no sitting MP matches well:
    # the all-time list is several thousand rows, fetched once a day.
    if include_former or not best or best[0][0] < 2.0:
        former = client.list_cached("/politicians/", {"include": "all"}, ttl_s=24 * 3600)
        current_urls = {p.get("url") for p in current}
        extra = [(score(p), p, False) for p in former if p.get("url") not in current_urls and score(p) > 0]
        best = sorted(best + extra, key=lambda r: -r[0])
    return [_politician_row(p, cur) for _, p, cur in best[:k]]


# ── bills ──


def parse_bill(bill: str, default_session: str) -> tuple[str, str]:
    m = _BILL_RE.match(bill.strip().replace(" ", "")) or _BILL_RE.match(bill.strip())
    if not m:
        raise ParliamentArgsError(f"bill {bill!r} should look like C-4, S-209 or 45-1/C-4 (session/number)")
    return m.group(1) or default_session, f"{m.group(2).upper()}-{m.group(3)}"


def find_bills(
    client: ParliamentClient, query: str, *, session: str | None = None, k: int = 5
) -> list[dict[str, Any]]:
    session = session or current_session(client)
    m = _BILL_RE.match(query.strip())
    if m:
        sess, number = parse_bill(query, session)
        candidates = [{"url": f"/bills/{sess}/{number}/"}]
    else:
        bills = client.list_cached("/bills/", {"session": session}, ttl_s=6 * 3600)
        q = _tokens(query)
        scored = []
        for b in bills:
            name = _tokens(str((b.get("name") or {}).get("en") or ""))
            hit = len(q & name)
            if hit:
                scored.append((hit, b))
        scored.sort(key=lambda r: -r[0])
        candidates = [b for _, b in scored[:k]]
    rows = []
    for b in candidates[:MAX_DETAIL_FETCHES]:
        d = client.get(str(b["url"]))
        rows.append(
            {
                "bill": f"{d.get('session')}/{d.get('number')}",
                "name": (d.get("name") or {}).get("en"),
                "short_title": (d.get("short_title") or {}).get("en"),
                "introduced": d.get("introduced"),
                "became_law": bool(d.get("law")),
                "sponsor": slug_of(d.get("sponsor_politician_url")) or None,
                "recorded_votes": len(d.get("vote_urls") or []),
                "url": site_url(b["url"]),
            }
        )
    return rows


# ── votes ──


def _vote_row(v: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": v.get("date"),
        "vote": f"{v.get('session')}/{v.get('number')}",
        "description": (v.get("description") or {}).get("en"),
        "result": v.get("result"),
        "yea": v.get("yea_total"),
        "nay": v.get("nay_total"),
        "bill": slug_of(v.get("bill_url")).upper() or None,
        "url": site_url(v.get("url")),
    }


def _party_breakdown(detail: dict[str, Any]) -> str:
    parts = []
    for pv in detail.get("party_votes") or []:
        party = ((pv.get("party") or {}).get("short_name") or {}).get("en")
        dissent = pv.get("disagreement") or 0
        parts.append(f"{party}: {pv.get('vote')}" + (f" ({dissent:.0%} split)" if dissent else ""))
    return "; ".join(parts)


def parliament_votes(
    client: ParliamentClient,
    *,
    politician: str | None = None,
    bill: str | None = None,
    session: str | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    session = session or current_session(client)
    if politician and not re.fullmatch(r"[a-z0-9-]+", politician):
        raise ParliamentArgsError(
            f"politician must be a slug from find_politician (e.g. pierre-poilievre), not {politician!r}"
        )
    bill_votes: list[dict[str, Any]] | None = None
    if bill:
        sess, number = parse_bill(bill, session)
        bill_votes = client.list_all("/votes/", {"bill": f"/bills/{sess}/{number}/"}, max_rows=50)
        if not bill_votes:
            return []

    if politician:
        pol_path = f"/politicians/{politician}/"
        if bill_votes is not None:
            rows = []
            for v in bill_votes[:MAX_DETAIL_FETCHES]:
                ballots = client.list_all(
                    "/votes/ballots/", {"vote": v.get("url"), "politician": pol_path}, max_rows=1
                )
                row = _vote_row(v)
                row["their_ballot"] = ballots[0].get("ballot") if ballots else "not recorded"
                rows.append(row)
            return rows
        ballots = client.list_all("/votes/ballots/", {"politician": pol_path}, max_rows=limit)
        by_url: dict[str, dict[str, Any]] = {}
        for sess in sorted({str(b.get("vote_url", "")).split("/")[2] for b in ballots if b.get("vote_url")}):
            for v in client.list_cached("/votes/", {"session": sess}, ttl_s=3600):
                by_url[str(v.get("url"))] = v
        rows = []
        for b in ballots:
            found = by_url.get(str(b.get("vote_url")))
            row = (
                _vote_row(found) if found else {"vote": b.get("vote_url"), "url": site_url(b.get("vote_url"))}
            )
            row["their_ballot"] = b.get("ballot")
            rows.append(row)
        return rows

    votes = (
        bill_votes
        if bill_votes is not None
        else client.list_all("/votes/", {"session": session}, max_rows=limit)
    )
    rows = []
    for i, v in enumerate(votes[:limit]):
        row = _vote_row(v)
        if i < MAX_DETAIL_FETCHES:
            row["by_party"] = _party_breakdown(client.get(str(v.get("url"))))
        rows.append(row)
    return rows


# ── speeches ──


def _plain(content: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", content or ""))).strip()


def politician_speeches(
    client: ParliamentClient,
    *,
    politician: str,
    query: str | None = None,
    since: str | None = None,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], int]:
    """(matching speech excerpts, speeches scanned)."""
    if not re.fullmatch(r"[a-z0-9-]+", politician or ""):
        raise ParliamentArgsError("politician must be a slug from find_politician")
    params: dict[str, Any] = {"politician": f"/politicians/{politician}/"}
    if since:
        params["time__gte"] = since
    scan = MAX_SPEECHES_SCANNED if query else limit
    speeches = client.list_all("/speeches/", params, max_rows=scan)
    q = _tokens(query or "")
    rows: list[tuple[int, dict[str, Any]]] = []
    for s in speeches:
        if s.get("procedural"):
            continue
        text = _plain(str((s.get("content") or {}).get("en") or ""))
        topic = " / ".join(t for t in ((s.get("h1") or {}).get("en"), (s.get("h2") or {}).get("en")) if t)
        hits = len(q & _tokens(text + " " + topic)) if q else 1
        if q and not hits:
            continue
        excerpt = text
        if q:
            low = norm(text)
            pos = min((low.find(t) for t in q if low.find(t) >= 0), default=0)
            start = max(0, pos - 250)
            excerpt = ("…" if start else "") + text[start : start + 600]
        elif len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        rows.append(
            (
                hits,
                {
                    "date": str(s.get("time") or "")[:10],
                    "topic": topic or None,
                    "said": excerpt,
                    "url": site_url(s.get("url")),
                },
            )
        )
    # Most on-topic first, then newest; shown newest-first.
    rows.sort(key=lambda r: (-r[0], [-ord(c) for c in r[1]["date"]]))
    picked = sorted((r for _, r in rows[:limit]), key=lambda r: r["date"], reverse=True)
    return picked, len(speeches)
