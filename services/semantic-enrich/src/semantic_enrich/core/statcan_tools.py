"""Live Statistics Canada tools: find a table, read its shape, fetch
series values.

The warehouse holds open.canada.ca's CSV corpus (program spending,
grants, contracts, travel). The headline economic questions people
actually ask — inflation, GDP, population growth, housing starts,
trade by partner, government debt and spending by function — live in
StatCan's statistical tables, which the Web Data Service serves live.
These three tools read them at question time; nothing is ingested.

Three steps, mirroring the warehouse flow:

1. `search_statcan_tables(query)`   → ranked table candidates
2. `describe_statcan_table(id)`     → dimensions and member ids
3. `get_statcan_data(id, series)`   → values for chosen series

Search is lexical (IDF-weighted title overlap plus a small synonym map)
over the WDS catalogue held in memory: no embedding cost, no index to
build or keep fresh, and StatCan titles are literal enough that word
overlap ranks well. The model reformulates when it doesn't.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

MAX_SERIES_PER_CALL = 12
MAX_POINTS_PER_SERIES = 240
MAX_LATEST_N = 1000
DEFAULT_LATEST_N = 12
DEFAULT_MAX_MEMBERS = 40

# frequencyCode → (label, periods per year). From WDS getCodeSets.
_FREQUENCIES: dict[int, tuple[str, float]] = {
    1: ("daily", 365.0),
    2: ("weekly", 52.0),
    4: ("every 2 weeks", 26.0),
    6: ("monthly", 12.0),
    7: ("every 2 months", 6.0),
    9: ("quarterly", 4.0),
    10: ("3 times a year", 3.0),
    11: ("semi-annual", 2.0),
    12: ("annual", 1.0),
    13: ("every 2 years", 0.5),
    14: ("every 3 years", 1 / 3),
    15: ("every 4 years", 0.25),
    16: ("every 5 years", 0.2),
    17: ("every 10 years", 0.1),
    18: ("occasional", 1.0),
    19: ("occasional quarterly", 4.0),
    20: ("occasional monthly", 12.0),
    21: ("occasional daily", 365.0),
}

_STOPWORDS = frozenset(
    {
        "a",
        "actually",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "canada",
        "canadian",
        "canadians",
        "change",
        "changed",
        "did",
        "do",
        "does",
        "each",
        "every",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "last",
        "many",
        "much",
        "of",
        "on",
        "or",
        "over",
        "since",
        "than",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "versus",
        "vs",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "year",
        "years",
    }
)

# Plain-language → StatCan title vocabulary. Expansion only adds terms;
# the user's own words still score.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "inflation": ("consumer", "price", "index"),
    "cpi": ("consumer", "price", "index"),
    "price": ("price", "index"),
    "grocery": ("consumer", "price", "index"),
    "groceries": ("consumer", "price", "index"),
    "food": ("food",),
    "gdp": ("gross", "domestic", "product"),
    "economy": ("gross", "domestic", "product"),
    "immigration": ("immigrant", "demographic", "growth", "component"),
    "immigrant": ("immigrant", "demographic", "component"),
    "population": ("population", "estimate", "demographic"),
    "job": ("employment", "labour", "force"),
    "jobs": ("employment", "labour", "force"),
    "unemployment": ("unemployment", "labour", "force"),
    "wage": ("wage", "earning"),
    "wages": ("wage", "earning"),
    "salary": ("wage", "earning"),
    "home": ("housing", "dwelling"),
    "homes": ("housing", "dwelling", "start"),
    "house": ("housing", "dwelling"),
    "build": ("start", "construction", "completion"),
    "building": ("start", "construction", "completion"),
    "built": ("start", "construction", "completion"),
    "rent": ("rent", "rental"),
    "debt": ("debt", "liability", "liabilities"),
    "owe": ("debt", "liability", "liabilities"),
    "tariff": ("custom", "duty", "import", "tax", "international", "trade"),
    "tariffs": ("custom", "duty", "import", "tax", "international", "trade"),
    "export": ("export", "merchandise", "trade"),
    "exports": ("export", "merchandise", "trade"),
    "import": ("import", "merchandise", "trade"),
    "imports": ("import", "merchandise", "trade"),
    "healthcare": ("health", "expenditure", "function", "government"),
    "health": ("health",),
    "spend": ("expenditure", "expense"),
    "spending": ("expenditure", "expense"),
    "spends": ("expenditure", "expense"),
    "revenue": ("revenue",),
    "federal": ("federal", "government"),
    "government": ("government",),
    "purchasing": ("real", "income", "disposable", "price"),
    "affordability": ("price", "income"),
    "household": ("household",),
    "income": ("income",),
}

# Multi-word phrases whose words mislead on their own ("purchasing
# power" is not electric power). Rewritten before tokenizing.
_PHRASES: dict[str, str] = {
    "purchasing power": "real disposable income consumer price index",
    "cost of living": "consumer price index",
    "food inflation": "consumer price index",
    "food prices": "consumer price index",
    "grocery prices": "consumer price index",
    "per capita": "per capita",
    "national debt": "central government debt",
    "trade partner": "merchandise trade country",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.casefold()) if t not in _STOPWORDS]


def query_terms(query: str) -> set[str]:
    folded = query.casefold()
    for phrase, replacement in _PHRASES.items():
        folded = folded.replace(phrase, replacement)
    raw = _TOKEN_RE.findall(folded)
    terms = {_stem(t) for t in raw if t not in _STOPWORDS}
    for t in raw:
        for syn in _SYNONYMS.get(t, ()):
            terms.add(_stem(syn))
    return terms


def table_id(product_id: int) -> str:
    """18100004 → '18-10-0004-01', the id StatCan prints on every table."""
    s = str(product_id)
    return f"{s[0:2]}-{s[2:4]}-{s[4:8]}-01"


def table_url(product_id: int) -> str:
    return f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={product_id}01"


def parse_product_id(value: Any) -> int:
    """Accept 18100004, '18100004', '18-10-0004-01' or '1810000401'."""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 10:
        digits = digits[:8]
    if len(digits) != 8:
        raise ValueError(
            f"not a StatCan table id: {value!r} "
            "(expected an 8-digit product_id like 18100004 or a table id like 18-10-0004-01)"
        )
    return int(digits)


# ── search ──


@dataclass
class _Index:
    source_id: int
    cubes: list[dict[str, Any]]
    tokens: list[set[str]]
    idf: dict[str, float]


_INDEX: _Index | None = None


# Words people use for a table that its title never says. Folded into
# the table's searchable tokens, so "tariff revenue" finds federal
# government finance and "who holds the debt" finds the holdings table.
TABLE_ALIASES: dict[int, str] = {
    18100004: "inflation cost living groceries grocery food prices shelter rent gasoline",
    18100005: "inflation cost living annual",
    36100706: "gdp per capita real household disposable income per person living standards productivity",
    36100663: "household income consumption saving quintile average household wealth",
    17100008: "immigration immigrants population growth non-permanent residents births deaths emigration",
    17100009: "population provinces quarterly per capita",
    34100126: "housing starts completions homes built supply province annual",
    12100171: "exports imports trade partner united states china share",
    10100016: "federal revenue customs import duties tariffs taxes expense deficit surplus debt",
    10100024: (
        "federal government spending function health healthcare defence education "
        "social protection by level of government"
    ),
    10100005: "all governments combined consolidated spending function health defence education",
    10100002: "federal debt national debt owe liabilities",
    36100673: (
        "who holds owns government debt bonds holders securities non-residents foreign "
        "(issuer: general governments; federal alone is not published)"
    ),
    18100007: "cpi basket weights share spending food shelter contribution expenses",
    14100287: "jobs unemployment rate employment labour force",
}


def _index(cubes: list[dict[str, Any]]) -> _Index:
    global _INDEX
    if _INDEX is not None and _INDEX.source_id == id(cubes):
        return _INDEX
    tokens = [
        set(_tokens(str(c.get("cubeTitleEn") or "")))
        | set(_tokens(TABLE_ALIASES.get(int(c.get("productId") or 0), "")))
        for c in cubes
    ]
    df: dict[str, int] = {}
    for ts in tokens:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(cubes))
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    _INDEX = _Index(source_id=id(cubes), cubes=cubes, tokens=tokens, idf=idf)
    return _INDEX


def search_tables(cubes: list[dict[str, Any]], query: str, *, k: int = 8) -> list[dict[str, Any]]:
    idx = _index(cubes)
    terms = query_terms(query)
    if not terms:
        return []
    literal = {_stem(t) for t in _TOKEN_RE.findall(query.casefold())} & terms
    scored: list[tuple[float, int]] = []
    for i, ts in enumerate(idx.tokens):
        hit = terms & ts
        if not hit:
            continue
        score = sum(idx.idf.get(t, 0.0) * (1.5 if t in literal else 1.0) for t in hit)
        # Short, specific titles beat long ones that mention everything.
        score /= 1.0 + 0.04 * len(ts)
        cube = idx.cubes[i]
        if str(cube.get("archived")) == "2":
            score *= 1.6
        end_year = _year(cube.get("cubeEndDate"))
        if end_year is not None and end_year >= datetime.now(UTC).year - 2:
            score *= 1.2
        scored.append((score, i))
    scored.sort(reverse=True)
    return [_candidate(idx.cubes[i], score) for score, i in scored[:k]]


def _candidate(cube: dict[str, Any], score: float) -> dict[str, Any]:
    pid = int(cube["productId"])
    return {
        "product_id": pid,
        "table_id": table_id(pid),
        "title": cube.get("cubeTitleEn"),
        "frequency": _FREQUENCIES.get(int(cube.get("frequencyCode") or 0), ("unknown", 1.0))[0],
        "start": _ym(cube.get("cubeStartDate")),
        "end": _ym(cube.get("cubeEndDate")),
        "current": str(cube.get("archived")) == "2",
        "url": table_url(pid),
        "score": round(score, 2),
    }


# ── describe ──


def describe_table(
    meta: dict[str, Any],
    codes: dict[str, Any] | None,
    *,
    member_filter: str | None = None,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> dict[str, Any]:
    pid = int(meta["productId"])
    uoms = _uom_names(codes)
    filter_terms = query_terms(member_filter) if member_filter else set()
    dims_out: list[dict[str, Any]] = []
    for dim in meta.get("dimension") or []:
        members = dim.get("member") or []
        chosen, truncated = _pick_members(members, filter_terms, max_members)
        dims_out.append(
            {
                "position": dim.get("dimensionPositionId"),
                "name": dim.get("dimensionNameEn"),
                "member_count": len(members),
                "members": [_member_out(m, uoms) for m in chosen],
                "truncated": truncated,
            }
        )
    freq = _FREQUENCIES.get(int(meta.get("frequencyCode") or 0), ("unknown", 1.0))[0]
    return {
        "product_id": pid,
        "table_id": table_id(pid),
        "title": meta.get("cubeTitleEn"),
        "frequency": freq,
        "start": _ym(meta.get("cubeStartDate")),
        "end": _ym(meta.get("cubeEndDate")),
        "current": str(meta.get("archiveStatusCode")) == "2",
        "url": table_url(pid),
        "dimensions": dims_out,
    }


def _pick_members(
    members: list[dict[str, Any]], filter_terms: set[str], cap: int
) -> tuple[list[dict[str, Any]], bool]:
    if len(members) <= cap:
        return members, False
    by_id = {m.get("memberId"): m for m in members}

    def depth(m: dict[str, Any]) -> int:
        d, seen = 0, set()
        parent = m.get("parentMemberId")
        while parent is not None and parent in by_id and parent not in seen:
            seen.add(parent)
            d += 1
            parent = by_id[parent].get("parentMemberId")
        return d

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for order, m in enumerate(members):
        name_terms = set(_tokens(str(m.get("memberNameEn") or "")))
        match = len(filter_terms & name_terms) if filter_terms else 0
        # Matches first, then shallow members (totals and top-level
        # groups), then catalogue order.
        ranked.append((-(match * 10) + depth(m), order, m))
    ranked.sort(key=lambda r: (r[0], r[1]))
    chosen = sorted((r[2] for r in ranked[:cap]), key=lambda m: members.index(m))
    return chosen, True


def _member_out(m: dict[str, Any], uoms: dict[int, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": m.get("memberId"), "name": m.get("memberNameEn")}
    if m.get("parentMemberId") is not None:
        out["parent"] = m.get("parentMemberId")
    uom = m.get("memberUomCode")
    if isinstance(uom, int) and uoms.get(uom):
        out["unit"] = uoms[uom]
    return out


# ── data ──


def build_coordinates(meta: dict[str, Any], series: list[list[int]]) -> tuple[list[str], list[str]]:
    """Validate member ids against the table and return (coordinates,
    labels). WDS coordinates always have ten positions; unused ones are
    0."""
    dims = meta.get("dimension") or []
    if not series:
        raise ValueError("series must name at least one member-id list")
    if len(series) > MAX_SERIES_PER_CALL:
        raise ValueError(f"at most {MAX_SERIES_PER_CALL} series per call")
    coords: list[str] = []
    labels: list[str] = []
    unique: list[list[int]] = []
    for s in series:
        if s not in unique:
            unique.append(s)
    for s in unique:
        if len(s) != len(dims):
            names = ", ".join(str(d.get("dimensionNameEn")) for d in dims)
            raise ValueError(
                f"each series needs exactly {len(dims)} member ids, one per dimension in order ({names}); "
                f"got {s}"
            )
        parts: list[str] = []
        for dim, member_id in zip(dims, s, strict=True):
            member = next(
                (m for m in dim.get("member") or [] if m.get("memberId") == member_id),
                None,
            )
            if member is None:
                raise ValueError(
                    f"member id {member_id} is not in dimension {dim.get('dimensionNameEn')!r}; "
                    "call describe_statcan_table (with member_filter) to see valid ids"
                )
            parts.append(str(member.get("memberNameEn")))
        coords.append(".".join(str(x) for x in [*s, *([0] * (10 - len(s)))]))
        labels.append("; ".join(parts))
    return coords, labels


MAX_PROBES = 40


def neighbour_series(meta: dict[str, Any], series: list[int]) -> list[list[int]]:
    """Every series that differs from `series` in exactly one dimension,
    nearest dimensions last-first (measure/unit dimensions tend to sit
    late, and that is where an unpublished combination usually goes
    wrong), capped so one probe stays one request."""
    dims = meta.get("dimension") or []
    out: list[list[int]] = []
    for i in reversed(range(len(dims))):
        for m in dims[i].get("member") or []:
            mid = m.get("memberId")
            if isinstance(mid, int) and mid != series[i]:
                out.append([*series[:i], mid, *series[i + 1 :]])
            if len(out) >= MAX_PROBES:
                return out
    return out


def build_coordinates_unchecked(meta: dict[str, Any], series: list[list[int]]) -> tuple[list[str], list[str]]:
    """`build_coordinates` without the per-call series cap, for probes
    whose members come from the table's own metadata."""
    dims = meta.get("dimension") or []
    names = [{m.get("memberId"): str(m.get("memberNameEn")) for m in d.get("member") or []} for d in dims]
    coords = [".".join(str(x) for x in [*s, *([0] * (10 - len(s)))]) for s in series]
    labels = ["; ".join(names[i].get(mid, "?") for i, mid in enumerate(s)) for s in series]
    return coords, labels


def latest_n_for(frequency_code: int, start_period: str | None, latest_n: int | None) -> int:
    if start_period:
        start = _parse_period(start_period)
        if start is None:
            raise ValueError(f"start_period {start_period!r} must look like 2015, 2015-06 or 2015-06-01")
        per_year = _FREQUENCIES.get(frequency_code, ("", 1.0))[1]
        years = max(0.0, (date.today() - start).days / 365.25)
        n = math.ceil(years * per_year) + math.ceil(per_year) + 2
        return max(1, min(MAX_LATEST_N, n))
    return max(1, min(MAX_LATEST_N, latest_n or DEFAULT_LATEST_N))


def shape_series(
    objects: list[dict[str, Any]],
    labels: list[str],
    *,
    frequency_code: int,
    codes: dict[str, Any] | None,
    start_period: str | None,
    end_period: str | None,
    keep_last: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (series for the model, flat rows for the evidence UI).

    Periods with no value (StatCan's ".." / "x") are dropped and counted:
    a latest month that is not yet published is not a zero, and handing
    it to the model as null read as "the data is missing"."""
    uoms = _uom_names(codes)
    scalars = _scalar_names(codes)
    symbols = _symbol_names(codes)
    start = _parse_period(start_period) if start_period else None
    end = _parse_period(end_period, upper=True) if end_period else None
    series_out: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for obj, label in zip(objects, labels, strict=True):
        if obj.get("missing"):
            series_out.append(
                {
                    "series": label,
                    "coordinate": obj.get("coordinate"),
                    "missing": True,
                    "note": (
                        "StatCan does not publish this member combination. Tables "
                        "only carry some combinations (e.g. a per-capita measure "
                        "may exist in current prices but not chained dollars): "
                        "change one member at a time and retry."
                    ),
                }
            )
            continue
        points = []
        unit = None
        scalar = None
        empty = 0
        for p in obj.get("vectorDataPoint") or []:
            ref = _parse_period(str(p.get("refPer") or ""))
            if ref is None:
                continue
            if p.get("value") is None:
                empty += 1
                continue
            if start is not None and ref < start:
                continue
            if end is not None and ref > end:
                continue
            code = int(p.get("scalarFactorCode") or 0)
            if code:
                scalar = scalars.get(code, scalar)
            sym = symbols.get(str(p.get("symbolCode") or "0"))
            point = {
                "period": _fmt_period(ref, frequency_code),
                "value": scale_value(p.get("value"), code),
            }
            if sym:
                point["flag"] = sym
            points.append(point)
        uom_code = obj.get("memberUomCode")
        if isinstance(uom_code, int):
            unit = uoms.get(uom_code)
        if keep_last is not None:
            points = points[-keep_last:]
        thinned = False
        if len(points) > MAX_POINTS_PER_SERIES:
            points, thinned = _thin(points), True
        entry: dict[str, Any] = {
            "series": label,
            "vector": f"v{obj.get('vectorId')}" if obj.get("vectorId") else None,
            "unit": unit,
            # Values are already multiplied out; this records what StatCan
            # published them in, so a reader can match the table.
            "published_in": scalar,
            "points": [[pt["period"], pt["value"], *([pt["flag"]] if "flag" in pt else [])] for pt in points],
        }
        if points:
            # Endpoints up front: a model reading a 50-point array picks
            # the wrong ends more often than you would think.
            entry["first"] = [points[0]["period"], points[0]["value"]]
            entry["latest"] = [points[-1]["period"], points[-1]["value"]]
        else:
            entry["note"] = (
                "no values for this member combination in the window; the "
                "combination may not exist in this table. Try other members "
                "(describe_statcan_table with member_filter) or a wider window."
            )
        if empty:
            entry["unpublished_periods_skipped"] = empty
        if thinned:
            entry["thinned"] = (
                "long monthly range: kept the first point, the last point, and one point per year "
                "in the last point's month; narrow start_period for every month"
            )
        series_out.append(entry)
        for pt in points:
            rows.append(
                {
                    "series": label,
                    "period": pt["period"],
                    "value": pt["value"],
                    "unit": unit,
                }
            )
    return series_out, rows


def scale_value(value: Any, scalar_code: int) -> Any:
    """StatCan publishes many money series "in thousands" or "in
    millions". Multiplying out here, exactly, means the model only ever
    sees base units and cannot forget a scalar."""
    if not isinstance(value, int | float) or isinstance(value, bool) or scalar_code == 0:
        return value
    scaled = Decimal(str(value)).scaleb(scalar_code)
    return int(scaled) if scaled == scaled.to_integral_value() else float(scaled)


def _thin(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_month = str(points[-1]["period"])[5:7]
    kept = [
        p
        for i, p in enumerate(points)
        if i == 0 or i == len(points) - 1 or str(p["period"])[5:7] == last_month
    ]
    return kept[-MAX_POINTS_PER_SERIES:]


# ── helpers ──


def _uom_names(codes: dict[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for u in (codes or {}).get("uom") or []:
        if isinstance(u.get("memberUomCode"), int) and u.get("memberUomEn"):
            out[u["memberUomCode"]] = u["memberUomEn"]
    return out


def _scalar_names(codes: dict[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for s in (codes or {}).get("scalar") or []:
        if isinstance(s.get("scalarFactorCode"), int):
            out[s["scalarFactorCode"]] = s.get("scalarFactorDescEn") or ""
    return out


def _symbol_names(codes: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in (codes or {}).get("symbol") or []:
        code = str(s.get("symbolCode"))
        if code != "0" and s.get("symbolDescEn"):
            out[code] = s["symbolDescEn"]
    return out


def _year(value: Any) -> int | None:
    m = re.match(r"(\d{4})", str(value or ""))
    return int(m.group(1)) if m else None


def _ym(value: Any) -> str | None:
    m = re.match(r"(\d{4})-(\d{2})", str(value or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _parse_period(value: str, *, upper: bool = False) -> date | None:
    value = value.strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", value)
    if m:
        y = int(m.group(1))
        if m.group(2) is None:
            return date(y, 12, 31) if upper else date(y, 1, 1)
        mo = int(m.group(2))
        if m.group(3) is None:
            if upper:
                nxt = date(y + (mo == 12), mo % 12 + 1, 1)
                return date.fromordinal(nxt.toordinal() - 1)
            return date(y, mo, 1)
        return date(y, mo, int(m.group(3)))
    q = re.fullmatch(r"(\d{4})-?Q([1-4])", value, re.IGNORECASE)
    if q:
        y, qn = int(q.group(1)), int(q.group(2))
        return date(y, 3 * qn, 30 if qn in (2, 3) else 31) if upper else date(y, 3 * qn - 2, 1)
    return None


def _fmt_period(d: date, frequency_code: int) -> str:
    if frequency_code in (6, 7, 20):
        return f"{d.year}-{d.month:02d}"
    if frequency_code in (9, 19):
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    if frequency_code in (12, 13, 14, 15, 16, 17) and d.month == 1 and d.day == 1:
        return str(d.year)
    return d.isoformat()
