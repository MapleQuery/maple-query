"""Live open.canada.ca DataStore tools: find a table, read its columns,
filter and aggregate it.

open.canada.ca's DataStore holds the full proactive-disclosure tables —
every grant, contribution, contract over $10K, travel and hospitality
claim — including files too large to have been mirrored into the
warehouse. CKAN filters them server-side (exact match per column, plus
word search scoped to a column); everything else — ranges, amendment
de-duplication, group-and-sum — happens here over the filtered rows, so
a question like "grants to Ukraine by year" is one call.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

from semantic_enrich.clients.opencanada import OpenCanadaClient

PAGE_SIZE = 10_000
MAX_ROWS = 30_000
# Date-window reads (newest-first, stop at the window's start).
WINDOW_PAGE_SIZE = 32_000
WINDOW_MAX_ROWS = 100_000
WINDOW_TIME_BUDGET_S = 20.0
MAX_RETURN_ROWS = 50
MAX_GROUPS = 50

# Tables people ask about most, so the model can go straight to them.
KNOWN_RESOURCES: dict[str, str] = {
    "1d15a62f-5656-49ad-8c88-f40ce689d831": (
        "Proactive Disclosure - Grants and Contributions (all departments, 2005 on)"
    ),
    "fac950c0-00d5-4ec1-a4d3-9cbebf98a305": "Proactive Publication - Contracts over $10,000",
    "8282db2a-878f-475c-af10-ad56aa8fa72c": "Proactive Disclosure - Travel Expenses",
    "7b301f1a-2a7a-48bd-9ea9-e0ac4a5313ed": "Proactive Disclosure - Hospitality Expenses",
    "a811cac0-2a2a-4440-8a81-2994fc753171": "Annual Expenditures on Travel, Hospitality and Conferences",
    "bdaa5515-3782-4e5c-9d44-c25e032addb7": "Proactive Disclosure - Position Reclassification",
}

_NOTHING_TO_REPORT = re.compile(r"nothing to report|néant", re.IGNORECASE)
_DERIVED = re.compile(r"^(year|fiscal_year|month):([A-Za-z0-9_]+)$")
_OPS = {"=", "!=", ">", ">=", "<", "<=", "contains", "starts_with"}


def dataset_url(package_id: str, resource_id: str | None = None) -> str:
    base = f"https://open.canada.ca/data/en/dataset/{package_id}"
    return f"{base}/resource/{resource_id}" if resource_id else base


# ── search ──


def search_packages(client: OpenCanadaClient, query: str, *, k: int = 8) -> list[dict[str, Any]]:
    packages = client.package_search(query, rows=25)
    out: list[dict[str, Any]] = []
    for pkg in packages:
        resources = [
            {
                "resource_id": r.get("id"),
                "name": r.get("name"),
                "format": r.get("format"),
                "size_bytes": r.get("size"),
            }
            for r in pkg.get("resources") or []
            if r.get("datastore_active") and not _NOTHING_TO_REPORT.search(str(r.get("name") or ""))
        ]
        out.append(
            {
                "package_id": pkg.get("id"),
                "title": pkg.get("title"),
                "organization": (pkg.get("organization") or {}).get("title"),
                "queryable": bool(resources),
                "resources": resources[:5],
                "url": dataset_url(str(pkg.get("id"))),
            }
        )
    # Queryable packages first; CKAN's order otherwise.
    out.sort(key=lambda p: not p["queryable"])
    return out[:k]


# ── describe ──


def describe_resource(client: OpenCanadaClient, resource_id: str) -> dict[str, Any]:
    fields, total = client.resource_fields(resource_id)
    sample = client.datastore_search({"resource_id": resource_id, "limit": 3})
    meta = _resource_meta(client, resource_id)
    return {
        "resource_id": resource_id,
        "title": meta["title"],
        "url": meta["url"],
        "total_rows": total,
        "columns": [{"name": f.get("id"), "type": f.get("type")} for f in fields],
        "sample_rows": [
            {k: _clip(v) for k, v in row.items() if k != "_id"} for row in sample.get("records") or []
        ],
    }


def _resource_meta(client: OpenCanadaClient, resource_id: str) -> dict[str, str]:
    res = client.resource_show(resource_id)
    package_id = str(res.get("package_id") or "")
    name = str(res.get("name") or KNOWN_RESOURCES.get(resource_id) or resource_id)
    return {
        "title": name,
        "package_id": package_id,
        "url": dataset_url(package_id, resource_id) if package_id else str(res.get("url") or ""),
    }


def _clip(value: Any, n: int = 80) -> Any:
    if isinstance(value, str) and len(value) > n:
        return value[: n - 1] + "…"
    return value


# ── query ──


class QueryArgsError(ValueError):
    """An argument the model can fix (unknown column, bad operator)."""


def run_query(
    client: OpenCanadaClient,
    *,
    resource_id: str,
    filters: dict[str, Any] | None = None,
    text: dict[str, str] | None = None,
    where: list[dict[str, Any]] | None = None,
    fields: list[str] | None = None,
    group_by: list[str] | None = None,
    sum_columns: list[str] | None = None,
    dedupe: dict[str, str] | None = None,
    sort: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    columns, _total = client.resource_fields(resource_id)
    known = {str(c.get("id")) for c in columns}
    where = list(where or [])
    # A derived key ("fiscal_year:agreement_start_date") cannot be sent
    # to CKAN as an exact-match filter; it becomes a where condition.
    if filters:
        derived = {k: v for k, v in filters.items() if _DERIVED.match(k)}
        filters = {k: v for k, v in filters.items() if k not in derived}
        for k, v in derived.items():
            where.append({"column": k, "op": "=", "value": v})
    group_by = group_by or []
    sum_columns = sum_columns or []
    aggregate = bool(group_by or sum_columns)

    needed: list[str] = []

    def need(col: str, role: str) -> None:
        if col not in known:
            hint = ""
            if col in ("year", "fiscal_year", "month"):
                dates = sorted(k for k in known if "date" in k)
                hint = (
                    f" To group or filter by {col}, derive it from a date column: "
                    + ", ".join(f"'{col}:{d}'" for d in dates[:3])
                    + "."
                )
            raise QueryArgsError(
                f"{role} column {col!r} is not in this table.{hint} Columns are: {', '.join(sorted(known))}"
            )
        if col not in needed:
            needed.append(col)

    for col in filters or {}:
        need(col, "filter")
    for col in text or {}:
        need(col, "text")
    for cond in where:
        if cond.get("op") not in _OPS:
            raise QueryArgsError(f"where op must be one of {sorted(_OPS)}")
        m = _DERIVED.match(str(cond.get("column")))
        need(m.group(2) if m else str(cond.get("column")), "where")
    group_keys: list[tuple[str, str | None, str]] = []
    for g in group_by:
        m = _DERIVED.match(g)
        col, fn = (m.group(2), m.group(1)) if m else (g, None)
        need(col, "group_by")
        group_keys.append((g, fn, col))
    for col in sum_columns:
        need(col, "sum")
    if dedupe:
        need(str(dedupe.get("key")), "dedupe key")
        need(str(dedupe.get("order")), "dedupe order")
    for col in fields or []:
        need(col, "field")
    if not aggregate and not fields:
        needed = [str(c.get("id")) for c in columns]

    # A date window (a lower bound on one date column) lets a query that
    # matches too many rows to read still finish: sort newest-first and
    # stop paging once the window's start is passed. Good for recent
    # windows (the last few months of grants). A whole fiscal year of
    # grants is >160K rows and took 54 s, so reads stop at 100K rows /
    # 20 s and say so rather than holding the turn hostage.
    window = _date_window(where)
    # CKAN stores most money columns as text, so a server-side
    # "contract_value desc" is a string sort: every top contract came back
    # as $99,999.99. Numeric orderings happen here, over the rows read
    # (the window's rows, newest-first, when there is a window).
    local_sort: tuple[str, bool] | None = None
    if sort:
        col, _, direction = sort.partition(" ")
        types = {str(c.get("id")): str(c.get("type") or "") for c in columns}
        if col in types and types[col] == "text" and not (window and col == window[0]):
            need(col, "sort")
            local_sort = (col, direction.strip().lower() == "desc")
            sort = None
    if window and not sort:
        sort = f"{window[0]} desc"
    windowed = bool(window and sort and sort.split()[0] == window[0] and sort.lower().endswith("desc"))
    if windowed and window:
        need(window[0], "date window")

    params: dict[str, Any] = {"resource_id": resource_id, "fields": needed}
    if filters:
        params["filters"] = filters
    if text:
        params["q"] = text
    if sort:
        params["sort"] = sort

    rows: list[dict[str, Any]] = []
    total = 0
    offset = 0
    page_size = WINDOW_PAGE_SIZE if windowed else PAGE_SIZE
    read_cap = WINDOW_MAX_ROWS if windowed else MAX_ROWS
    page_target = read_cap if (aggregate or where or dedupe or local_sort) else max(limit, 1)
    started = time.monotonic()
    window_closed = False
    while True:
        page = client.datastore_search({**params, "offset": offset, "limit": min(page_size, page_target)})
        total = int(page.get("total") or 0)
        if aggregate and not windowed and total > MAX_ROWS:
            return {
                "status": "too_broad",
                "matched_rows": total,
                "message": (
                    f"{total:,} rows match; aggregation reads at most {MAX_ROWS:,}. "
                    "Narrow with filters (e.g. owner_org) or text, or bound a date "
                    "column (where >= / fiscal_year:<date col>) so it can read "
                    "newest-first, then retry."
                ),
            }
        records = page.get("records") or []
        rows.extend(records)
        offset += len(records)
        if windowed and window and records:
            oldest = str(records[-1].get(window[0]) or "")
            if oldest and oldest < window[1]:
                window_closed = True
                break
        if not records or offset >= total or len(rows) >= page_target:
            break
        if time.monotonic() - started > WINDOW_TIME_BUDGET_S:
            break
    if windowed and aggregate and not window_closed and offset < total:
        return {
            "status": "too_broad",
            "matched_rows": total,
            "message": (
                f"Read {offset:,} rows newest-first without reaching {window[1] if window else ''}; "
                "narrow the window or add filters, then retry."
            ),
        }

    fetched = len(rows)
    rows = [r for r in rows if all(_match(r, c) for c in where)]
    deduped = 0
    if dedupe:
        before = len(rows)
        rows = _latest(rows, str(dedupe["key"]), str(dedupe["order"]))
        deduped = before - len(rows)

    out: dict[str, Any] = {
        "status": "ok",
        "matched_rows": total,
        "rows_read": fetched,
        "rows_after_where": len(rows) + deduped,
        "amendments_collapsed": deduped,
    }
    if total > fetched:
        out["truncated"] = (
            f"read the first {fetched:,} of {total:,} matching rows; narrow with filters or text "
            "for a complete result"
        )
    if local_sort:
        col, desc = local_sort
        rows.sort(
            key=lambda r: (
                _num(r.get(col)) is None,
                -(_num(r.get(col)) or 0.0) if desc else (_num(r.get(col)) or 0.0),
            )
        )
        out["sorted_by"] = f"{col} ({'largest' if desc else 'smallest'} first, numeric)"
    if aggregate:
        groups = _aggregate(rows, group_keys, sum_columns)
        out["groups"] = groups[:MAX_GROUPS]
        out["group_count"] = len(groups)
        out["totals"] = {
            "rows": len(rows),
            **{f"sum_{c}": round(sum(_num(r.get(c)) or 0.0 for r in rows), 2) for c in sum_columns},
        }
    else:
        out["rows"] = [{k: _clip(v, 200) for k, v in r.items() if k != "_id"} for r in rows[:limit]]
    return out


def _date_window(where: list[dict[str, Any]]) -> tuple[str, str] | None:
    """(date column, earliest ISO date) when `where` bounds a date from
    below: `col >= 2024-04-01`, or a derived `fiscal_year:col = 2024-25`
    / `year:col = 2024`."""
    for cond in where:
        col, op, val = str(cond.get("column")), cond.get("op"), cond.get("value")
        m = _DERIVED.match(col)
        if m and op == "=" and not isinstance(val, list):
            year = re.match(r"(\d{4})", str(val))
            if not year:
                continue
            if m.group(1) == "fiscal_year":
                return m.group(2), f"{year.group(1)}-04-01"
            if m.group(1) == "year":
                return m.group(2), f"{year.group(1)}-01-01"
        if not m and op in (">=", ">") and re.match(r"\d{4}-\d{2}", str(val)):
            return col, str(val)
    return None


def _match(row: dict[str, Any], cond: dict[str, Any]) -> bool:
    col, op, want = str(cond.get("column")), cond.get("op"), cond.get("value")
    if isinstance(want, list):
        if op != "=":
            return False
        return any(_match(row, {**cond, "value": w}) for w in want)
    m = _DERIVED.match(col)
    have = _derive(m.group(1), row.get(m.group(2))) if m else row.get(col)
    if op == "contains":
        return str(want).casefold() in str(have or "").casefold()
    if op == "starts_with":
        return str(have or "").casefold().startswith(str(want).casefold())
    hn, wn = _num(have), _num(want)
    a: Any
    b: Any
    a, b = (hn, wn) if hn is not None and wn is not None else (str(have or ""), str(want))
    if op == "=":
        return bool(a == b)
    if op == "!=":
        return bool(a != b)
    if op == ">":
        return bool(a > b)
    if op == ">=":
        return bool(a >= b)
    if op == "<":
        return bool(a < b)
    return bool(a <= b)


def _latest(rows: list[dict[str, Any]], key: str, order: str) -> list[dict[str, Any]]:
    """One row per key: the one with the highest order value. Grants and
    contracts republish a row per amendment, each carrying the agreement's
    full value — summing them all counts one agreement several times."""
    best: dict[Any, dict[str, Any]] = {}
    for r in rows:
        k = r.get(key)
        cur = best.get(k)
        if cur is None or (_num(r.get(order)) or 0) >= (_num(cur.get(order)) or 0):
            best[k] = r
    return list(best.values())


def _derive(fn: str | None, value: Any) -> str:
    s = str(value or "")
    if fn is None:
        return s
    m = re.match(r"(\d{4})-(\d{2})", s)
    if not m:
        return "(unknown)"
    year, month = int(m.group(1)), int(m.group(2))
    if fn == "year":
        return str(year)
    if fn == "month":
        return f"{year}-{month:02d}"
    start = year if month >= 4 else year - 1  # federal fiscal year starts April 1
    return f"{start}-{str(start + 1)[2:]}"


def _aggregate(
    rows: list[dict[str, Any]],
    group_keys: list[tuple[str, str | None, str]],
    sum_columns: list[str],
) -> list[dict[str, Any]]:
    acc: dict[tuple[str, ...], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        key = tuple(_derive(fn, r.get(col)) for _, fn, col in group_keys)
        g = acc[key]
        g["count"] = g.get("count", 0) + 1
        for c in sum_columns:
            g[f"sum_{c}"] = g.get(f"sum_{c}", 0.0) + (_num(r.get(c)) or 0.0)
    out = []
    for key, vals in acc.items():
        row: dict[str, Any] = {name: k for (name, _, _), k in zip(group_keys, key, strict=True)}
        row["count"] = vals["count"]
        for c in sum_columns:
            row[f"sum_{c}"] = round(vals[f"sum_{c}"], 2)
        out.append(row)
    rank = f"sum_{sum_columns[0]}" if sum_columns else "count"
    # Time groupings read best in time order; everything else by size.
    if group_keys and all(fn is not None for _, fn, _ in group_keys):
        out.sort(key=lambda r: tuple(str(r[n]) for n, _, _ in group_keys))
        _flag_thin_tail(out, rank)
    else:
        out.sort(key=lambda r: r[rank], reverse=True)
    return out


def _flag_thin_tail(groups: list[dict[str, Any]], rank: str) -> None:
    """Mark trailing periods far below the ones before them.

    Disclosures land months after the fact, so the newest fiscal year
    usually holds a sliver of its eventual total. Reported as "last
    year's" figure, it understated aid to Ukraine thirtyfold. The flag
    is a fact about the numbers, deterministic, so the model does not
    have to notice it."""
    if len(groups) < 4:
        return
    for i in range(len(groups) - 1, max(len(groups) - 3, 2), -1):
        prior = sorted(float(g[rank]) for g in groups[i - 3 : i])
        median = prior[1]
        if median > 0 and float(groups[i][rank]) < 0.25 * median:
            groups[i]["likely_incomplete"] = True
        else:
            break


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[,$\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None
