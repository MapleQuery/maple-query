"""Client-supplied package scope for exploration turns.

A turn can say "explore *these* datasets". The scope arrives on the
request, so it is untrusted: validation is defensive and non-raising,
and a bad scope degrades to an ordinary unscoped turn rather than to an
error. It is an optimisation hint from a UI, not an authorisation
boundary — it selects among packages the user could already reach by
searching.

Two pure functions: `sanitize` (validate + cap) and `render_hint`
(template the scope into one system hint). Neither touches state.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

# Four is the same ceiling 11.1's evidence footer names, and the same
# order of magnitude as the suggestion cap: a scope wider than this is
# not a scope.
MAX_SCOPE_PACKAGES = 4

# Conservative shape check. Package ids in this corpus are CKAN uuids,
# but the pattern stays permissive enough to survive an id-format change
# upstream — the point is to reject prose and injection attempts, not to
# assert the id exists. `list_documents` resolves it against the
# warehouse either way.
_PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,}$")

_MAX_TITLE_CHARS = 70


def sanitize(raw: Any) -> tuple[str, ...]:
    """One client-supplied scope → the ids worth honouring.

    Malformed entries are dropped silently, never raised. An empty
    result is an ordinary unscoped turn — i.e. exactly the pre-PRD
    behaviour.
    """
    if not isinstance(raw, (list, tuple)):
        return ()
    kept: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip().lower()
        if not _PACKAGE_ID_RE.match(candidate):
            continue
        if candidate not in kept:
            kept.append(candidate)
        if len(kept) == MAX_SCOPE_PACKAGES:
            break
    return tuple(kept)


def render_hint(
    scope: Iterable[str], *, titles: Mapping[str, str | None]
) -> str:
    """The scope as one system-hint line.

    Deliberately a *preference*, not a filter: the model may still
    search broadly if the scope turns up nothing. Hard-filtering
    retrieval on a client-supplied id list would turn a UI hint into a
    correctness constraint, and a stale chip — from a notebook block
    whose dataset was re-ingested — would produce a dead turn instead of
    a degraded one.
    """
    rendered = [_render_one(pid, titles.get(pid)) for pid in scope]
    if not rendered:
        return ""
    return (
        "The user is exploring these datasets specifically: "
        + ", ".join(rendered)
        + ". Prefer `list_documents`, `search_columns`, and "
        "`sample_rows` scoped to them. Do not re-run a broad dataset "
        "search unless these turn up nothing relevant."
    )


def titles_from_records(records: Any) -> dict[str, str | None]:
    """package_id → title, harvested from client-echoed turn records.

    Records are client-supplied and inspected defensively — a title is
    cosmetic here, so anything malformed degrades to rendering the raw
    id rather than failing the turn.
    """
    titles: dict[str, str | None] = {}
    if not isinstance(records, list):
        return titles
    for record in records:
        if not isinstance(record, dict):
            continue
        for package in record.get("packages") or []:
            if not isinstance(package, dict):
                continue
            pid = package.get("package_id")
            if not isinstance(pid, str) or not pid:
                continue
            title = package.get("title")
            titles.setdefault(
                pid, title if isinstance(title, str) and title else None
            )
    return titles


def _render_one(package_id: str, title: str | None) -> str:
    if not title:
        return f"`{package_id}`"
    clean = " ".join(title.split())
    if len(clean) > _MAX_TITLE_CHARS:
        clean = clean[: _MAX_TITLE_CHARS - 1].rstrip() + "…"
    return f"*{clean}* (`{package_id}`)"
