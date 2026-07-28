"""Evidence-bearing surrender: naming what the loop actually searched.

When the loop can't answer it ends by referring the user out of the
product — "check with the relevant government departments" — while
holding the package ids, titles, and column inventories of everything
it just opened. This module turns that held state into a deterministic
footer appended to the surrender.

Everything here is pure. `collect_evidence` reads state the turn
already accumulated (`TurnTrace.packages_researched` / `.searches`,
`LoopState.search_results` / `.doc_columns` / `.doc_package`) and
`compose_footer` renders it as markdown. No model call, no query, no
new state — the cost of the whole feature is string concatenation.

Ordering is load-bearing: packages the loop actually *listed* come
first, because those are the ones it opened and can describe. Ranked
-but-unopened search candidates follow in retrieval order. The next
PRD in this milestone builds its suggestions from the head of that
list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.header_recovery import generated_header_ratio

if TYPE_CHECKING:  # circular-import guard: phases imports stay type-only
    from semantic_enrich.core.agent.phases import (
        ResearchResult,
        TurnContext,
    )

# Four datasets is as much as a footer can name before it reads as a
# dump rather than a statement; the remainder is counted, not listed.
MAX_PACKAGES = 4
MAX_QUERIES = 3
MAX_TITLE_CHARS = 70


@dataclass(frozen=True)
class EvidencePackage:
    """One dataset the turn touched. `title is None` renders the id;
    `column_count is None` omits the count — never a guess."""

    package_id: str
    title: str | None
    column_count: int | None
    # True when most of the package's column names are auto-generated
    # placeholders (`__col_1 …`) because the source header row was
    # never parsed at ingest. Worth surfacing rather than hiding: a
    # dataset whose columns cannot be named cannot be queried by name,
    # and a user who learns that from the footer has been spared the
    # three turns it otherwise takes to find out.
    headers_unnamed: bool = False


@dataclass(frozen=True)
class SearchEvidence:
    packages: tuple[EvidencePackage, ...]  # listed first, then ranked
    queries_tried: tuple[str, ...]
    truncated: int  # packages omitted by MAX_PACKAGES


def collect_evidence(
    ctx: TurnContext, result: ResearchResult | None
) -> SearchEvidence:
    """Extract what this turn searched, from state it already holds."""
    listed: list[str] = []
    for pid in (result.packages_cited if result else []):
        if pid and pid not in listed:
            listed.append(pid)
    for pid in ctx.trace.packages_researched:
        if pid and pid not in listed:
            listed.append(pid)

    titles = titles_by_package(ctx)
    by_package = columns_by_package(ctx)
    counts = {pid: len(cols) for pid, cols in by_package.items()}

    ordered = list(listed)
    for pid in titles:
        # Ranked-but-unopened candidates, in retrieval order: dict
        # insertion order is the order the searches ran and the order
        # each search ranked its candidates.
        if pid not in ordered:
            ordered.append(pid)

    packages = tuple(
        EvidencePackage(
            package_id=pid,
            title=titles.get(pid),
            column_count=counts.get(pid),
            headers_unnamed=_mostly_unnamed(
                by_package.get(pid, []), ctx.deps.settings
            ),
        )
        for pid in ordered[:MAX_PACKAGES]
    )
    return SearchEvidence(
        packages=packages,
        queries_tried=_queries_tried(ctx),
        truncated=max(0, len(ordered) - MAX_PACKAGES),
    )


def compose_footer(evidence: SearchEvidence) -> str:
    """Render the footer, or `""` when there is nothing to say.

    The return value is appended verbatim, so it carries its own
    leading blank line — the footer must never fuse into the model's
    final paragraph.
    """
    if not evidence.packages:
        # Zero packages is the below-floor case that should be
        # clarifying anyway. An empty header is worse than nothing.
        return ""
    items = [_render_package(p) for p in evidence.packages]
    if evidence.truncated > 0:
        items.append(f"+{evidence.truncated} more")
    lines = ["**What I searched:** " + " · ".join(items)]
    if evidence.queries_tried:
        terms = ", ".join(f'"{q}"' for q in evidence.queries_tried)
        lines.append(f"**Search terms tried:** {terms}")
    return "\n\n" + "\n\n".join(lines)


# ── extraction helpers ──


def titles_by_package(ctx: TurnContext) -> dict[str, str | None]:
    """package_id → title, from every tool payload that carries one.

    Two sources, and both are needed. `search_datasets` results give
    titles in retrieval order, which is most turns. But a *scoped* turn
    has no search at all — a clicked chip goes straight to
    `list_documents` — so on those the only title available is the one
    `list_documents` recorded per document. Reading search results alone
    made every scoped turn render raw package uuids in place of names,
    in the footer and in the chips built from it.
    """
    titles: dict[str, str | None] = {}
    for payload in ctx.state.search_results.values():
        for candidate in payload.get("candidates", []):
            pid = str(candidate.get("package_id", ""))
            if not pid:
                continue
            title = candidate.get("title")
            cleaned = _clean(title) if isinstance(title, str) else ""
            titles.setdefault(pid, cleaned or None)
    for doc_id, pid in ctx.state.doc_package.items():
        raw = ctx.state.doc_title.get(doc_id)
        cleaned = _clean(raw) if isinstance(raw, str) else ""
        titles.setdefault(pid, cleaned or None)
    return titles


def columns_by_package(ctx: TurnContext) -> dict[str, list[str]]:
    """package_id → the distinct column names across every document
    listed for it, in listing order.

    An earlier version read only the *first* document per package, on
    the reasoning that `list_documents` sorts clean documents first so
    the first one is what the model was steered at. That undercounts
    badly on a real package: one housing dataset reported "(3 columns)"
    in the footer while its eight documents — a different breakdown per
    table — carried nine distinct columns between them. A user reading
    "3 columns" and then seeing nine has been told something false
    about the dataset, and the browse chip built on the same number
    would promise the wrong count.

    Deduped by name, because heterogeneous documents in one package
    routinely repeat a column.
    """
    out: dict[str, list[str]] = {}
    for doc_id, columns in ctx.state.doc_columns.items():
        pid = ctx.state.doc_package.get(doc_id)
        if not pid:
            continue
        seen = out.setdefault(pid, [])
        for column in columns:
            if column not in seen:
                seen.append(column)
    return out


def mostly_unnamed(ctx: TurnContext, package_id: str) -> bool:
    """Whether this package's column names are mostly placeholders."""
    return _mostly_unnamed(
        columns_by_package(ctx).get(package_id, []), ctx.deps.settings
    )


def _mostly_unnamed(columns: list[str], settings: Settings) -> bool:
    """Reuses the same ratio and threshold `list_documents` uses to
    demote a garbage-header document, so the footer, the chip, and the
    tool all agree on what "unnamed" means rather than growing three
    definitions that drift."""
    if not columns:
        return False
    ratio: float = generated_header_ratio(columns)
    return ratio > settings.agent_generated_header_ratio


def _queries_tried(ctx: TurnContext) -> tuple[str, ...]:
    seen: list[str] = []
    for search in ctx.trace.searches:
        query = search.get("query")
        if not isinstance(query, str):
            continue
        cleaned = _clean(query)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen[:MAX_QUERIES])


# ── rendering helpers ──


def _render_package(package: EvidencePackage) -> str:
    label = package.title or package.package_id
    name = f"*{_truncate(label)}*" if package.title else f"`{label}`"
    if package.column_count is None:
        return name
    unit = "column" if package.column_count == 1 else "columns"
    note = ", most unnamed" if package.headers_unnamed else ""
    return f"{name} ({package.column_count} {unit}{note})"


def _truncate(text: str) -> str:
    if len(text) <= MAX_TITLE_CHARS:
        return text
    return text[: MAX_TITLE_CHARS - 1].rstrip() + "…"


def _clean(text: str) -> str:
    """Collapse whitespace and drop question marks.

    Dropping `?` is not cosmetic. The turn outcome is derived partly by
    testing `"?" in message`, and the footer is appended to that
    message — so a dataset title or a model-authored search query
    carrying a question mark could flip a `no_data` turn to
    `clarified`. The pipeline already computes the outcome before
    appending, so this cannot bite today; stripping the character makes
    the invariant hold by construction for every future caller instead
    of by luck.
    """
    return " ".join(text.replace("?", "").split())
