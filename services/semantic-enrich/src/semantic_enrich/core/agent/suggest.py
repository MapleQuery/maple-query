"""Next-step suggestions: turning evidence into offers.

The evidence footer tells the user what was searched. This turns the
same extraction into two or three concrete next questions, each already
scoped to a package the loop inspected, so a dead end becomes something
clickable.

Everything here is a pure function over turn state — no model call, no
I/O, no cost on the un-accepted path, which is most turns. A model asked
to phrase offers varies its wording every turn, cannot reliably attach a
package scope, and spends tokens on turns nobody acts on; composed
offers are testable, diffable, free, and gate-able on hard signals
rather than on model judgement.

The builder is emit-only: nothing consumes its output until the chat and
notebook surfaces render it.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from semantic_enrich.core.agent.derivation_units import is_monetary_column
from semantic_enrich.core.agent.evidence import EvidencePackage, SearchEvidence

if TYPE_CHECKING:  # circular-import guard: phases imports stay type-only
    from semantic_enrich.core.agent.phases import (
        ResearchResult,
        TurnContext,
    )

SuggestionKind = Literal[
    "summarize_dataset", "list_columns", "sample_rows", "group_total"
]

# Button text, not prose: no markdown, because a chip renders its label
# literally and `*Title*` in a button reads as a typo.
MAX_LABEL_CHARS = 60

# Outcomes worth offering a next step on. A clean `answered` turn needs
# no recovery; `clarified` and `deflected` are the below-floor and
# off-scope cases, where an offer would be a false invitation.
OFFER_OUTCOMES = frozenset(
    {"no_data", "answered_with_caveat", "explored"}
)


@dataclass(frozen=True)
class Suggestion:
    kind: SuggestionKind
    label: str
    question: str
    package_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "question": self.question,
            "package_ids": list(self.package_ids),
        }


def build_suggestions(
    ctx: TurnContext,
    result: ResearchResult | None,
    evidence: SearchEvidence,
    outcome: str,
) -> list[Suggestion]:
    """Offers for this turn, in priority order, capped and deduped."""
    settings = ctx.deps.settings
    if not _offers_allowed(ctx, evidence, outcome):
        return []

    listed = {pid for pid in ctx.trace.packages_researched if pid}
    builders = (
        _summarize_dataset,
        _list_columns,
        _sample_rows,
        _group_total,
    )
    out: list[Suggestion] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for build in builders:
        suggestion = build(ctx, evidence, listed)
        if suggestion is None:
            continue
        key = (suggestion.kind, suggestion.package_ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(suggestion)
        if len(out) >= settings.agent_suggestions_max:
            break
    return out


# ── gating ──


def _offers_allowed(
    ctx: TurnContext, evidence: SearchEvidence, outcome: str
) -> bool:
    settings = ctx.deps.settings
    if not settings.agent_suggestions_enabled:
        return False
    if outcome not in OFFER_OUTCOMES:
        return False
    if not evidence.packages:
        return False
    if not _retrieval_was_sound(ctx):
        return False
    return _explore_chain_length(ctx) < settings.agent_explore_chain_max


def _retrieval_was_sound(ctx: TurnContext) -> bool:
    """The false-invitation guard: below the similarity floor the loop
    has no dataset worth exploring, and inviting the user into one is
    worse than a clean "I don't have this."

    A scoped turn passes without a search of its own. It has none — a
    clicked chip goes straight to `list_documents` — but its packages
    came from a previous turn's offers, which cleared this same gate
    before they were ever rendered. Re-applying it here would silently
    kill every follow-up offer in a chain.
    """
    if ctx.scope_package_ids:
        return True
    return any(
        search.get("retrieval_quality") == "ok"
        for search in ctx.trace.searches
    )


def _explore_chain_length(ctx: TurnContext) -> int:
    """How many explorations immediately precede this turn.

    Guided recovery should converge on an answer, not become a browsing
    mode: without a cap a user can ride chips indefinitely, which feels
    productive and is not. Records are client-supplied, so the walk is
    defensive and stops at the first thing it does not recognise.

    Chat-only by construction — only that surface echoes turn records.
    Notebook blocks send none, so every block arrives looking like turn
    one and the cap never engages there. That is deliberate: a chat chip
    is one frictionless click and the chain scrolls out of view, while a
    notebook follow-up costs two deliberate actions and stays on the
    page as a document. Counting preceding blocks would also be
    order-dependent, so reordering blocks would make the affordance
    appear and disappear.
    """
    length = 0
    for record in reversed(ctx.request.turn_records or []):
        if not isinstance(record, dict):
            break
        if record.get("outcome") != "explored":
            break
        length += 1
    return length


# ── the four kinds ──


def _summarize_dataset(
    ctx: TurnContext, evidence: SearchEvidence, listed: set[str]
) -> Suggestion | None:
    """The only kind that may point at a merely-ranked package.

    A summary is honest over a package the loop never opened: the
    explore turn's first move is `list_documents`, and a package with
    nothing in it produces "this dataset turned out to be empty" rather
    than a wrong answer. Restricting every kind to listed packages
    would have fired on a quarter of measured surrenders while
    retrieval was sound on all of them — screening out turns where the
    loop stopped one tool call short, not turns with nothing to offer.
    """
    package = evidence.packages[0]
    name = _display_name(package)
    return Suggestion(
        kind="summarize_dataset",
        label=_label("Summarize ", name),
        question=(
            f"Summarize what data is in {name} — what it covers, its "
            "time range, and its main columns."
        ),
        package_ids=(package.package_id,),
    )


def _list_columns(
    ctx: TurnContext, evidence: SearchEvidence, listed: set[str]
) -> Suggestion | None:
    """Browse, not search — the chip carries no filter term.

    Any term available here is derived from the searches this turn
    already ran, which are the searches that just failed. A chip that
    re-applies the failed filter re-runs the failure. What rescues the
    motivating case is a human scanning the column names and
    recognising the bucket the question actually lives in, and that
    needs the inventory rather than a filter over it.
    """
    minimum = ctx.deps.settings.agent_suggest_min_columns
    for package in evidence.packages:
        if package.package_id not in listed:
            continue
        if package.column_count is None or package.column_count < minimum:
            # Below the threshold `summarize_dataset` already enumerates
            # the columns, so this chip would return what the chip next
            # to it returns. `None` means the package was never opened,
            # so its size is unknown and it is not eligible either.
            continue
        name = _display_name(package)
        return Suggestion(
            kind="list_columns",
            label=_label(
                f"Show all {package.column_count} columns in ", name
            ),
            question=(
                f"List the columns in {name}, grouped by theme, so I "
                "can see what the dataset actually contains."
            ),
            package_ids=(package.package_id,),
        )
    return None


def _sample_rows(
    ctx: TurnContext, evidence: SearchEvidence, listed: set[str]
) -> Suggestion | None:
    for package in _listed_packages(evidence, listed):
        name = _display_name(package)
        return Suggestion(
            kind="sample_rows",
            label=_label("Sample rows from ", name),
            question=(
                f"Show me a sample of rows from {name} so I can see "
                "its structure."
            ),
            package_ids=(package.package_id,),
        )
    return None


def _group_total(
    ctx: TurnContext, evidence: SearchEvidence, listed: set[str]
) -> Suggestion | None:
    """Conditional, and frequently absent — which is correct behaviour,
    not a gap. Offering to total a column that may not be monetary would
    reintroduce, at the UI layer, exactly the guessing the numeric-trust
    work removed from the numeric layer."""
    for package in _listed_packages(evidence, listed):
        column = _monetary_column(ctx, package.package_id)
        if column is None:
            continue
        name = _display_name(package)
        return Suggestion(
            kind="group_total",
            label=_label("Total ", column, " by department"),
            question=(
                f"Total the {column} column in {name}, grouped by "
                "department."
            ),
            package_ids=(package.package_id,),
        )
    return None


def _listed_packages(
    evidence: SearchEvidence, listed: set[str]
) -> Iterable[EvidencePackage]:
    """Packages the loop actually opened. These kinds promise *contents*
    — sample rows, a column to total — and only an opened package has
    contents we know about."""
    return (p for p in evidence.packages if p.package_id in listed)


def _monetary_column(ctx: TurnContext, package_id: str) -> str | None:
    """A money-shaped column known to belong to this package.

    `column_metadata` is keyed by bare column name, so membership is
    resolved through the doc→package map rather than assumed: a
    monetary column seen on some *other* package is not a column this
    package has.
    """
    columns: list[str] = []
    for doc_id, pid in ctx.state.doc_package.items():
        if pid == package_id:
            columns.extend(ctx.state.doc_columns.get(doc_id, []))
    for column in columns:
        meta = ctx.state.column_metadata.get(column)
        if meta is None:
            continue
        if is_monetary_column(
            column_name=column,
            semantic_type=meta.get("semantic_type"),
            description=meta.get("description"),
        ):
            return column
    return None


# ── rendering ──


def _display_name(package: EvidencePackage) -> str:
    return package.title or package.package_id


def _label(prefix: str, variable: str, suffix: str = "") -> str:
    """Compose a label within the button budget, truncating only the
    variable part — a label whose fixed words were cut would read as
    broken rather than abbreviated."""
    budget = MAX_LABEL_CHARS - len(prefix) - len(suffix)
    text = " ".join(variable.split())
    if len(text) > budget:
        text = text[: max(1, budget - 1)].rstrip() + "…"
    return f"{prefix}{text}{suffix}"
