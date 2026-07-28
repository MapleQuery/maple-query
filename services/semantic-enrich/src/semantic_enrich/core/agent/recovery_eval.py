"""Guided-recovery regression fixture.

The milestone's promises are *shape* promises — a surrender names its
evidence, offers a way forward, and an accepted offer is neither
replaced by a question nor stripped of its numeric guarantees. Shape is
exactly what scripted fakes assert well, so the weight of this fixture
sits in a tier that costs nothing and runs on every change. A
regression lock priced at a dollar a run gets skipped; one that is free
does not.

Half the cases are **negative**. Not offering — below the retrieval
floor, on a clean answer, at the chain cap — is as much of the design
as offering, and it is the half that rots silently. A chip graveyard is
the most likely way this milestone fails in production, and it is
invisible unless something tests for its absence.

This module is pure: cases and a grader. Driving them belongs to the
tests, which own the fakes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

# The opening of `verify.compose_clarify`. A fixture turn that expects a
# substantive answer must never ship text starting with this — it is the
# headline failure the milestone exists to close, and it is checked
# globally rather than per case so it cannot be reintroduced by someone
# adding a case without having read the parent doc.
CLARIFY_MARKER = "I couldn't confidently find data for this as asked"

# The evidence footer's header, from `evidence.compose_footer`.
FOOTER_MARKER = "**What I searched:**"


@dataclass(frozen=True)
class RecoveryCase:
    id: str
    question: str
    expect_outcome: str
    expect_footer: bool
    # Inclusive (min, max). A range rather than an exact count because
    # kind eligibility legitimately varies with what the turn listed.
    expect_suggestions: tuple[int, int]
    expect_derivation: bool
    scope_package_ids: tuple[str, ...] = ()
    # Synthetic `turn_records` outcomes preceding this turn, for the
    # chain cap.
    prior_outcomes: tuple[str, ...] = ()
    forbid_clarify_replacement: bool = True
    live_only: bool = False
    notes: str = ""


@dataclass(frozen=True)
class Observation:
    """What one run of a case actually produced."""

    outcome: str
    message: str
    suggestion_count: int
    derivation_count: int
    event_types: tuple[str, ...]


@dataclass(frozen=True)
class GradeResult:
    case_id: str
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)


CASES: tuple[RecoveryCase, ...] = (
    RecoveryCase(
        id="air-travel-2025",
        question="how much did the government spend on air travel in 2025?",
        expect_outcome="no_data",
        expect_footer=True,
        expect_suggestions=(1, 3),
        expect_derivation=False,
        notes=(
            "The canonical failure the milestone was written against: "
            "the loop held the datasets and referred the user out of "
            "the product instead of naming them."
        ),
    ),
    RecoveryCase(
        id="air-travel-summarize",
        question=(
            "Summarize what data is in Supplementary Estimates (B) "
            "2025-26 — what it covers, its time range, and its main "
            "columns."
        ),
        scope_package_ids=("ec676496-a50d-4afa-9a2f-4d97748e79e5",),
        expect_outcome="explored",
        expect_footer=False,
        expect_suggestions=(0, 3),
        expect_derivation=False,
        notes=(
            "An accepted offer. The turn runs no SQL, so without the "
            "scoped-turn contract verify would replace this summary "
            "with a clarifying question."
        ),
    ),
    RecoveryCase(
        id="air-travel-total",
        question=(
            "Total the Authorities_To_Date column in Supplementary "
            "Estimates (B) 2025-26, grouped by Organization."
        ),
        # Verified against the warehouse rather than remembered: this
        # package has 11 columns, *zero* generated headers, and
        # `Authorities_To_Date` carries semantic_type currency_cad. The
        # first draft scoped this to the housing package — wrong title,
        # and the unnamed-columns one, so no total was ever possible.
        scope_package_ids=("ec676496-a50d-4afa-9a2f-4d97748e79e5",),
        expect_outcome="answered",
        expect_footer=False,
        expect_suggestions=(0, 0),
        expect_derivation=True,
        notes=(
            "The M6 non-regression case. A scoped *numeric* follow-up "
            "must keep its derivation: if guided recovery ever becomes "
            "a route to an untraced number, this is what catches it. "
            "Note the live tier does not currently reach the numeric "
            "path on this case — the column is real at *package* level "
            "(semantic.columns) but absent from the documents "
            "list_documents returns, so the scoped turn declines. The "
            "unscoped `clean-total` case answers the same question "
            "against the same package and does carry its derivation, "
            "which is what actually proves the invariant live. See the "
            "header-recovery milestone: package-level and "
            "document-level column sets diverge."
        ),
    ),
    RecoveryCase(
        id="below-floor",
        question=(
            "how much did federal departments spend on office plants "
            "in 2019-20?"
        ),
        expect_outcome="clarified",
        expect_footer=False,
        expect_suggestions=(0, 0),
        expect_derivation=False,
        forbid_clarify_replacement=False,
        notes=(
            "The false-invitation guard. Offering exploration over "
            "datasets that scored below the similarity floor is the "
            "failure most likely to teach users that every chip is "
            "noise, and it is invisible unless tested directly. "
            "Deliberately federal, specific and monetary in shape so "
            "triage routes it in_scope — an earlier draft asked for the "
            "'vibe' of fiscal policy, which deflects as opinion and "
            "tests the wrong gate entirely. Live, this question still "
            "retrieves above the floor and answers with a caveat — a "
            "genuinely below-floor question is hard to pin against a "
            "3.6k-package corpus, so the deterministic tier (which "
            "scripts weak retrieval directly) is what actually holds "
            "this guard."
        ),
    ),
    RecoveryCase(
        id="clean-total",
        question=(
            "What are the total proposed authorities to date in "
            "Supplementary Estimates (B) 2025-26?"
        ),
        expect_outcome="answered",
        expect_footer=False,
        expect_suggestions=(0, 0),
        expect_derivation=True,
        notes=(
            "Guided recovery stays out of the way of a working answer: "
            "no footer, no chips."
        ),
    ),
    RecoveryCase(
        id="chain-cap",
        question="how much did the government spend on air travel in 2025?",
        prior_outcomes=("explored", "explored", "explored"),
        expect_outcome="no_data",
        expect_footer=True,
        expect_suggestions=(0, 0),
        expect_derivation=False,
        notes=(
            "The treadmill guard. Guided recovery should converge on an "
            "answer, not become a browsing mode — the footer still "
            "informs, the offers stop."
        ),
    ),
    RecoveryCase(
        id="explore-typed",
        question="what's in the supplementary estimates?",
        expect_outcome="explored",
        expect_footer=False,
        expect_suggestions=(0, 3),
        expect_derivation=False,
        live_only=True,
        notes="Needs a real triage call: the `explore` classification.",
    ),
    RecoveryCase(
        id="meta-boundary",
        question="what data do you have?",
        expect_outcome="deflected",
        expect_footer=False,
        expect_suggestions=(0, 0),
        expect_derivation=False,
        live_only=True,
        notes=(
            "Pins the meta/explore line: corpus-wide is meta, a named "
            "dataset is explore. Needs a real classification."
        ),
    ),
)


def deterministic_cases() -> tuple[RecoveryCase, ...]:
    """Cases assertable on scripted fakes — everything whose outcome
    does not depend on a real classification."""
    return tuple(c for c in CASES if not c.live_only)


def case_by_id(case_id: str) -> RecoveryCase:
    for case in CASES:
        if case.id == case_id:
            return case
    raise KeyError(case_id)


def grade(case: RecoveryCase, observed: Observation) -> GradeResult:
    """Compare one run against its case. Returns every failure rather
    than the first, so a broken run reports its whole shape at once."""
    failures: list[str] = []

    if observed.outcome != case.expect_outcome:
        failures.append(
            f"outcome: expected {case.expect_outcome!r}, "
            f"got {observed.outcome!r}"
        )

    has_footer = FOOTER_MARKER in observed.message
    if has_footer != case.expect_footer:
        failures.append(
            f"footer: expected {'present' if case.expect_footer else 'absent'}"
        )

    low, high = case.expect_suggestions
    if not low <= observed.suggestion_count <= high:
        failures.append(
            f"suggestions: expected {low}-{high}, "
            f"got {observed.suggestion_count}"
        )

    has_derivation = observed.derivation_count > 0
    if has_derivation != case.expect_derivation:
        failures.append(
            "derivation: expected "
            f"{'present' if case.expect_derivation else 'absent'}"
        )

    if case.forbid_clarify_replacement and observed.message.startswith(
        CLARIFY_MARKER
    ):
        failures.append(
            "clarify replacement: a substantive answer was replaced by "
            "a clarifying question"
        )

    failures.extend(_ordering_failures(observed.event_types))

    return GradeResult(
        case_id=case.id, passed=not failures, failures=tuple(failures)
    )


def _ordering_failures(event_types: tuple[str, ...]) -> list[str]:
    """A consumer reading the stream in order must get the answer, then
    its trace, then its offers, then the record."""
    order = ["message_delta", "derivation", "suggestions", "turn_record"]
    positions = [
        (name, event_types.index(name))
        for name in order
        if name in event_types
    ]
    failures: list[str] = []
    for (a, i), (b, j) in pairwise(positions):
        if i > j:
            failures.append(f"ordering: {a} came after {b}")
    return failures
