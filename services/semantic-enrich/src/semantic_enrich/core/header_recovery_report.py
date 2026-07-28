"""Measuring the header detector against the corpus it has to survive.

The gate is **precision, not recall**. A detector that recovers 30% of
documents perfectly is shippable. One that recovers 90% with a handful of
wrong names is not, because a wrong name is one the model will use
confidently in SQL and nobody will see it in the answer.

So this module produces an artefact a person reads, not a verdict a
machine issues. `decline_reasons` is the field that pays for the next
iteration — "the preamble was deeper than the scan window" is a config
change and "contrast failed" is a detector change, and the two need
telling apart before anyone tunes anything. The sample ships with the
rows either side of each recovered header precisely so review is a
reading task rather than a warehouse-querying task.

No model calls. One bounded, cluster-pruned read per document.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from semantic_enrich.core.header_recovery import explain_header

# The go/no-go, as data rather than as a sentence in someone's memory.
# Softening any of it shows up in a diff.
GATE: dict[str, Any] = {
    # Not a percentage. One wrong name in fifty implies ~2% of a
    # 5k-document corpus carries a confidently wrong column name, and
    # nothing downstream would catch it.
    "wrong_names_max": 0,
    # Below this the plumbing is not earning its keep, and the decline
    # reasons should be read before shipping rather than after.
    "recovery_rate_min": 0.3,
    # How many recovered documents a human reads.
    "review_sample_size": 50,
}

# Rows either side of the header that ship with each sampled document,
# so a reviewer can see what the detector saw.
_CONTEXT_ROWS = 2

# Config-fixable declines, as opposed to detector-fixable ones. The
# distinction is the point of tallying reasons at all.
_CONFIG_FIXABLE = frozenset({"no_data_rows_in_window"})
# Declines that mean there was no header to find, rather than a header
# the detector failed on. Counting these as failures would misreport the
# ceiling: no amount of tuning recovers a file that never had a header.
_NOTHING_TO_RECOVER = frozenset(
    {"data_starts_at_row_0", "no_generated_names", "no_generated_columns"}
)


@dataclass(frozen=True)
class ScannedDocument:
    """One document's bounded read, ready for the detector."""

    document_id: str
    package_id: str
    title: str | None
    generated_columns: list[str]
    rows: list[dict[str, object]] = field(default_factory=list)
    total_columns: int = 0


def build_report(
    documents: Sequence[ScannedDocument],
    *,
    scan_rows: int,
    min_density: float,
    bytes_scanned: int,
    sampled_from: int | None = None,
    review_sample_size: int = GATE["review_sample_size"],
) -> dict[str, Any]:
    """Run the detector over already-read documents and describe what
    happened. Pure: no I/O, no model, deterministic."""
    recovered: list[dict[str, Any]] = []
    decline_reasons: dict[str, int] = {}
    # package_id -> [generated columns, columns given a name]. Rolled up
    # at column level rather than document level: a package where two of
    # four columns got named is partially recovered, even though not one
    # of its documents was recovered in full.
    by_package: dict[str, list[int]] = {}
    columns_total = 0
    columns_generated = 0
    columns_named = 0

    for doc in documents:
        report = explain_header(
            doc.rows,
            doc.generated_columns,
            scan_rows=scan_rows,
            min_density=min_density,
        )
        columns_total += doc.total_columns
        columns_generated += len(doc.generated_columns)
        named_here = 0
        if report.recovery is not None:
            named_here = len(report.recovery.names)
            columns_named += named_here
            recovered.append(_sampled(doc, report))
        else:
            decline_reasons[report.reason] = (
                decline_reasons.get(report.reason, 0) + 1
            )
        tally = by_package.setdefault(doc.package_id, [0, 0])
        tally[0] += len(doc.generated_columns)
        tally[1] += named_here

    scanned = len(documents)
    declined = scanned - len(recovered)
    packages_full = sum(
        1 for generated, named in by_package.values()
        if generated > 0 and named == generated
    )
    packages_partial = sum(
        1 for generated, named in by_package.values()
        if 0 < named < generated
    )
    unnamed_before = _share(columns_generated, columns_total)
    unnamed_after = _share(columns_generated - columns_named, columns_total)

    return {
        "documents_scanned": scanned,
        "packages_scanned": len(by_package),
        "sampled_from_documents": sampled_from,
        "recovered": len(recovered),
        "declined": declined,
        "recovery_rate": _share(len(recovered), scanned),
        "decline_reasons": dict(sorted(decline_reasons.items())),
        "decline_reason_classes": _classify(decline_reasons),
        "packages_fully_recovered": packages_full,
        "packages_partially_recovered": packages_partial,
        "columns_scanned": columns_total,
        "columns_generated": columns_generated,
        "columns_named": columns_named,
        # Over the SCANNED subset, which is drawn from affected packages
        # only — not the whole corpus. Naming it `_scanned` rather than
        # `_corpus` keeps the denominator honest.
        "unnamed_share_scanned_before": unnamed_before,
        "unnamed_share_scanned_after": unnamed_after,
        "bytes_scanned": bytes_scanned,
        "scan_rows": scan_rows,
        "min_density": min_density,
        "gate": dict(GATE),
        "gate_result": _gate_result(
            recovery_rate=_share(len(recovered), scanned),
            sample_size=min(review_sample_size, len(recovered)),
        ),
        "sample": recovered[:review_sample_size],
    }


def _gate_result(*, recovery_rate: float, sample_size: int) -> dict[str, Any]:
    """The half of the gate a machine can decide, and the half it cannot.

    Whether a recovered name is *wrong* is a reading task — the sample
    exists so a person can do it — so this records the bar and leaves the
    verdict pending rather than inventing a pass.
    """
    return {
        "recovery_rate_pass": recovery_rate >= GATE["recovery_rate_min"],
        "recovery_rate_observed": recovery_rate,
        "review_sample_available": sample_size,
        "wrong_names_observed": None,
        "wrong_names_verdict": "pending_human_review",
        "decision": "no_go_until_reviewed",
        "note": (
            "Precision is the gate; recall is a measurement. Fill "
            "`wrong_names_observed` by reading `sample` — a name is wrong "
            "if it came from a row that is not the header, is attached to "
            "the wrong positional key, or appears nowhere in the document. "
            "Awkward is not wrong: 'Total Amount ($000)' is a correct "
            "recovery."
        ),
    }


def _classify(decline_reasons: Mapping[str, int]) -> dict[str, int]:
    config = sum(
        n for reason, n in decline_reasons.items() if reason in _CONFIG_FIXABLE
    )
    absent = sum(
        n
        for reason, n in decline_reasons.items()
        if reason in _NOTHING_TO_RECOVER
    )
    total = sum(decline_reasons.values())
    return {
        # A wider scan window would fix these.
        "config_fixable": config,
        # There is no header in the stored rows to find. Not a detector
        # miss, and not something tuning can move.
        "no_header_present": absent,
        # A header exists and the detector declined it — multi-tier,
        # repeated labels, sparse. The only bucket worth tuning against.
        "detector_declined": total - config - absent,
    }


def _sampled(doc: ScannedDocument, report: Any) -> dict[str, Any]:
    recovery = report.recovery
    index = recovery.header_row_index
    lo = max(0, index - _CONTEXT_ROWS)
    hi = min(len(doc.rows), index + _CONTEXT_ROWS + 1)
    return {
        "document_id": doc.document_id,
        "package_id": doc.package_id,
        "title": doc.title,
        "header_row_index": index,
        "preamble_rows": recovery.preamble_rows,
        "names": dict(recovery.names),
        "signals": dict(recovery.signals),
        "generated_columns": list(doc.generated_columns),
        # The rows either side, so review is reading rather than querying.
        "context_rows": [
            {"row_index": i, "is_header": i == index, "values": doc.rows[i]}
            for i in range(lo, hi)
        ],
    }


def _share(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part / whole, 4)
