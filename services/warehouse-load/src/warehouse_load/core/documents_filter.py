"""Stage 1 + 2 of §6: format/status filter, then source_url dedupe.

Filter runs *before* dedupe so a quarantined-csv row never shadows a
real success row at the same `source_url`. See the
`bq_loader_format_filter` memory.

Dedupe key is `source_url` (not `document_id`) — CKAN URL-sharing
and failed-row placeholder `document_id`s produce within-run
`source_url` dupes by design. See the `runlog_failed_rows` memory.

Tie-break on equal `ingested_at`: `document_id` ASC. The CKAN
URL-sharing case can land N rows in one pass with sub-millisecond
deltas; pinning the secondary key makes the dedupe winner
deterministic regardless of file/line iteration order.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from warehouse_load.types import RawRunlogRow

FilterReason = Literal["not_csv", "not_success"]


@dataclass(frozen=True)
class FilteredRow:
    row: RawRunlogRow
    reason: FilterReason


@dataclass(frozen=True)
class DedupedRow:
    dropped: RawRunlogRow
    kept: RawRunlogRow
    reason: Literal["older_ingested_at"]


def filter_rows(
    rows: Iterable[RawRunlogRow],
) -> Iterator[RawRunlogRow | FilteredRow]:
    """Yield kept rows verbatim and dropped rows wrapped in FilteredRow.

    Caller is expected to type-discriminate via `isinstance` and route
    accordingly (count + log filtered rows; pass kept rows on to
    `dedupe_by_source_url`). Single-pass iterator so it composes with
    streaming reads from disk/GCS.
    """
    for row in rows:
        if row.file_format != "csv":
            yield FilteredRow(row=row, reason="not_csv")
            continue
        if row.ingestion_status != "success":
            yield FilteredRow(row=row, reason="not_success")
            continue
        yield row


def dedupe_by_source_url(
    rows: Iterable[RawRunlogRow],
) -> tuple[list[RawRunlogRow], list[DedupedRow]]:
    """Return `(kept, dropped)`.

    Latest `ingested_at` wins; `document_id` ASC breaks ties so the
    winner is deterministic when CKAN URL-sharing puts N rows in one
    pass at the same timestamp.

    In-memory dict over the full input. Fine at the M2 scale
    (~15K rows for the whole corpus, per §7.1); revisit if the corpus
    grows past tens of millions.
    """
    latest: dict[str, RawRunlogRow] = {}
    dropped: list[DedupedRow] = []

    for row in rows:
        prev = latest.get(row.source_url)
        if prev is None:
            latest[row.source_url] = row
            continue

        if _wins(row, prev):
            dropped.append(DedupedRow(dropped=prev, kept=row, reason="older_ingested_at"))
            latest[row.source_url] = row
        else:
            dropped.append(DedupedRow(dropped=row, kept=prev, reason="older_ingested_at"))

    return list(latest.values()), dropped


def _wins(candidate: RawRunlogRow, incumbent: RawRunlogRow) -> bool:
    """Does `candidate` beat `incumbent` for the dedupe slot?"""
    if candidate.ingested_at != incumbent.ingested_at:
        return candidate.ingested_at > incumbent.ingested_at
    # Tie on timestamp: smaller document_id wins (deterministic, easy
    # to reason about in tests). The actual choice between candidate
    # and incumbent is arbitrary as long as it's stable across runs.
    return candidate.document_id < incumbent.document_id
