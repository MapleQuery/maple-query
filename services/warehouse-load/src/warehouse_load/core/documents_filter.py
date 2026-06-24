"""Format/status filter, dedupe, then bucket-existence intersection.

Filter runs *before* dedupe so a quarantined-csv row never shadows a
real success row at the same `source_url`.

Dedupe key is `source_url` (not `document_id`) — CKAN URL-sharing
and failed-row placeholder `document_id`s produce within-run
`source_url` dupes by design.

Tie-break on equal `ingested_at`: `document_id` ASC. The CKAN
URL-sharing case can land N rows in one pass with sub-millisecond
deltas; pinning the secondary key makes the dedupe winner
deterministic regardless of file/line iteration order.

`intersect_bucket` runs *after* dedupe and drops rows whose
`gcs_uri` is not present in the bucket-truth set, so a bucket
clean is self-healing on the next load. Running before dedupe
would waste existence checks on rows that lose dedupe; running
after MERGE would defeat the purpose.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from warehouse_load.types import RawRunlogRow

FilterReason = Literal["not_csv", "not_success", "blob_missing"]


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

    In-memory dict over the full input. Fine at current corpus scale
    (~15K rows); revisit if it grows past tens of millions.
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


def intersect_bucket(
    rows: Iterable[RawRunlogRow],
    existing_uris: frozenset[str],
) -> tuple[list[RawRunlogRow], list[FilteredRow]]:
    """Return `(kept, dropped)` after intersecting `gcs_uri` against the bucket.

    Rows with a `gcs_uri` of None are dropped as `blob_missing` — a
    success-csv row in the runlog should always carry a `gcs_uri`,
    so a None value here is itself a sign the blob is gone.

    `existing_uris` is the materialized set returned by
    `GcsClient.list_existing`; passing a frozenset signals to callers
    that this function does not mutate it.
    """
    kept: list[RawRunlogRow] = []
    dropped: list[FilteredRow] = []
    for row in rows:
        if row.gcs_uri is None or row.gcs_uri not in existing_uris:
            dropped.append(FilteredRow(row=row, reason="blob_missing"))
            continue
        kept.append(row)
    return kept, dropped


def _wins(candidate: RawRunlogRow, incumbent: RawRunlogRow) -> bool:
    """Does `candidate` beat `incumbent` for the dedupe slot?"""
    if candidate.ingested_at != incumbent.ingested_at:
        return candidate.ingested_at > incumbent.ingested_at
    # Tie on timestamp: smaller document_id wins (deterministic, easy
    # to reason about in tests). The actual choice between candidate
    # and incumbent is arbitrary as long as it's stable across runs.
    return candidate.document_id < incumbent.document_id
