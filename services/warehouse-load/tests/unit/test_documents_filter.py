"""§12.1 unit tests for the filter + dedupe truth table."""
from __future__ import annotations

from datetime import UTC, datetime

from tests.conftest import make_row
from warehouse_load.core.documents_filter import (
    DedupedRow,
    FilteredRow,
    dedupe_by_source_url,
    filter_rows,
)
from warehouse_load.types import RawRunlogRow


def _filtered(
    rows: list[RawRunlogRow],
) -> tuple[list[RawRunlogRow], list[FilteredRow]]:
    kept: list[RawRunlogRow] = []
    dropped: list[FilteredRow] = []
    for r in filter_rows(rows):
        if isinstance(r, FilteredRow):
            dropped.append(r)
        else:
            kept.append(r)
    return kept, dropped


def test_success_csv_is_kept() -> None:
    row = make_row(file_format="csv", ingestion_status="success")
    kept, dropped = _filtered([row])
    assert kept == [row]
    assert dropped == []


def test_quarantined_csv_is_filtered_as_not_success() -> None:
    row = make_row(file_format="csv", ingestion_status="quarantined")
    kept, dropped = _filtered([row])
    assert kept == []
    assert len(dropped) == 1
    assert dropped[0].reason == "not_success"


def test_failed_csv_is_filtered_as_not_success() -> None:
    row = make_row(file_format="csv", ingestion_status="failed")
    kept, dropped = _filtered([row])
    assert kept == []
    assert dropped[0].reason == "not_success"


def test_success_xlsx_is_filtered_as_not_csv() -> None:
    row = make_row(file_format="xlsx", ingestion_status="success")
    kept, dropped = _filtered([row])
    assert kept == []
    assert dropped[0].reason == "not_csv"


def test_filter_runs_before_dedupe_keeps_success_over_quarantined_at_same_url() -> None:
    """The `bq_loader_format_filter` memory's invariant.

    Two rows at the same source_url: a quarantined-csv with a newer
    ingested_at and a success-csv with an older one. After §6.1
    drops the quarantined row, §6.2 sees only the success row, and
    that's what reaches the loader. If dedupe had run first, the
    quarantined row's newer timestamp would have won — bug.
    """
    url = "https://example.org/data.csv"
    older_success = make_row(
        source_url=url,
        document_id="1" + "a" * 63,
        ingestion_status="success",
        ingested_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    newer_quarantined = make_row(
        source_url=url,
        document_id="2" + "a" * 63,
        ingestion_status="quarantined",
        ingested_at=datetime(2026, 6, 2, tzinfo=UTC),
    )

    kept_after_filter, _ = _filtered([newer_quarantined, older_success])
    deduped, dropped = dedupe_by_source_url(kept_after_filter)

    assert len(deduped) == 1
    assert deduped[0].document_id == older_success.document_id
    assert dropped == []


def test_dedupe_newest_ingested_at_wins() -> None:
    url = "https://example.org/data.csv"
    older = make_row(
        source_url=url,
        document_id="1" + "a" * 63,
        ingested_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    newer = make_row(
        source_url=url,
        document_id="2" + "a" * 63,
        ingested_at=datetime(2026, 6, 2, tzinfo=UTC),
    )
    deduped, dropped = dedupe_by_source_url([older, newer])
    assert deduped == [newer]
    assert len(dropped) == 1
    assert isinstance(dropped[0], DedupedRow)
    assert dropped[0].kept == newer


def test_dedupe_tiebreak_on_document_id_is_deterministic() -> None:
    """Same ingested_at; smaller document_id wins regardless of input order."""
    url = "https://example.org/data.csv"
    ts = datetime(2026, 6, 1, tzinfo=UTC)
    a = make_row(source_url=url, document_id="a" * 64, ingested_at=ts)
    b = make_row(source_url=url, document_id="b" * 64, ingested_at=ts)

    forward, _ = dedupe_by_source_url([a, b])
    reverse, _ = dedupe_by_source_url([b, a])

    assert forward[0].document_id == "a" * 64
    assert reverse[0].document_id == "a" * 64


def test_dedupe_yields_distinct_rows_for_distinct_urls() -> None:
    a = make_row(source_url="https://example.org/a.csv", document_id="a" * 64)
    b = make_row(source_url="https://example.org/b.csv", document_id="b" * 64)
    deduped, dropped = dedupe_by_source_url([a, b])
    assert {r.source_url for r in deduped} == {a.source_url, b.source_url}
    assert dropped == []
