"""The recovery report's shape, over documents that really exist.

The report is the artefact, not a summary of one: whoever revisits this
in three months needs the decline reasons and the sample rows, not a
recovery rate and a verdict. So these tests check that the things a
future reader would need are actually in the file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_enrich.core.header_recovery import (
    HEADER_MIN_DENSITY,
    HEADER_SCAN_ROWS,
)
from semantic_enrich.core.header_recovery_report import (
    GATE,
    ScannedDocument,
    build_report,
)

_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "header_recovery_documents.json"
)
_DOCUMENTS: list[dict[str, Any]] = json.loads(
    _FIXTURE.read_text(encoding="utf-8")
)


def _scanned(prefix: str, *, package: str = "pkg-1") -> ScannedDocument:
    doc = next(d for d in _DOCUMENTS if d["document_id"].startswith(prefix))
    keys: list[str] = []
    for row in doc["rows"]:
        for key in row:
            if key not in keys:
                keys.append(key)
    return ScannedDocument(
        document_id=doc["document_id"],
        package_id=package,
        title=doc["title"],
        generated_columns=doc["generated_columns"],
        rows=doc["rows"],
        total_columns=len(keys),
    )


def _report(documents: list[ScannedDocument], **kwargs: Any) -> dict[str, Any]:
    return build_report(
        documents,
        scan_rows=HEADER_SCAN_ROWS,
        min_density=HEADER_MIN_DENSITY,
        bytes_scanned=kwargs.pop("bytes_scanned", 1024),
        **kwargs,
    )


# One that recovers, one that declines on a real three-tier header.
RECOVERED = "43d968b125"
DECLINED = "57f2ef5417"


def test_counts_and_rate_are_consistent() -> None:
    report = _report([_scanned(RECOVERED), _scanned(DECLINED)])
    assert report["documents_scanned"] == 2
    assert report["recovered"] == 1
    assert report["declined"] == 1
    assert report["recovery_rate"] == 0.5


def test_decline_reasons_are_tallied_per_gate() -> None:
    report = _report([_scanned(DECLINED), _scanned("235723ded4")])
    assert report["decline_reasons"] == {
        "data_starts_at_row_0": 1,
        "density": 1,
    }


def test_decline_reasons_separate_config_from_detector_from_absent() -> None:
    """The distinction that pays for the next iteration. A wider scan
    window fixes one bucket; a detector change fixes another; nothing
    fixes the third, because those files have no header to find."""
    report = _report(
        [
            _scanned("deba8847a0"),  # all-text sheet: config-fixable
            _scanned("235723ded4"),  # data from row 0: no header exists
            _scanned(DECLINED),  # three-tier header: detector declined
        ]
    )
    classes = report["decline_reason_classes"]
    assert classes["config_fixable"] == 1
    assert classes["no_header_present"] == 1
    assert classes["detector_declined"] == 1
    assert sum(classes.values()) == report["declined"]


def test_package_rollup_distinguishes_full_from_partial() -> None:
    report = _report(
        [
            _scanned(RECOVERED, package="pkg-full"),
            _scanned("fdf3e12873", package="pkg-partial"),
            _scanned(DECLINED, package="pkg-partial"),
        ]
    )
    # Housing table 1 names both of its generated columns.
    assert report["packages_fully_recovered"] == 1
    # CSIS names two of four, and the legal-aid doc names none.
    assert report["packages_partially_recovered"] == 1


def test_unnamed_share_is_reported_before_and_after() -> None:
    report = _report([_scanned(RECOVERED)])
    assert report["unnamed_share_scanned_before"] > 0
    assert (
        report["unnamed_share_scanned_after"]
        < report["unnamed_share_scanned_before"]
    )


def test_the_denominator_is_named_for_what_it_is() -> None:
    """The scan draws from affected packages only, so the share is over
    the scanned subset — not the corpus. The field name has to say so,
    because a `corpus_` prefix on a sample is how a measurement becomes
    a false claim."""
    report = _report([_scanned(RECOVERED)])
    assert "unnamed_share_scanned_after" in report
    assert not any(key.startswith("corpus_") for key in report)
    assert report["sampled_from_documents"] is None


def test_the_sample_ships_the_rows_either_side_of_the_header() -> None:
    """Review is meant to be a reading task, not a warehouse-querying
    task, which only works if the context travels with the report."""
    report = _report([_scanned(RECOVERED)])
    entry = report["sample"][0]
    assert entry["header_row_index"] == 1
    indexes = [r["row_index"] for r in entry["context_rows"]]
    assert indexes == [0, 1, 2, 3]
    header = [r for r in entry["context_rows"] if r["is_header"]]
    assert len(header) == 1
    assert header[0]["row_index"] == 1
    # The values are the real cells, so a reviewer can check the name
    # against the row it came from without another query.
    assert "Total Amount ($000)" in json.dumps(
        header[0]["values"], ensure_ascii=False
    )


def test_the_sample_carries_the_names_and_the_signals() -> None:
    entry = _report([_scanned(RECOVERED)])["sample"][0]
    assert entry["names"]["__col_2"] == "Total Amount ($000)"
    assert set(entry["signals"]) == {
        "positional",
        "all_text",
        "density",
        "distinctness",
        "contrast",
    }


def test_the_sample_is_capped() -> None:
    report = _report([_scanned(RECOVERED) for _ in range(80)],)
    assert report["recovered"] == 80
    assert len(report["sample"]) == GATE["review_sample_size"]


def test_bytes_scanned_and_settings_are_recorded() -> None:
    report = _report([_scanned(RECOVERED)], bytes_scanned=123_456)
    assert report["bytes_scanned"] == 123_456
    assert report["scan_rows"] == HEADER_SCAN_ROWS
    assert report["min_density"] == HEADER_MIN_DENSITY


def test_sampling_is_disclosed_when_it_happened() -> None:
    report = _report([_scanned(RECOVERED)], sampled_from=5307)
    assert report["sampled_from_documents"] == 5307
    assert report["documents_scanned"] == 1


def test_an_empty_scan_does_not_divide_by_zero() -> None:
    report = _report([])
    assert report["documents_scanned"] == 0
    assert report["recovery_rate"] == 0.0
    assert report["sample"] == []
