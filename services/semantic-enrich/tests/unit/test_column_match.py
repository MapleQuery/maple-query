"""Matching a guessed column name onto a real header.

The fixtures are the actual headers of the five documents in the
`2025-26 Estimates` package, because the failure this module exists for
was found there and the spellings are the point.
"""
from __future__ import annotations

from semantic_enrich.core.column_match import (
    is_exact,
    match_column,
    match_columns,
    normalize_column,
)

ORG_SUMMARY = [
    "Organization",
    "Vote",
    "Description",
    "2023-24 Expenditures",
    "2024-25 Main Estimates",
    "2024-25 Estimates To Date",
    "2025-26 Main Estimates",
]
# Note the en dashes: the same fiscal year, spelled differently from
# `organization-summary`, in the same package.
STATUTORY = [
    "Department, Agency or Crown corporation",
    "2023–24 Expenditures",
    "2024–25 Estimates To Date",
    "2025–26 Main Estimates",
]
TRANSFER = ["Organization", "Transfer Payment", "Amount"]


def test_exact_match_wins_and_excludes_looser_hits() -> None:
    """`Organization` must not also drag in every column containing it."""
    assert match_columns("Organization", ORG_SUMMARY) == ["Organization"]
    assert is_exact("Organization", ORG_SUMMARY)


def test_token_containment_finds_the_real_header() -> None:
    assert match_columns("Expenditures", ORG_SUMMARY) == [
        "2023-24 Expenditures"
    ]
    assert match_columns("Department", STATUTORY) == [
        "Department, Agency or Crown corporation"
    ]


def test_a_loose_hit_is_not_an_exact_one() -> None:
    """The distinction `list_documents` narrows on."""
    assert match_columns("Expenditures", ORG_SUMMARY)
    assert not is_exact("Expenditures", ORG_SUMMARY)


def test_dash_variants_fold_together() -> None:
    assert normalize_column("2023–24 Expenditures") == normalize_column(
        "2023-24 Expenditures"
    )
    assert match_columns("2023-24 Expenditures", STATUTORY) == [
        "2023–24 Expenditures"
    ]


def test_no_match_returns_empty() -> None:
    assert match_columns("Airplane", ORG_SUMMARY) == []
    assert match_columns("Department", ORG_SUMMARY) == []
    assert match_columns("Department", TRANSFER) == []


def test_ambiguity_is_returned_whole_not_ranked() -> None:
    """Two fiscal years both matching `Estimates` is a real ambiguity;
    inventing a preference between them is the guess this declines."""
    hits = match_columns("Estimates", ORG_SUMMARY)
    assert set(hits) == {
        "2024-25 Main Estimates",
        "2024-25 Estimates To Date",
        "2025-26 Main Estimates",
    }
    assert match_column("Estimates", ORG_SUMMARY) is None
    assert match_column("Expenditures", ORG_SUMMARY) == "2023-24 Expenditures"


def test_case_and_punctuation_are_ignored() -> None:
    assert match_columns("organization", ORG_SUMMARY) == ["Organization"]
    assert match_columns("transfer payment", TRANSFER) == ["Transfer Payment"]


def test_empty_and_punctuation_only_requests_match_nothing() -> None:
    assert match_columns("", ORG_SUMMARY) == []
    assert match_columns("   ", ORG_SUMMARY) == []
    assert match_columns("---", ORG_SUMMARY) == []


def test_a_multi_word_request_needs_all_its_words() -> None:
    assert match_columns("Main Estimates", ORG_SUMMARY) == [
        "2024-25 Main Estimates",
        "2025-26 Main Estimates",
    ]
    assert match_columns("Capital Estimates", ORG_SUMMARY) == []
