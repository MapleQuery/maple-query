"""The evidence footer template. Pure, deterministic, no model.

Same posture as the caveat composer: the footer is composed, never
generated, so it is diffable and unit-testable and costs nothing.
"""
from __future__ import annotations

from semantic_enrich.core.agent.evidence import (
    MAX_TITLE_CHARS,
    EvidencePackage,
    SearchEvidence,
    compose_footer,
)


def _evidence(
    *packages: EvidencePackage,
    queries: tuple[str, ...] = (),
    truncated: int = 0,
) -> SearchEvidence:
    return SearchEvidence(
        packages=packages, queries_tried=queries, truncated=truncated
    )


def test_full_render() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(
                package_id="p1",
                title="Supplementary Estimates B, 2025-26",
                column_count=312,
            ),
            EvidencePackage(
                package_id="p2",
                title="Public Accounts of Canada, Volume II",
                column_count=89,
            ),
            queries=("air travel expenditures", "travel costs 2025-26"),
            truncated=2,
        )
    )
    assert out == (
        "\n\n**What I searched:** *Supplementary Estimates B, 2025-26* "
        "(312 columns) · *Public Accounts of Canada, Volume II* "
        "(89 columns) · +2 more"
        "\n\n**Search terms tried:** \"air travel expenditures\", "
        "\"travel costs 2025-26\""
    )


def test_leading_blank_line_keeps_the_footer_off_the_last_paragraph() -> None:
    out = compose_footer(
        _evidence(EvidencePackage(package_id="p1", title="D", column_count=1))
    )
    assert out.startswith("\n\n")
    assert ("the model's closing sentence." + out).splitlines()[1] == ""


def test_unknown_column_count_renders_no_parenthetical() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(
                package_id="p1", title="Ranked Only", column_count=None
            )
        )
    )
    assert out.strip() == "**What I searched:** *Ranked Only*"


def test_untitled_package_falls_back_to_its_id() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(package_id="pkg-abc", title=None, column_count=7)
        )
    )
    assert "`pkg-abc` (7 columns)" in out


def test_single_column_is_singular() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(package_id="p1", title="Tiny", column_count=1)
        )
    )
    assert "(1 column)" in out


def test_no_searches_recorded_omits_the_second_line() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(package_id="p1", title="D", column_count=3)
        )
    )
    assert "Search terms tried" not in out
    assert out.count("\n\n") == 1


def test_zero_packages_renders_nothing_at_all() -> None:
    # An empty header is worse than no footer: this is the below-floor
    # case that should be clarifying anyway.
    assert compose_footer(_evidence(queries=("anything",))) == ""


def test_long_titles_are_truncated() -> None:
    title = "Supplementary Estimates " + "B" * 100
    out = compose_footer(
        _evidence(
            EvidencePackage(package_id="p1", title=title, column_count=None)
        )
    )
    rendered = out.rsplit("*", 2)[1]
    assert len(rendered) == MAX_TITLE_CHARS
    assert rendered.endswith("…")


def test_truncated_count_appends_a_more_marker() -> None:
    out = compose_footer(
        _evidence(
            EvidencePackage(package_id="p1", title="D", column_count=None),
            truncated=5,
        )
    )
    assert out.rstrip().endswith("· +5 more")
