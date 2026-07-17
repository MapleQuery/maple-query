"""Template composition for caveats, clarify questions, and retry
hints. Pure functions, no model involved — the answer text the user
sees is the answer text the research model wrote."""
from __future__ import annotations

from semantic_enrich.core.agent.verify import (
    compose_caveat,
    compose_clarify,
    compose_retry_hint,
)


def test_caveat_prepends_and_preserves_answer_verbatim() -> None:
    answer = "Total was **$4.2M** ([DS](/datasets/p1)).\n\n| a | b |"
    out = compose_caveat(gap="per-province figures since 2020", answer=answer)
    assert out == (
        "**Partial answer:** this does not cover per-province figures "
        "since 2020.\n\n" + answer
    )


def test_caveat_normalizes_trailing_period_and_whitespace() -> None:
    out = compose_caveat(gap="  the 2021 data.  ", answer="x")
    assert out.startswith(
        "**Partial answer:** this does not cover the 2021 data.\n\n"
    )
    assert "..\n" not in out


def test_clarify_question_embeds_gap() -> None:
    out = compose_clarify(gap="which federal program you mean")
    assert "which federal program you mean?" in out
    assert out.startswith("I couldn't confidently find data")


def test_retry_hint_with_and_without_lookup_hint() -> None:
    both = compose_retry_hint(
        gap="per-province breakdown", retry_hint="provincial columns"
    )
    assert both == (
        "Your previous answer missed: per-province breakdown. "
        "Look for: provincial columns."
    )
    gap_only = compose_retry_hint(gap="a time series", retry_hint=None)
    assert gap_only == "Your previous answer missed: a time series."
