"""The explore rubric's dispositions are restricted structurally.

This is the safety property that makes the rubric strictly better than
the bypass it replaces. The fit checker guards `clarify` with a runtime
demotion — and that demotion failing to fire is precisely the bug this
milestone exists to close. An absent enum value depends on nothing
firing correctly, so even a badly miscalibrated explore checker cannot
replace a summary with a question or spend a second research leg.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent.verify import (
    _EXPLORE_ACTIONS,
    EXPLORE_CHECK_SCHEMA,
    _validate,
    compose_explore_caveat,
)


def _response(action: str, **overrides: Any) -> dict[str, Any]:
    return {
        "fits": False,
        "confidence": 0.9,
        "gap": "the 2019 data",
        "action": action,
        **overrides,
    }


def test_the_enum_contains_exactly_answer_and_caveat() -> None:
    assert {"answer", "caveat"} == _EXPLORE_ACTIONS
    assert EXPLORE_CHECK_SCHEMA["properties"]["action"]["enum"] == [
        "answer",
        "caveat",
    ]


def test_clarify_and_retry_are_absent_not_demoted() -> None:
    for forbidden in ("clarify", "retry"):
        assert forbidden not in _EXPLORE_ACTIONS
        # A response naming one fails validation outright, which the
        # caller turns into a fail-open `answer`.
        assert (
            _validate(_response(forbidden), actions=_EXPLORE_ACTIONS)
            is None
        )


def test_permitted_actions_validate() -> None:
    for allowed in ("answer", "caveat"):
        check = _validate(_response(allowed), actions=_EXPLORE_ACTIONS)
        assert check is not None
        assert check.action == allowed


def test_the_fit_checker_still_accepts_its_own_four() -> None:
    """The shared validator must not have narrowed the numeric path."""
    for allowed in ("answer", "caveat", "retry", "clarify"):
        assert _validate(_response(allowed, retry_hint=None)) is not None


def test_explore_caveat_preserves_the_answer_verbatim() -> None:
    answer = "This dataset has 312 columns.\n\n| a | b |"
    out = compose_explore_caveat(gap="the 2019 fiscal year.", answer=answer)
    assert out == (
        "**Note:** this description does not cover the 2019 fiscal "
        "year.\n\n" + answer
    )
