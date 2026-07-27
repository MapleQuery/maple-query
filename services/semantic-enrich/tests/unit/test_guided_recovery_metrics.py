"""The guided-recovery baseline arithmetic.

`exploitable_rate` decides whether the interaction surfaces built on
top of the evidence footer are worth building at full scope, so the
denominator has to be right: it is *surrenders*, not turns, and a turn
counts only when it has both a listed package and a search that
cleared the retrieval floor. One is useless without the other — a
dataset nobody scored as relevant is not a next step, and a good
retrieval score with nothing opened has nothing to point at.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent_eval import _guided_recovery


def _turn(
    qid: str,
    *,
    outcome: str,
    packages: list[str] | None = None,
    qualities: list[str] | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "id": qid,
        "run": {
            "outcome": outcome,
            "final_message": message,
            "packages_listed": [
                {"package_id": p, "title": None} for p in (packages or [])
            ],
            "searches_tried": [
                {"query": f"q{i}", "retrieval_quality": q}
                for i, q in enumerate(qualities or [])
            ],
        },
    }


def test_rate_is_over_surrenders_not_over_turns() -> None:
    summary = _guided_recovery(
        [
            _turn("a", outcome="answered", packages=["p1"], qualities=["ok"]),
            _turn("b", outcome="no_data", packages=["p1"], qualities=["ok"]),
            _turn("c", outcome="no_data", packages=[], qualities=["weak"]),
        ]
    )
    assert summary["turns"] == 3
    assert summary["surrenders"] == 2
    assert summary["exploitable_rate"] == 0.5
    assert summary["surrender_ids"] == ["b", "c"]
    assert summary["exploitable_ids"] == ["b"]


def test_both_conditions_are_required() -> None:
    summary = _guided_recovery(
        [
            # Listed a package, but nothing cleared the floor.
            _turn("a", outcome="no_data", packages=["p1"], qualities=["weak"]),
            # Good retrieval, but never opened anything.
            _turn("b", outcome="no_data", packages=[], qualities=["ok"]),
        ]
    )
    assert summary["surrenders_with_packages"] == 1
    assert summary["surrenders_with_ok_retrieval"] == 1
    assert summary["exploitable_rate"] == 0.0
    assert summary["verdict"] == "stop_and_reconsider"


def test_verdict_bands() -> None:
    def rate_of(exploitable: int, total: int) -> str:
        turns = [
            _turn(
                f"e{i}",
                outcome="no_data",
                packages=["p1"],
                qualities=["ok"],
            )
            for i in range(exploitable)
        ] + [
            _turn(f"n{i}", outcome="no_data")
            for i in range(total - exploitable)
        ]
        return str(_guided_recovery(turns)["verdict"])

    assert rate_of(5, 10) == "proceed"
    assert rate_of(4, 10) == "proceed"
    assert rate_of(3, 10) == "proceed_after_revisiting_gating"
    assert rate_of(2, 10) == "proceed_after_revisiting_gating"
    assert rate_of(1, 10) == "stop_and_reconsider"


def test_no_surrenders_is_not_a_zero_rate() -> None:
    summary = _guided_recovery(
        [_turn("a", outcome="answered", packages=["p1"], qualities=["ok"])]
    )
    assert summary["surrenders"] == 0
    assert summary["verdict"] == "no_surrenders"


def test_footer_presence_is_counted_on_surrenders_only() -> None:
    summary = _guided_recovery(
        [
            _turn(
                "a",
                outcome="no_data",
                packages=["p1"],
                qualities=["ok"],
                message="nothing found.\n\n**What I searched:** *D*",
            ),
            _turn("b", outcome="no_data", message="nothing found."),
            _turn(
                "c",
                outcome="answered",
                message="ok.\n\n**What I searched:** *D*",
            ),
        ]
    )
    assert summary["surrenders_with_footer"] == 1


def test_queries_are_captured_verbatim_for_every_turn() -> None:
    summary = _guided_recovery(
        [
            _turn("a", outcome="no_data", qualities=["ok", "weak"]),
            _turn("b", outcome="answered"),
        ]
    )
    assert summary["search_queries"] == [
        {"id": "a", "outcome": "no_data", "queries": ["q0", "q1"]},
        {"id": "b", "outcome": "answered", "queries": []},
    ]


def test_missing_turn_record_fields_do_not_crash() -> None:
    """The v1 loop emits no turn record, so every field reads None."""
    summary = _guided_recovery(
        [
            {
                "id": "a",
                "run": {
                    "outcome": None,
                    "final_message": "x",
                    "packages_listed": None,
                    "searches_tried": None,
                },
            }
        ]
    )
    assert summary["surrenders"] == 0
    assert summary["exploitable_rate"] == 0.0
