"""Plan-hint selection and rendering: strictly answered-only, lexical
overlap thresholds, and the deterministic hint template."""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent.memory import (
    render_plan_hints,
    select_plan_hints,
)


def _record(
    *,
    question: str = "airplane spending by federal officials",
    gist: str = "airplane spending federal officials",
    outcome: str = "answered",
    columns: list[str] | None = None,
    title: str = "Travel expenses",
) -> dict[str, Any]:
    return {
        "v": 1,
        "question": question,
        "question_gist": gist,
        "outcome": outcome,
        "packages": [
            {"package_id": "proactive-disclosure-travel", "title": title}
        ],
        "columns_used": columns if columns is not None else ["Airfare"],
        "document_ids": ["d1", "d2"],
        "sql": "SELECT SUM(x) FROM r WHERE document_id IN ('d1') LIMIT 5",
    }


def _select(
    question: str, records: list[dict[str, Any]], **kw: Any
) -> list[dict[str, Any]]:
    kw.setdefault("max_hints", 2)
    kw.setdefault("min_overlap", 0.2)
    return select_plan_hints(
        question=question, turn_records=records, **kw
    )


def test_overlapping_answered_record_selected() -> None:
    selected = _select(
        "how much airplane spending by officials?", [_record()]
    )
    assert len(selected) == 1


def test_non_answered_outcomes_never_prime() -> None:
    for outcome in (
        "answered_with_caveat",
        "no_data",
        "clarified",
        "deflected",
        "error",
    ):
        assert (
            _select(
                "how much airplane spending by officials?",
                [_record(outcome=outcome)],
            )
            == []
        )


def test_below_threshold_not_selected() -> None:
    assert (
        _select("housing starts in ontario", [_record()]) == []
    )


def test_titles_and_columns_count_toward_overlap() -> None:
    # The question shares no gist tokens with the record's question,
    # but matches its package title + column vocabulary.
    record = _record(
        question="q1",
        gist="completely unrelated gist tokens here",
        columns=["Travel_Expenses"],
        title="Travel expenses",
    )
    selected = _select("travel expenses", [record], min_overlap=0.2)
    assert selected == [record]


def test_top_k_by_overlap() -> None:
    close = _record(gist="airplane spending federal officials")
    mid = _record(gist="airplane spending totals")
    far = _record(gist="airplane maintenance contracts logistics")
    selected = _select(
        "airplane spending federal officials",
        [far, mid, close],
        max_hints=2,
    )
    assert selected[0] == close
    assert len(selected) == 2


def test_empty_records_degrade_cleanly() -> None:
    assert _select("anything", []) == []


def test_identical_plans_deduplicated() -> None:
    # Replayed turns re-emit identical records; only one hint results.
    selected = _select(
        "how much airplane spending by officials?",
        [_record(), _record(), _record()],
    )
    assert len(selected) == 1


def test_render_template() -> None:
    text = render_plan_hints([_record()])
    assert text.startswith("Prior resolved plans from this conversation")
    assert "go straight to list_documents" in text
    assert "do NOT call search_datasets" in text
    assert '"airplane spending by federal officials"' in text
    assert 'proactive-disclosure-travel ("Travel expenses")' in text
    assert "columns [Airfare]" in text
    assert "document_ids [d1, d2]" in text
    assert "outcome: answered." in text
    # SQL shape has literals blanked.
    assert "'d1'" not in text.split("SQL shape:")[1]
