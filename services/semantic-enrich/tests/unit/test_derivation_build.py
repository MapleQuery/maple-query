"""Deterministic derivation construction from executed SQL + result.

No I/O, no model call: build_derivation is a pure function of the SQL
string, the run_sql result payload, and the turn's LoopState.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent.derivation import build_derivation
from semantic_enrich.core.agent_tools import LoopState


def _state(**meta: dict[str, Any]) -> LoopState:
    state = LoopState(conversation_id="c", turn_id="t", question="q")
    for name, m in meta.items():
        state.column_metadata[name] = m
    return state


_SCALAR_SUM = (
    "SELECT SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
    "FROM raw.rows WHERE document_id IN ('doc-1')"
)


def test_scalar_sum_captures_value_and_shape() -> None:
    state = _state(
        Amount={"semantic_type": "currency", "description": "Dollar amount."}
    )
    deriv = build_derivation(
        sql=_SCALAR_SUM,
        result={"row_count": 1, "rows": [{"total": 4200.5}]},
        state=state,
    )
    assert deriv.complete is True
    assert deriv.aggregation == "SUM"
    assert deriv.value_columns == ("Amount",)
    assert deriv.group_by_columns == ()
    assert deriv.row_count == 1
    assert deriv.result_value == 4200.5
    assert deriv.result_label == "total"
    # currency column, no scale cue -> unknown (not silently dollars).
    assert deriv.unit_scale == "unknown"
    # literals masked out of the shapes.
    assert "doc-1" not in deriv.sql_shape
    assert "document_id" in deriv.predicate_shape


def test_grouped_aggregate_has_no_scalar_value() -> None:
    sql = (
        "SELECT JSON_VALUE(payload, '$.province') AS province, "
        "SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
        "FROM raw.rows WHERE document_id IN ('doc-1') GROUP BY province"
    )
    deriv = build_derivation(
        sql=sql,
        result={"row_count": 13, "rows": [{"province": "AB", "total": 5.0}]},
        state=_state(),
    )
    assert deriv.complete is True
    assert deriv.result_value is None
    assert "grouped_aggregate" in deriv.notes
    assert deriv.group_by_columns == ("province",)


def test_non_numeric_result_cell() -> None:
    deriv = build_derivation(
        sql=_SCALAR_SUM,
        result={"row_count": 1, "rows": [{"total": "not a number"}]},
        state=_state(),
    )
    assert deriv.result_value is None
    assert "non_numeric_result" in deriv.notes


def test_string_numeric_result_is_parsed() -> None:
    # JSON_VALUE can surface numbers as strings; they must still parse.
    deriv = build_derivation(
        sql=_SCALAR_SUM,
        result={"row_count": 1, "rows": [{"total": "1,234.50"}]},
        state=_state(),
    )
    assert deriv.result_value == 1234.5


def test_no_aggregate_query() -> None:
    sql = (
        "SELECT JSON_VALUE(payload, '$.Amount') AS a FROM raw.rows "
        "WHERE document_id IN ('doc-1') LIMIT 10"
    )
    deriv = build_derivation(
        sql=sql, result={"row_count": 10, "rows": [{"a": "1"}]}, state=_state()
    )
    assert deriv.aggregation == "none"
    assert deriv.result_value is None
    assert "no_scalar_aggregate" in deriv.notes


def test_unparseable_sql_is_incomplete_never_raises() -> None:
    deriv = build_derivation(
        sql="this is not sql (((", result={"row_count": 1, "rows": []}, state=_state()
    )
    assert deriv.complete is False
    assert deriv.result_value is None


def test_count_aggregation_units_are_count() -> None:
    sql = "SELECT COUNT(*) AS n FROM raw.rows WHERE document_id IN ('doc-1')"
    deriv = build_derivation(
        sql=sql, result={"row_count": 1, "rows": [{"n": 1400}]}, state=_state()
    )
    assert deriv.aggregation == "COUNT"
    assert deriv.unit_scale == "count"
    assert deriv.result_value == 1400.0


def test_sum_preferred_over_count_when_both_present() -> None:
    sql = (
        "SELECT COUNT(*) AS n, "
        "SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
        "FROM raw.rows WHERE document_id IN ('doc-1')"
    )
    deriv = build_derivation(
        sql=sql,
        result={"row_count": 1, "rows": [{"n": 1400, "total": 9.0}]},
        state=_state(),
    )
    # SUM is the money-bearing aggregate; it must win for unit/value.
    assert deriv.aggregation == "SUM"
    assert deriv.value_columns == ("Amount",)
