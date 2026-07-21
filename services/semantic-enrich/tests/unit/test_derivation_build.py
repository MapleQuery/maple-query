"""Deterministic derivation construction from executed SQL + result.

No I/O, no model call: build_derivation is a pure function of the SQL
string, the run_sql result payload, and the turn's LoopState.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.core.agent.derivation import Derivation, build_derivation
from semantic_enrich.core.agent_tools import LoopState


def _state(**meta: dict[str, Any]) -> LoopState:
    state = LoopState(conversation_id="c", turn_id="t", question="q")
    for name, m in meta.items():
        state.column_metadata[name] = m
    return state


def _one(**kwargs: Any) -> Derivation:
    """build_derivation returns one derivation per scalar-aggregate
    column; these single-column cases expect exactly one."""
    derivs = build_derivation(**kwargs)
    assert len(derivs) == 1
    return derivs[0]


_SCALAR_SUM = (
    "SELECT SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
    "FROM raw.rows WHERE document_id IN ('doc-1')"
)


def test_scalar_sum_captures_value_and_shape() -> None:
    state = _state(
        Amount={"semantic_type": "currency", "description": "Dollar amount."}
    )
    deriv = _one(
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
    deriv = _one(
        sql=sql,
        result={"row_count": 13, "rows": [{"province": "AB", "total": 5.0}]},
        state=_state(),
    )
    assert deriv.complete is True
    assert deriv.result_value is None
    assert "grouped_aggregate" in deriv.notes
    assert deriv.group_by_columns == ("province",)


def test_non_numeric_result_cell() -> None:
    deriv = _one(
        sql=_SCALAR_SUM,
        result={"row_count": 1, "rows": [{"total": "not a number"}]},
        state=_state(),
    )
    assert deriv.result_value is None
    assert "non_numeric_result" in deriv.notes


def test_string_numeric_result_is_parsed() -> None:
    # JSON_VALUE can surface numbers as strings; they must still parse.
    deriv = _one(
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
    deriv = _one(
        sql=sql, result={"row_count": 10, "rows": [{"a": "1"}]}, state=_state()
    )
    assert deriv.aggregation == "none"
    assert deriv.result_value is None
    assert "no_scalar_aggregate" in deriv.notes


def test_unparseable_sql_is_incomplete_never_raises() -> None:
    deriv = _one(
        sql="this is not sql (((", result={"row_count": 1, "rows": []}, state=_state()
    )
    assert deriv.complete is False
    assert deriv.result_value is None


def test_count_aggregation_units_are_count() -> None:
    sql = "SELECT COUNT(*) AS n FROM raw.rows WHERE document_id IN ('doc-1')"
    deriv = _one(
        sql=sql, result={"row_count": 1, "rows": [{"n": 1400}]}, state=_state()
    )
    assert deriv.aggregation == "COUNT"
    assert deriv.unit_scale == "count"
    assert deriv.result_value == 1400.0


def test_multi_scalar_query_yields_one_derivation_per_column() -> None:
    # The two-total case that showed NO trace before: each scalar
    # aggregate column now gets its own derivation with its own value.
    sql = (
        "SELECT "
        "SUM(SAFE_CAST(JSON_VALUE(r.row, '$.main_estimates') AS FLOAT64)) AS total_main, "
        "SUM(SAFE_CAST(JSON_VALUE(r.row, '$.to_date') AS FLOAT64)) AS total_to_date "
        "FROM raw.rows AS r WHERE r.document_id IN ('doc-1')"
    )
    derivs = build_derivation(
        sql=sql,
        result={
            "row_count": 1,
            "rows": [{"total_main": 450.42e9, "total_to_date": 489.84e9}],
        },
        state=_state(),
    )
    assert len(derivs) == 2
    by_label = {d.result_label: d for d in derivs}
    assert by_label["total_main"].result_value == 450.42e9
    assert by_label["total_main"].value_columns == ("main_estimates",)
    assert by_label["total_to_date"].result_value == 489.84e9
    assert by_label["total_to_date"].value_columns == ("to_date",)
    assert all(d.aggregation == "SUM" for d in derivs)


def test_count_and_sum_each_captured() -> None:
    sql = (
        "SELECT COUNT(*) AS n, "
        "SUM(CAST(JSON_VALUE(payload, '$.Amount') AS FLOAT64)) AS total "
        "FROM raw.rows WHERE document_id IN ('doc-1')"
    )
    derivs = build_derivation(
        sql=sql,
        result={"row_count": 1, "rows": [{"n": 1400, "total": 9.0}]},
        state=_state(),
    )
    by_label = {d.result_label: d for d in derivs}
    assert by_label["n"].aggregation == "COUNT"
    assert by_label["n"].unit_scale == "count"
    assert by_label["total"].aggregation == "SUM"
    assert by_label["total"].value_columns == ("Amount",)
