"""Dropping the preamble rows from a query that knows where they are.

The recovered header row is still a row. Its values are text, so
`SAFE_CAST(... AS FLOAT64)` yields NULL and it falls out of SUM and AVG
by itself — but not out of `COUNT(*)`, where it and the blank lines above
it inflate the answer by the preamble depth. Invisible on a table of
thousands; material on one with twenty rows, which is exactly the shape
these preambled statistical tables have.
"""
from __future__ import annotations

from semantic_enrich.core.sql_header_alias import apply_recovered_names

DOC = "doc-1"
OTHER = "doc-2"


def _sql(doc_ids: str = f"'{DOC}'", where_extra: str = "") -> str:
    return (
        "SELECT COUNT(*) AS n FROM `p.raw.rows` "
        f"WHERE document_id IN ({doc_ids}){where_extra} LIMIT 10"
    )


def test_preamble_rows_are_excluded_for_a_known_header() -> None:
    result = apply_recovered_names(
        _sql(), recovered_names={}, header_rows={DOC: 2}
    )
    assert f"NOT (document_id = '{DOC}' AND row_index <= 2)" in result.sql
    assert result.preamble_excluded == {DOC: 2}
    assert result.preamble_skipped is False


def test_a_document_with_no_known_header_row_is_unchanged() -> None:
    sql = _sql()
    result = apply_recovered_names(
        sql, recovered_names={}, header_rows={}
    )
    assert result.sql == sql
    assert result.preamble_excluded == {}


def test_a_header_on_row_zero_adds_no_predicate() -> None:
    """No preamble means nothing to exclude — and `row_index <= 0` would
    wrongly drop the first data row of a headerless read."""
    sql = _sql()
    result = apply_recovered_names(
        sql, recovered_names={}, header_rows={DOC: 0}
    )
    assert result.sql == sql
    assert result.preamble_excluded == {}


def test_a_document_not_inlined_contributes_no_predicate() -> None:
    result = apply_recovered_names(
        _sql(), recovered_names={}, header_rows={OTHER: 3}
    )
    assert "row_index" not in result.sql
    assert result.preamble_excluded == {}


def test_several_documents_get_their_own_depths() -> None:
    """Per-document predicates, because preamble depth differs per file
    and a bare `row_index > 2` would silently trim a document that has
    no preamble at all."""
    result = apply_recovered_names(
        _sql(f"'{DOC}', '{OTHER}'"),
        recovered_names={},
        header_rows={DOC: 2, OTHER: 4},
    )
    assert f"NOT (document_id = '{DOC}' AND row_index <= 2)" in result.sql
    assert f"NOT (document_id = '{OTHER}' AND row_index <= 4)" in result.sql
    assert result.preamble_excluded == {DOC: 2, OTHER: 4}


def test_the_predicate_lands_inside_the_existing_filter() -> None:
    result = apply_recovered_names(
        _sql(where_extra=" AND JSON_VALUE(row, '$.__col_1') IS NOT NULL"),
        recovered_names={},
        header_rows={DOC: 1},
    )
    # Appended right after the IN-list, so it is ANDed with everything
    # else and never lands after the LIMIT.
    assert result.sql.index("row_index") < result.sql.index("LIMIT")
    assert result.sql.index("row_index") < result.sql.index("IS NOT NULL")


def test_an_or_in_the_where_clause_skips_rather_than_guesses() -> None:
    """Appending after the IN-list is only safe in an AND-chain. Inside a
    disjunct it would change what the query means, so the pass declines
    and says it declined instead of returning a quietly inflated count."""
    sql = (
        "SELECT COUNT(*) FROM `p.raw.rows` WHERE "
        f"(document_id IN ('{DOC}') OR document_id IN ('{OTHER}')) LIMIT 10"
    )
    result = apply_recovered_names(
        sql, recovered_names={}, header_rows={DOC: 2}
    )
    assert result.sql == sql
    assert result.preamble_excluded == {}
    assert result.preamble_skipped is True


def test_unparseable_sql_skips_rather_than_guesses() -> None:
    sql = f"SELECT COUNT(*) FROM WHERE document_id IN ('{DOC}') ((("
    result = apply_recovered_names(
        sql, recovered_names={}, header_rows={DOC: 2}
    )
    assert result.preamble_excluded == {}
    assert result.preamble_skipped is True


def test_a_union_arm_split_gets_the_predicate_on_every_arm() -> None:
    """The per-document UNION ALL split is the sanctioned way to combine
    columns that do not co-occur. The predicate is written per document,
    so adding it to every arm is a tautology wherever it does not apply."""
    sql = (
        "SELECT COUNT(*) FROM `p.raw.rows` "
        f"WHERE document_id IN ('{DOC}') "
        "UNION ALL "
        "SELECT COUNT(*) FROM `p.raw.rows` "
        f"WHERE document_id IN ('{OTHER}') LIMIT 10"
    )
    result = apply_recovered_names(
        sql, recovered_names={}, header_rows={DOC: 2, OTHER: 1}
    )
    assert result.sql.count(f"NOT (document_id = '{DOC}'") == 2
    assert result.sql.count(f"NOT (document_id = '{OTHER}'") == 2


def test_a_hostile_document_id_is_never_interpolated() -> None:
    """Document ids come from the warehouse, but the predicate is built
    by string concatenation, so anything that is not checksum-shaped is
    dropped rather than embedded."""
    result = apply_recovered_names(
        _sql("'x'' OR 1=1 --'"),
        recovered_names={},
        header_rows={"x' OR 1=1 --": 2},
    )
    assert "OR 1=1" not in result.sql.replace("'x'' OR 1=1 --'", "")
    assert result.preamble_excluded == {}


def test_translation_and_exclusion_compose() -> None:
    result = apply_recovered_names(
        "SELECT SUM(SAFE_CAST(JSON_VALUE(row, '$.\"Total Amount ($000)\"') "
        "AS FLOAT64)) FROM `p.raw.rows` "
        f"WHERE document_id IN ('{DOC}') LIMIT 10",
        recovered_names={DOC: {"__col_2": "Total Amount ($000)"}},
        header_rows={DOC: 1},
    )
    assert "'$.__col_2'" in result.sql
    assert f"NOT (document_id = '{DOC}' AND row_index <= 1)" in result.sql
